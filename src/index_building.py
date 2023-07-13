import faiss
import faiss.contrib.torch_utils
import time
import logging

import torch
import numpy as np

code_size = 64

class DatastoreBatch():
    def __init__(self, dim, batch_size, flat_index=False, gpu_index=False, verbose=False, index_device=None) -> None:
        self.indices = []
        self.batch_size = batch_size
        self.device = index_device if index_device is not None else torch.device('cuda' if gpu_index else 'cpu')
        for i in range(batch_size):
            self.indices.append(Datastore(dim, use_flat_index=flat_index, gpu_index=gpu_index, verbose=verbose, device=self.device))
    
    def move_to_gpu(self):
        for i in range(self.batch_size):
            self.indices[i].move_to_gpu()

    def add_keys(self, keys, num_keys_to_add_at_a_time=100000):
        for i in range(self.batch_size):
            self.indices[i].add_keys(keys[i], num_keys_to_add_at_a_time)
        
    def train_index(self, keys):
        for index, example_keys in zip(self.indices, keys):
            index.train_index(example_keys)
    
    def search(self, queries, k):
        found_scores, found_values = [], []
        for i in range(self.batch_size):
            scores, values = self.indices[i].search(queries[i], k)
            found_scores.append(scores)
            found_values.append(values)
        return torch.stack(found_scores, dim=0), torch.stack(found_values, dim=0)

    def search_and_reconstruct(self, queries, k):
        found_scores, found_values = [], []
        found_vectors = []
        for i in range(self.batch_size):
            scores, values, vectors = self.indices[i].search_and_reconstruct(queries[i], k)
            found_scores.append(scores)
            found_values.append(values)
            found_vectors.append(vectors)     
        return torch.stack(found_scores, dim=0), torch.stack(found_values, dim=0), torch.stack(found_vectors, dim=0)

class Datastore():
    def __init__(self, dim, use_flat_index=False, gpu_index=False, verbose=False, device=None) -> None:
        self.dimension = dim
        self.device = device if device is not None else torch.device('cuda' if gpu_index else 'cpu')
        self.logger = logging.getLogger('index_building')
        self.logger.setLevel(20)
        self.use_flat_index = use_flat_index
        self.gpu_index = gpu_index

        # Initialize faiss index
        # TODO: is preprocessing efficient enough to spend time on?
        if not use_flat_index:
            self.index = faiss.IndexFlatIP(self.dimension) # inner product index because we use IP attention
        
        # need to wrap in index ID map to enable add_with_ids 
        # self.index = faiss.IndexIDMap(self.index) 

        self.index_size = 0
        # if self.gpu_index:
        #     self.move_to_gpu()
        
    def move_to_gpu(self):
        if self.use_flat_index:
            # self.keys = self.keys.to(self.device)
            return
        else:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device.index, self.index, co)
    
    def train_index(self, keys):
        if self.use_flat_index:
            self.add_keys(keys=keys, index_is_trained=True)
        else:
            keys = keys.cpu().float()
            ncentroids = int(keys.shape[0] / 128)
            self.index = faiss.IndexIVFPQ(self.index, self.dimension,
                ncentroids, code_size, 8)
            self.index.nprobe = min(32, ncentroids)
            # if not self.gpu_index:
            #     keys = keys.cpu()

            self.logger.info('Training index')
            start_time = time.time()
            self.index.train(keys)
            self.logger.info(f'Training took {time.time() - start_time} s')
            self.add_keys(keys=keys, index_is_trained=True)
            # self.keys = None
            if self.gpu_index:
                self.move_to_gpu()

    def add_keys(self, keys, num_keys_to_add_at_a_time=1000000, index_is_trained=False):
        self.keys = keys
        if not self.use_flat_index and index_is_trained:
            start = 0
            while start < keys.shape[0]:
                end = min(len(keys), start + num_keys_to_add_at_a_time)
                to_add = keys[start:end]
                # if not self.gpu_index:
                #     to_add = to_add.cpu()
                # self.index.add_with_ids(to_add, torch.arange(start+self.index_size, end+self.index_size))
                self.index.add(to_add)
                self.index_size += end - start
                start += end
                if (start % 1000000) == 0:
                    self.logger.info(f'Added {start} tokens so far')
        # else:
        #     self.keys.append(keys)

        # self.logger.info(f'Adding total {start} keys')
        # self.logger.info(f'Adding took {time.time() - start_time} s')

    def search_and_reconstruct(self, queries, k):
        if len(queries.shape) == 1: # searching for only 1 vector, add one extra dim
            self.logger.info("Searching for a single vector; unsqueezing")
            queries = queries.unsqueeze(0)
        # self.logger.info("Searching with reconstruct")
        assert queries.shape[-1] == self.dimension # query vectors are same shape as "key" vectors
        scores, values, vectors = self.index.index.search_and_reconstruct(queries.cpu().detach(), k)
        # self.logger.info("Searching done")
        return scores, values, vectors
    
    def search(self, queries, k):
        # model_device = queries.device
        # model_dtype = queries.dtype
        if len(queries.shape) == 1: # searching for only 1 vector, add one extra dim
            self.logger.info("Searching for a single vector; unsqueezing")
            queries = queries.unsqueeze(0)
        assert queries.shape[-1] == self.dimension # query vectors are same shape as "key" vectors
        # if not self.gpu_index:
        #     queries = queries.cpu()
        # else:
        #     queries = queries.to(self.device)
        if self.use_flat_index:
            if self.gpu_index:
                scores, values = faiss.knn_gpu(faiss.StandardGpuResources(), queries, self.keys, k, 
                    metric=faiss.METRIC_INNER_PRODUCT, device=self.device.index)
            else:
                scores, values = faiss.knn(queries, self.keys, k, metric=faiss.METRIC_INNER_PRODUCT)
                scores = torch.from_numpy(scores).to(queries.dtype)
                values = torch.from_numpy(values) #.to(model_dtype)
        else:
            scores, values = self.index.search(queries.float(), k)
        
        # avoid returning -1 as a value
        # TODO: get a handle on the attention mask and mask the values that were -1
        values = torch.where(torch.logical_or(values < 0, values >= self.keys.shape[0]), torch.zeros_like(values), values)
        # self.logger.info("Searching done")
        # return scores.to(model_dtype).to(model_device), values.to(model_device)
        return scores, values

    
    