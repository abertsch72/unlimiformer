import faiss
import faiss.contrib.torch_utils
import time
import logging

import torch
import numpy as np

code_size = 64

class DatastoreBatch():
    def __init__(self, dim, batch_size, flat_index=False, gpu_index=False, verbose=False) -> None:
        self.indices = []
        self.batch_size = batch_size
        for i in range(batch_size):
            self.indices.append(Datastore(dim, use_flat_index=flat_index, gpu_index=gpu_index, verbose=verbose))
    
    def move_to_gpu(self):
        for i in range(self.batch_size):
            self.indices[i].move_to_gpu()

    def add_keys(self, keys, num_keys_to_add_at_a_time=100000):
        for i in range(self.batch_size):
            self.indices[i].add_keys(keys[i], num_keys_to_add_at_a_time)
        
    def train_index(self):
        for index in self.indices:
            index.train_index()
    
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
    def __init__(self, dim, use_flat_index=False, gpu_index=False, verbose=False) -> None:
        self.dimension = dim
        self.logger = logging.getLogger('index_building')
        self.logger.setLevel(20)
        self.use_flat_index = use_flat_index
        self.gpu_index = gpu_index
        self.keys = []

        # Initialize faiss index
        # TODO: is preprocessing efficient enough to spend time on?
        self.index = faiss.IndexFlatIP(self.dimension) # inner product index because we use IP attention
        
        # need to wrap in index ID map to enable add_with_ids 
        # self.index = faiss.IndexIDMap(self.index) 

        self.index_size = 0
        if self.gpu_index:
            self.move_to_gpu()
        
    def move_to_gpu(self):
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index, co)
    
    def train_index(self):
        if self.use_flat_index:
            return
        self.keys = torch.cat(self.keys, axis=0)

        ncentroids = int(self.keys.shape[0] / 128)
        self.index = faiss.IndexIVFPQ(self.index, self.dimension,
            ncentroids, code_size, 8)
        self.index.nprobe = min(32, ncentroids)
        if self.gpu_index:
            self.move_to_gpu()
        else:
            self.keys = self.keys.cpu()

        self.logger.info('Training index')
        start_time = time.time()
        self.index.train(self.keys)
        self.logger.info(f'Training took {time.time() - start_time} s')
        self.add_keys(keys=self.keys, index_is_trained=True)

    def add_keys(self, keys, num_keys_to_add_at_a_time=1000000, index_is_trained=False):
        if self.use_flat_index or index_is_trained:
            start = 0
            while start < keys.shape[0]:
                end = min(len(keys), start + num_keys_to_add_at_a_time)
                to_add = keys[start:end]
                if not self.gpu_index:
                    to_add = to_add.cpu()
                # self.index.add_with_ids(to_add, torch.arange(start+self.index_size, end+self.index_size))
                self.index.add(to_add)
                self.index_size += end - start
                start += end
                if (start % 1000000) == 0:
                    self.logger.info(f'Added {start} tokens so far')
        else:
            self.keys.append(keys)

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
        if len(queries.shape) == 1: # searching for only 1 vector, add one extra dim
            self.logger.info("Searching for a single vector; unsqueezing")
            queries = queries.unsqueeze(0)
        assert queries.shape[-1] == self.dimension # query vectors are same shape as "key" vectors
        if not self.gpu_index:
            queries = queries.cpu()
        scores, values = self.index.search(queries, k)
        # avoid returning -1 as a value
        values[values == -1] = 0
        # self.logger.info("Searching done")
        return scores, values

    
    