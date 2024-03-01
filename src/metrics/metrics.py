from typing import List, Dict
import os
import importlib
from abc import ABC, abstractmethod
import inspect
import shutil

import numpy as np

from utils.decoding import decode
from datasets import load_metric as hf_load_metric
from huggingface_hub import hf_hub_download


class Metric(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs

        self.prefix = os.path.splitext(os.path.basename(inspect.getfile(self.__class__)))[0]
        self.requires_decoded = False

    def __call__(self, id_to_pred, id_to_labels, is_decoded=False):
        if self.requires_decoded and is_decoded is False:
            id_to_pred = self._decode(id_to_pred)
            id_to_labels = self._decode(id_to_labels)
        return self._compute_metrics(id_to_pred, id_to_labels)

    @abstractmethod
    def _compute_metrics(self, id_to_pred, id_to_labels) -> Dict[str, float]:
        return

    def _decode(self, id_to_something):
        tokenizer = self._kwargs.get("tokenizer")
        data_args = self._kwargs.get("data_args")
        return decode(id_to_something, tokenizer, data_args)


class MetricCollection(Metric):
    def __init__(self, metrics: List[Metric], **kwargs):
        super().__init__(**kwargs)
        self._metrics = metrics

    def __call__(self, id_to_pred, id_to_labels):
        return self._compute_metrics(id_to_pred, id_to_labels)

    def _compute_metrics(self, id_to_pred, id_to_labels):
        results = {}

        id_to_pred_decoded = None
        id_to_labels_decoded = None
        for metric in self._metrics:
            metric_prefix = f"{metric.prefix}/" if metric.prefix else ""
            if metric.requires_decoded:
                if id_to_pred_decoded is None:
                    id_to_pred_decoded = self._decode(id_to_pred)
                if id_to_labels_decoded is None:
                    id_to_labels_decoded = self._decode(id_to_labels)

                result = metric(id_to_pred_decoded, id_to_labels_decoded, is_decoded=True)
            else:
                result = metric(id_to_pred, id_to_labels)

            results.update({f"{metric_prefix}{k}": np.mean(v) if type(v) is list else v for k, v in result.items() if type(v) is not str})

        results["num_predicted"] = len(id_to_pred)
        results["mean_prediction_length_characters"] = np.mean([len(pred) for pred in id_to_pred_decoded.values()])

        elem = next(iter(id_to_pred.values()))
        if not ((isinstance(elem, list) and isinstance(elem[0], str)) or isinstance(elem, str)):
            tokenizer = self._kwargs["tokenizer"]
            results["mean_prediction_length_tokens"] = np.mean(
                [np.count_nonzero(np.array(pred) != tokenizer.pad_token_id) for pred in id_to_pred.values()]
            )  # includes BOS/EOS tokens

        results = {key: round(value, 4) for key, value in results.items()}
        return results


def load_metric(paths: List[str], **kwargs):
    if paths is None or len(paths) == 0:
        return None
    if isinstance(paths, str):
        paths = [paths]
    else:
        paths = [path for path in paths]

    metric_cls_list = []

    scrolls_custom_metrics = []
    to_remove = []
    for i, path in enumerate(paths):
        if not os.path.isfile(path):
            scrolls_custom_metrics.append(path)
            to_remove.append(i)
    for i in sorted(to_remove, reverse=True):
        del paths[i]
    if len(scrolls_custom_metrics) > 0:
        scrolls_custom_metrics.insert(0, "")  # In order to have an identifying comma in the beginning
        metric_cls_list.append(ScrollsWrapper(",".join(scrolls_custom_metrics), **kwargs))

    for path in paths:
        path = path.strip()
        if len(path) == 0:
            continue
        if os.path.isfile(path) is False:
            path = os.path.join("src", "metrics", f"{path}.py")

        module = path[:-3].replace(os.sep, ".")

        metric_cls = import_main_class(module)
        metric_cls_list.append(metric_cls(**kwargs))

    return MetricCollection(metric_cls_list, **kwargs)


# Modified from datasets.load
def import_main_class(module_path):
    """Import a module at module_path and return its main class"""
    module = importlib.import_module(module_path)

    main_cls_type = Metric

    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, main_cls_type):
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            break

    return module_main_cls


class ScrollsWrapper(Metric):
    def __init__(self, comma_separated_metric_names, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix = None

        self._metric = hf_load_metric(download_metric(), comma_separated_metric_names, keep_in_memory=True)

        self.requires_decoded = True

    def _compute_metrics(self, id_to_pred, id_to_labels) -> Dict[str, float]:
        return self._metric.compute(**self._metric.convert_from_map_format(id_to_pred, id_to_labels))

class HFMetricWrapper(Metric):
    def __init__(self, metric_name, **kwargs) -> None:
        super().__init__(**kwargs)
        self._metric = hf_load_metric(metric_name)
        self.kwargs = HFMetricWrapper.metric_specific_kwargs.get(metric_name, {})
        self.requires_decoded = True
        self.prefix = metric_name
        self.requires_decoded = True

    def _compute_metrics(self, id_to_pred, id_to_labels) -> Dict[str, float]:
        return self._metric.compute(**self.convert_from_map_format(id_to_pred, id_to_labels), **self.kwargs)

    def convert_from_map_format(self, id_to_pred, id_to_labels):
        index_to_id = list(id_to_pred.keys())
        predictions = [id_to_pred[id_] for id_ in index_to_id]
        references = [id_to_labels[id_] for id_ in index_to_id]
        return {"predictions": predictions, "references": references}

    metric_specific_kwargs = {
        'bertscore': {
            # 'model_type': 'microsoft/deberta-large-mnli' or the larger 'microsoft/deberta-xlarge-mnli'
            'model_type': 'facebook/bart-large-mnli', # has context window of 1024,
            'num_layers': 11 # according to: https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0
        }
    }


def download_metric():
    # here we load the custom metrics
    scrolls_metric_path = hf_hub_download(repo_id="tau/scrolls", filename="metrics/scrolls.py", repo_type='dataset')
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path) + os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path
