import json
import os
import evaluate
import numpy as np

from typing import Optional, Literal, Dict, Any

import torch
from numpy import floating
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss
from transformers import EvalPrediction


class MetricsAtK:

    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray, k: Optional[int] = None):
        y_true = np.nan_to_num(np.asarray(y_true, dtype=np.float32), nan=0.0)
        y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float32), nan=0.0)
        n_samples = y_true.shape[0]
        self.mean_r_precision = 0
        self.mean_recall = 0
        self.mean_precision = 0
        self.ndcg = 0
        self.k = k
        self.r_precisions = np.zeros(n_samples)
        self.recalls = np.zeros(n_samples)
        self.precisions = np.zeros(n_samples)
        self.ndcgs = np.zeros(n_samples)

        for i in range(n_samples):
            row_true = y_true[i]
            row_prob = y_prob[i]
            total_relevant = int(np.sum(row_true))
            effective_k = total_relevant if self.k is None else min(int(self.k), row_prob.shape[0])
            total_relevant_at_k = total_relevant if self.k is None else min(total_relevant, effective_k)

            if effective_k > 0:
                top_k_indices = np.argsort(row_prob)[::-1][:effective_k]
                relevant_in_k = float(np.sum(row_true[top_k_indices]))
                dcg = self._dcg_at_k(row_true[top_k_indices])
            else:
                relevant_in_k = 0.0
                dcg = 0.0

            self.r_precisions[i] = relevant_in_k / total_relevant_at_k if total_relevant_at_k > 0 else 0.0
            self.recalls[i] = relevant_in_k / total_relevant if total_relevant > 0 else 0.0
            self.precisions[i] = relevant_in_k / effective_k if effective_k > 0 else 0.0
            self.ndcgs[i] = dcg / self._ideal_dcg_at_k(total_relevant_at_k) if total_relevant_at_k > 0 else 0.0

        self.mean_r_precision = np.mean(self.r_precisions)
        self.mean_recall = np.mean(self.recalls)
        self.mean_precision = np.mean(self.precisions)
        self.ndcg = np.mean(self.ndcgs)

    @staticmethod
    def _dcg_at_k(relevance: np.ndarray) -> float:
        discounts = np.log2(np.arange(2, relevance.shape[0] + 2, dtype=np.float32))
        return float(np.sum(relevance / discounts))

    @classmethod
    def _ideal_dcg_at_k(cls, total_relevant_at_k: int) -> float:
        if total_relevant_at_k <= 0:
            return 0.0
        ideal_relevance = np.ones(total_relevant_at_k, dtype=np.float32)
        return cls._dcg_at_k(ideal_relevance)

    def todict(self, prefix: str = '') -> dict[str, floating[Any] | Any]:
        suffix = f'@{self.k}' if self.k is not None else ''
        return {
            f'{prefix}r-p{suffix}': self.mean_r_precision,
            f'{prefix}p{suffix}': self.mean_precision,
            f'{prefix}r{suffix}': self.mean_recall,
            f'{prefix}ndcg{suffix}': self.ndcg,
        }


class TokenClassificationMetrics:
    def __init__(self, id2label, ignore_index=-100):
        self.id2label = id2label
        self.ignore_index = ignore_index
        self.log_epochs = []
        self.seqeval = evaluate.load("seqeval")

    def align_predictions(self, predictions, label_ids):
        preds = np.argmax(predictions, axis=-1)

        true_labels = []
        true_preds = []

        for pred_seq, label_seq in zip(preds, label_ids):
            seq_true = []
            seq_pred = []
            for p, l in zip(pred_seq, label_seq):
                if l == self.ignore_index:
                    continue
                seq_true.append(self.id2label[int(l)])
                seq_pred.append(self.id2label[int(p)])
            #if seq_true != seq_pred:
            #    print(f"Prediction mismatch: true={seq_true}, pred={seq_pred}")
            true_labels.append(seq_true)
            true_preds.append(seq_pred)

        return true_preds, true_labels

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        preds_list, labels_list = self.align_predictions(predictions, labels)
        results = self.seqeval.compute(
            predictions=preds_list,
            references=labels_list
        )
        metrics = {
            "p": results["overall_precision"],
            "r": results["overall_recall"],
            "f1": results["overall_f1"],
            "acc": results["overall_accuracy"],
        }
        for label, label_metrics in results.items():
            if label.startswith("overall"):
                continue
            if not isinstance(label_metrics, dict):
                continue
            if "precision" not in label_metrics:
                continue
            metrics[f"label.{label}.p"] = label_metrics["precision"]
            metrics[f"label.{label}.r"] = label_metrics["recall"]
            metrics[f"label.{label}.f1"] = label_metrics["f1"]
        self.log_epochs.append(metrics)
        return metrics


class Metrics:

    def __init__(self, model_name: str, prob_type: Literal['multilabel', 'multiclass', 'binary'] = 'multilabel',
                 avg_k: Optional[int] = None):
        self.log_epochs = []
        self.prob_type = prob_type
        self.model_name = model_name
        self.k_values = [1, 3, 5, 7, 9]
        self.avg_k = avg_k

    def compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, prefix: str = '', prob_threshold: float = 0.5):
        if self.prob_type == 'multilabel':
            y_pred = (y_prob > prob_threshold).astype(np.float32)
        else:
            y_pred = np.argmax(y_prob, axis=-1)

        # There was an issue where we had a single TP prediction among 400 samples with 1600 labels
        num_required_predictions = y_pred.shape[0] / 100  # at least 1% of the samples should have predictions
        num_required_predictions = 2 if num_required_predictions < 2 else num_required_predictions
        num_predicted = len(np.nonzero(y_pred)[0])

        metric = {}
        for average_type in ['micro', 'macro', 'weighted']:
            if self.prob_type == 'binary' and not average_type == 'macro':
                continue
            p = precision_score(y_true, y_pred, average=average_type)
            r = recall_score(y_true, y_pred, average=average_type)
            f1 = f1_score(y_true, y_pred, average=average_type)
            if p > 0.99 and num_predicted <= num_required_predictions:
                p = 0.0
                f1 = 0.0
                r = 0.0
            metric[f'{prefix}{average_type}.f1'] = f1
            metric[f'{prefix}{average_type}.p'] = p
            metric[f'{prefix}{average_type}.r'] = r

        acc = accuracy_score(y_true, y_pred)
        if num_predicted <= num_required_predictions:
            acc = 0.0
        metric[f'{prefix}acc'] = acc
        if self.prob_type == 'multilabel':
            for k in self.k_values:
                metric = metric | MetricsAtK(y_true, y_prob, k).todict(prefix)
            metric = metric | MetricsAtK(y_true, y_prob).todict(prefix)
            metric[f'{prefix}hamming_loss'] = hamming_loss(y_true, y_pred)
        self.log_epochs.append(metric)
        return metric

    def preprocess_logits(self, logits: torch.Tensor, _: torch.Tensor):
        if isinstance(logits, tuple):
            logits = logits[0]
        if self.prob_type == 'multilabel':
            prob = torch.sigmoid(logits)
        else:
            prob = torch.softmax(logits, dim=-1)
        return prob

    def __call__(self, eval_pred: EvalPrediction):
        y_true = eval_pred.label_ids
        y_prob = eval_pred.predictions
        return self.compute_metrics(y_true, y_prob)


class MultilabelSequenceMetrics(Metrics):

    def __init__(self, model_name: str, avg_k: Optional[int] = None):
        super().__init__(model_name=model_name, prob_type='multilabel', avg_k=avg_k)


# noinspection DuplicatedCode
def r_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: Optional[int]):
    """
    Compute R-Precision@K for multiple samples.

    Args:
    y_true: 2D array of true relevance labels, shape (n_samples, n_labels)
    y_pred: 2D array of predicted scores or probabilities, shape (n_samples, n_labels)
    k: The number of top items to consider for Recall@K

    Returns:
    Array of Recall@K scores for each sample and the mean Recall@K.
    """
    metrics = MetricsAtK(y_true, y_pred, k)
    return metrics.mean_r_precision, metrics.r_precisions
