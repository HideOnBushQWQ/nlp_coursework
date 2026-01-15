"""
评估指标模块
"""

from typing import List, Dict, Set, Tuple
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: List[List[int]],
    labels: List[List[int]],
    label_encoder
) -> Dict[str, float]:
    """
    计算NER评估指标

    Args:
        predictions: 预测标签ID列表
        labels: 真实标签ID列表
        label_encoder: 标签编码器

    Returns:
        评估指标字典
    """
    # 转换为标签字符串
    pred_labels = [
        [label_encoder.decode(p) for p in pred]
        for pred in predictions
    ]
    true_labels = [
        [label_encoder.decode(l) for l in label]
        for label in labels
    ]

    # 计算指标
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_detailed_metrics(
    predictions: List[List[int]],
    labels: List[List[int]],
    label_encoder
) -> Dict:
    """
    计算详细的评估指标，包括每个实体类型的指标

    Args:
        predictions: 预测标签ID列表
        labels: 真实标签ID列表
        label_encoder: 标签编码器

    Returns:
        详细的评估指标字典
    """
    # 转换为标签字符串
    pred_labels = [
        [label_encoder.decode(p) for p in pred]
        for pred in predictions
    ]
    true_labels = [
        [label_encoder.decode(l) for l in label]
        for label in labels
    ]

    # 整体指标
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    # 详细报告
    report = classification_report(
        true_labels,
        pred_labels,
        output_dict=True,
        zero_division=0
    )

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }


class NERMetrics:
    """NER评估指标计算器"""

    def __init__(self, label_encoder):
        """
        初始化评估器

        Args:
            label_encoder: 标签编码器
        """
        self.label_encoder = label_encoder

    def compute(
        self,
        predictions: List[List[int]],
        labels: List[List[int]]
    ) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            predictions: 预测标签ID列表
            labels: 真实标签ID列表

        Returns:
            评估指标字典
        """
        return compute_detailed_metrics(predictions, labels, self.label_encoder)

    def entity_level_accuracy(
        self,
        predictions: List[List[int]],
        labels: List[List[int]]
    ) -> float:
        """
        计算实体级别准确率

        Args:
            predictions: 预测标签ID列表
            labels: 真实标签ID列表

        Returns:
            实体级别准确率
        """
        pred_entities = self._extract_entities(predictions)
        true_entities = self._extract_entities(labels)

        total_true = sum(len(entities) for entities in true_entities)
        if total_true == 0:
            return 0.0

        correct = 0
        for pred_ents, true_ents in zip(pred_entities, true_entities):
            correct += len(pred_ents & true_ents)

        return correct / total_true

    def _extract_entities(
        self,
        label_sequences: List[List[int]]
    ) -> List[Set[Tuple[str, int, int]]]:
        """
        从标签序列中提取实体

        Args:
            label_sequences: 标签ID序列列表

        Returns:
            每个句子的实体集合：{(entity_type, start, end), ...}
        """
        all_entities = []

        for labels in label_sequences:
            entities = set()
            entity_start = None
            entity_type = None

            for i, label_id in enumerate(labels):
                label = self.label_encoder.decode(label_id)

                if label.startswith('B-'):
                    # 保存上一个实体
                    if entity_start is not None:
                        entities.add((entity_type, entity_start, i))
                    # 开始新实体
                    entity_type = label[2:]
                    entity_start = i
                elif label.startswith('I-'):
                    # 继续当前实体
                    if entity_start is None or entity_type != label[2:]:
                        # 错误：I-tag出现但没有对应的B-tag，或类型不匹配
                        entity_type = label[2:]
                        entity_start = i
                else:  # 'O'
                    # 结束当前实体
                    if entity_start is not None:
                        entities.add((entity_type, entity_start, i))
                        entity_start = None
                        entity_type = None

            # 处理句子结尾的实体
            if entity_start is not None:
                entities.add((entity_type, entity_start, len(labels)))

            all_entities.append(entities)

        return all_entities

    def confusion_matrix_data(
        self,
        predictions: List[List[int]],
        labels: List[List[int]]
    ) -> Dict:
        """
        生成混淆矩阵数据

        Args:
            predictions: 预测标签ID列表
            labels: 真实标签ID列表

        Returns:
            混淆矩阵数据
        """
        from collections import defaultdict

        confusion = defaultdict(lambda: defaultdict(int))

        for pred_seq, true_seq in zip(predictions, labels):
            for pred_id, true_id in zip(pred_seq, true_seq):
                pred_label = self.label_encoder.decode(pred_id)
                true_label = self.label_encoder.decode(true_id)
                confusion[true_label][pred_label] += 1

        return dict(confusion)
