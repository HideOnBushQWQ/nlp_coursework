"""
标签编码器模块
"""

from typing import List, Dict


class LabelEncoder:
    """标签编码器：标签 <-> ID 转换"""

    def __init__(self, labels: List[str] = None):
        """
        初始化标签编码器

        Args:
            labels: 标签列表，如 ['O', 'B-PER', 'I-PER', ...]
                   如果为None，则使用默认标签
        """
        if labels is None:
            # 默认标签：O + 4类实体的B/I
            labels = ['O']
            for entity_type in ['PER', 'LOC', 'ORG', 'MISC']:
                labels.extend([f'B-{entity_type}', f'I-{entity_type}'])

        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def encode(self, label: str) -> int:
        """
        标签 -> ID

        Args:
            label: 标签字符串

        Returns:
            标签ID
        """
        return self.label2id.get(label, 0)  # 默认返回O的ID

    def encode_batch(self, labels: List[str]) -> List[int]:
        """
        批量编码

        Args:
            labels: 标签列表

        Returns:
            标签ID列表
        """
        return [self.encode(label) for label in labels]

    def decode(self, label_id: int) -> str:
        """
        ID -> 标签

        Args:
            label_id: 标签ID

        Returns:
            标签字符串
        """
        return self.id2label.get(label_id, 'O')

    def decode_batch(self, label_ids: List[int]) -> List[str]:
        """
        批量解码

        Args:
            label_ids: 标签ID列表

        Returns:
            标签字符串列表
        """
        return [self.decode(lid) for lid in label_ids]

    @property
    def num_labels(self) -> int:
        """返回标签数量"""
        return len(self.label2id)

    @property
    def labels(self) -> List[str]:
        """返回所有标签"""
        return list(self.label2id.keys())

    def save(self, path: str):
        """
        保存标签编码器

        Args:
            path: 保存路径
        """
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': {str(k): v for k, v in self.id2label.items()}
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'LabelEncoder':
        """
        加载标签编码器

        Args:
            path: 文件路径

        Returns:
            LabelEncoder实例
        """
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        encoder = cls(labels=list(data['label2id'].keys()))
        return encoder
