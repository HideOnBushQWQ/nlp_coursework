"""
数据集加载器模块
"""

from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class CoNLLDatasetLoader:
    """CoNLL格式数据集加载器（适用于CoNLL-2003和MSRA）"""

    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        """
        初始化数据加载器

        Args:
            file_path: 数据文件路径
            encoding: 文件编码
        """
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> Tuple[List[List[str]], List[List[str]]]:
        """
        加载数据集

        Returns:
            sentences: 句子列表，每个句子是词/字的列表
            labels: 标签列表，每个标签序列对应一个句子
        """
        sentences = []
        labels = []

        current_tokens = []
        current_labels = []

        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # 空行表示句子结束
                    if not line:
                        if current_tokens:
                            sentences.append(current_tokens)
                            labels.append(current_labels)
                            current_tokens = []
                            current_labels = []
                        continue

                    # 跳过注释行（以#或-DOCSTART-开头）
                    if line.startswith('#') or line.startswith('-DOCSTART-'):
                        continue

                    # 解析token和label
                    try:
                        token, label = self._parse_line(line)
                        current_tokens.append(token)
                        current_labels.append(label)
                    except ValueError as e:
                        logger.warning(f"Line {line_num}: {e} - Skipping line: {line}")
                        continue

                # 处理文件末尾的句子
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading data from {self.file_path}: {e}")

        logger.info(f"Loaded {len(sentences)} sentences from {self.file_path}")
        return sentences, labels

    def _parse_line(self, line: str) -> Tuple[str, str]:
        """
        解析单行数据

        Args:
            line: 数据行

        Returns:
            token: 词/字
            label: 标签

        Raises:
            ValueError: 如果行格式不正确
        """
        parts = line.split()

        if len(parts) < 2:
            raise ValueError(f"Invalid format: expected at least 2 fields")

        # CoNLL格式可能有多列，通常第一列是token，最后一列是NER标签
        token = parts[0]
        label = parts[-1]

        # 标准化标签（处理可能的大小写问题）
        label = self._normalize_label(label)

        return token, label

    def _normalize_label(self, label: str) -> str:
        """
        标准化标签格式

        Args:
            label: 原始标签

        Returns:
            标准化后的标签
        """
        # 确保标签格式为 O 或 B-TYPE 或 I-TYPE
        label = label.strip()

        # 处理可能的小写标签
        if label.lower() == 'o':
            return 'O'

        # 处理 B-type 或 I-type 格式
        if '-' in label:
            prefix, entity_type = label.split('-', 1)
            prefix = prefix.upper()
            entity_type = entity_type.upper()
            return f'{prefix}-{entity_type}'

        return label


class NERDataset(Dataset):
    """NER任务的PyTorch数据集"""

    def __init__(
        self,
        sentences: List[List[str]],
        labels: List[List[str]],
        tokenizer,
        label_encoder,
        max_length: int = 128
    ):
        """
        初始化NER数据集

        Args:
            sentences: 句子列表
            labels: 标签列表
            tokenizer: 分词器（HuggingFace tokenizer）
            label_encoder: 标签编码器
            max_length: 最大序列长度
        """
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Args:
            idx: 样本索引

        Returns:
            包含input_ids, attention_mask, labels的字典
        """
        tokens = self.sentences[idx]
        labels = self.labels[idx]

        # 使用tokenizer进行分词（is_split_into_words=True表示输入已分词）
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 对齐标签到子词
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = self._align_labels(tokens, labels, word_ids)

        # 转换为tensor
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels_tensor
        }

    def _align_labels(
        self,
        tokens: List[str],
        labels: List[str],
        word_ids: List[Optional[int]]
    ) -> List[int]:
        """
        对齐标签到子词

        策略：只保留每个词的第一个子词的标签，其余子词标签设为-100

        Args:
            tokens: 原始token列表
            labels: 原始标签列表
            word_ids: tokenizer返回的word_ids

        Returns:
            对齐后的标签ID列表
        """
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # 特殊token（[CLS], [SEP], [PAD]）
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # 新词的第一个子词：使用原标签
                if word_id < len(labels):
                    aligned_labels.append(self.label_encoder.encode(labels[word_id]))
                else:
                    # 处理可能的索引越界
                    aligned_labels.append(-100)
            else:
                # 新词的后续子词：设为-100（忽略）
                aligned_labels.append(-100)

            previous_word_id = word_id

        return aligned_labels


class BiLSTMDataset(Dataset):
    """BiLSTM模型的数据集（基于词表索引）"""

    def __init__(
        self,
        sentences: List[List[str]],
        labels: List[List[str]],
        vocab: Dict[str, int],
        label_encoder,
        max_length: int = 128
    ):
        """
        初始化BiLSTM数据集

        Args:
            sentences: 句子列表
            labels: 标签列表
            vocab: 词表（word -> id）
            label_encoder: 标签编码器
            max_length: 最大序列长度
        """
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.pad_id = vocab.get('<PAD>', 0)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sentences[idx]
        labels = self.labels[idx]

        # 转换token为ID
        token_ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1))
                     for token in tokens]

        # 转换label为ID
        label_ids = self.label_encoder.encode_batch(labels)

        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            label_ids = label_ids[:self.max_length]

        # 创建attention mask
        attention_mask = [1] * len(token_ids)

        # 填充
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids.extend([self.pad_id] * padding_length)
            label_ids.extend([-100] * padding_length)  # 填充位置的标签设为-100
            attention_mask.extend([0] * padding_length)

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


def build_vocab(sentences: List[List[str]], min_freq: int = 2, max_size: int = 5000) -> Dict[str, int]:
    """
    从句子列表构建词表

    Args:
        sentences: 句子列表
        min_freq: 最小词频
        max_size: 词表最大大小

    Returns:
        词表字典
    """
    from collections import Counter

    # 统计词频
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)

    # 构建词表
    vocab = {'<PAD>': 0, '<UNK>': 1}

    # 按词频排序，取前max_size个
    for word, count in word_counts.most_common(max_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)

    logger.info(f"Built vocabulary with {len(vocab)} words")
    return vocab
