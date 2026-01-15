"""
NER推理器模块
"""

from typing import List, Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)


class NERPredictor:
    """NER推理器"""

    def __init__(
        self,
        model,
        tokenizer,
        label_encoder,
        device: str = 'cuda'
    ):
        """
        初始化推理器

        Args:
            model: NER模型
            tokenizer: 分词器
            label_encoder: 标签编码器
            device: 设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.device = device

    def predict(self, text: str, return_offsets: bool = False) -> Dict[str, Any]:
        """
        单句推理

        Args:
            text: 输入文本
            return_offsets: 是否返回字符级别的偏移

        Returns:
            包含tokens, labels, entities的字典
        """
        # 处理文本（如果是句子，先分词；如果已分词，保持原样）
        if isinstance(text, str):
            # 对于中文，按字符分割；对于英文，按空格分割
            if self._is_chinese(text):
                tokens = list(text.replace(' ', ''))
            else:
                tokens = text.split()
        else:
            tokens = text

        # 分词
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=return_offsets
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model.predict(input_ids, attention_mask)

        # 解码
        word_ids = encoding.word_ids(batch_index=0)
        pred_labels = []
        aligned_tokens = []

        previous_word_id = None
        for i, (word_id, pred_id) in enumerate(zip(word_ids, predictions[0])):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                if word_id < len(tokens):
                    aligned_tokens.append(tokens[word_id])
                    pred_labels.append(self.label_encoder.decode(pred_id.item()))
            previous_word_id = word_id

        # 提取实体
        entities = self._extract_entities(aligned_tokens, pred_labels)

        result = {
            'text': text if isinstance(text, str) else ' '.join(text),
            'tokens': aligned_tokens,
            'labels': pred_labels,
            'entities': entities
        }

        return result

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        批量推理

        Args:
            texts: 文本列表
            batch_size: 批大小

        Returns:
            预测结果列表
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._predict_batch(batch_texts)
            results.extend(batch_results)

        return results

    def _predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量推理的内部实现"""
        # 预处理文本
        all_tokens = []
        for text in texts:
            if self._is_chinese(text):
                tokens = list(text.replace(' ', ''))
            else:
                tokens = text.split()
            all_tokens.append(tokens)

        # 批量分词
        encoding = self.tokenizer(
            all_tokens,
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model.predict(input_ids, attention_mask)

        # 解码每个样本
        results = []
        for idx, (text, tokens) in enumerate(zip(texts, all_tokens)):
            word_ids = encoding.word_ids(batch_index=idx)
            pred = predictions[idx]

            pred_labels = []
            aligned_tokens = []

            previous_word_id = None
            for word_id, pred_id in zip(word_ids, pred):
                if word_id is None:
                    continue
                if word_id != previous_word_id:
                    if word_id < len(tokens):
                        aligned_tokens.append(tokens[word_id])
                        pred_labels.append(self.label_encoder.decode(pred_id.item()))
                previous_word_id = word_id

            entities = self._extract_entities(aligned_tokens, pred_labels)

            results.append({
                'text': text,
                'tokens': aligned_tokens,
                'labels': pred_labels,
                'entities': entities
            })

        return results

    def _extract_entities(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[Dict[str, Any]]:
        """
        从token和标签中提取实体

        Args:
            tokens: token列表
            labels: 标签列表

        Returns:
            实体列表
        """
        entities = []
        entity_tokens = []
        entity_type = None
        entity_start = None

        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                # 保存上一个实体
                if entity_tokens:
                    entities.append({
                        'text': self._tokens_to_string(entity_tokens),
                        'type': entity_type,
                        'start': entity_start,
                        'end': i
                    })
                # 开始新实体
                entity_tokens = [token]
                entity_type = label[2:]
                entity_start = i
            elif label.startswith('I-') and entity_tokens:
                # 继续当前实体
                if entity_type == label[2:]:
                    entity_tokens.append(token)
                else:
                    # 类型不匹配，结束旧实体，开始新实体
                    entities.append({
                        'text': self._tokens_to_string(entity_tokens),
                        'type': entity_type,
                        'start': entity_start,
                        'end': i
                    })
                    entity_tokens = [token]
                    entity_type = label[2:]
                    entity_start = i
            else:  # 'O' or mismatched I-
                # 结束当前实体
                if entity_tokens:
                    entities.append({
                        'text': self._tokens_to_string(entity_tokens),
                        'type': entity_type,
                        'start': entity_start,
                        'end': i
                    })
                    entity_tokens = []
                    entity_type = None
                    entity_start = None

        # 处理最后一个实体
        if entity_tokens:
            entities.append({
                'text': self._tokens_to_string(entity_tokens),
                'type': entity_type,
                'start': entity_start,
                'end': len(tokens)
            })

        return entities

    def _tokens_to_string(self, tokens: List[str]) -> str:
        """
        将token列表转换为字符串

        Args:
            tokens: token列表

        Returns:
            拼接后的字符串
        """
        text = ''.join(tokens)
        # 移除特殊标记
        text = text.replace('##', '')  # BERT的子词标记
        text = text.replace('Ġ', ' ')  # RoBERTa的空格标记
        return text.strip()

    def _is_chinese(self, text: str) -> bool:
        """
        判断文本是否主要是中文

        Args:
            text: 输入文本

        Returns:
            是否是中文
        """
        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        return chinese_count / max(len(text), 1) > 0.3

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_class,
        tokenizer,
        device: str = 'cuda'
    ):
        """
        从预训练模型加载推理器

        Args:
            model_path: 模型路径
            model_class: 模型类
            tokenizer: 分词器
            device: 设备

        Returns:
            NERPredictor实例
        """
        import os
        from src.data_preprocessing import LabelEncoder

        # 加载模型
        model = model_class.from_pretrained(model_path, device=device)

        # 加载标签编码器
        label_encoder_path = os.path.join(model_path, 'label_encoder.json')
        label_encoder = LabelEncoder.load(label_encoder_path)

        return cls(model, tokenizer, label_encoder, device)
