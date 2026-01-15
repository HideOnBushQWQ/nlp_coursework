"""
BiLSTM-CRF模型
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from torchcrf import CRF

from .base_model import BaseNERModel


class BiLSTMCRF(BaseNERModel):
    """BiLSTM + CRF基线模型"""

    def __init__(self, config: Dict):
        """
        初始化BiLSTM-CRF模型

        Args:
            config: 模型配置，包含：
                - vocab_size: 词表大小
                - embedding_dim: 词嵌入维度
                - hidden_dim: LSTM隐藏层维度
                - num_layers: LSTM层数
                - dropout: dropout概率
                - num_labels: 标签数量
                - pretrained_embeddings: 预训练词向量（可选）
        """
        super().__init__(config)

        # 词嵌入层
        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embedding_dim'],
            padding_idx=0
        )

        # 加载预训练词向量（可选）
        if config.get('pretrained_embeddings') is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(config['pretrained_embeddings'])
            )
            # 可选：冻结embedding层
            if config.get('freeze_embeddings', False):
                self.embedding.weight.requires_grad = False

        # BiLSTM层
        self.lstm = nn.LSTM(
            config['embedding_dim'],
            config['hidden_dim'] // 2,  # 双向，所以除以2
            num_layers=config.get('num_layers', 2),
            bidirectional=True,
            batch_first=True,
            dropout=config.get('dropout', 0.5) if config.get('num_layers', 2) > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(config.get('dropout', 0.5))

        # 分类层
        self.classifier = nn.Linear(config['hidden_dim'], self.num_labels)

        # CRF层
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] (可选)

        Returns:
            包含loss, logits, predictions的字典
        """
        # 词嵌入
        embeddings = self.embedding(input_ids)  # [B, L, E]
        embeddings = self.dropout(embeddings)

        # BiLSTM（使用pack_padded_sequence优化）
        # 计算实际长度
        lengths = attention_mask.sum(dim=1).cpu()

        # pack序列
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, _ = self.lstm(packed_embeddings)

        # unpack序列
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        lstm_out = self.dropout(lstm_out)

        # 分类
        logits = self.classifier(lstm_out)  # [B, L, num_labels]

        # CRF
        mask = (attention_mask == 1)
        loss = None

        if labels is not None:
            # 处理标签
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0
            mask_for_loss = (labels != -100) & mask
            loss = -self.crf(logits, labels_for_crf, mask=mask_for_loss, reduction='mean')

        # 解码
        predictions = self.crf.decode(logits, mask=mask)
        # 转换为tensor
        max_len = logits.size(1)
        predictions_padded = []
        for pred in predictions:
            pred_padded = pred + [0] * (max_len - len(pred))
            predictions_padded.append(pred_padded)
        predictions = torch.tensor(predictions_padded, device=logits.device)

        return {
            'loss': loss,
            'logits': logits,
            'predictions': predictions
        }
