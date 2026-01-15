"""
BERT NER模型
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

from .base_model import BaseNERModel


class BertNER(BaseNERModel):
    """BERT + CRF NER模型"""

    def __init__(self, config: Dict):
        """
        初始化BERT NER模型

        Args:
            config: 模型配置，包含：
                - pretrained_model: 预训练模型名称
                - num_labels: 标签数量
                - dropout: dropout概率
                - use_crf: 是否使用CRF层
        """
        super().__init__(config)

        # BERT编码器
        self.bert = BertModel.from_pretrained(
            config['pretrained_model']
        )

        # Dropout
        self.dropout = nn.Dropout(config.get('dropout', 0.1))

        # 分类层
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            self.num_labels
        )

        # CRF层（可选）
        self.use_crf = config.get('use_crf', True)
        if self.use_crf:
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
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state  # [B, L, H]

        # Dropout + 分类
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [B, L, num_labels]

        # 损失计算和预测
        loss = None
        predictions = None

        if self.use_crf:
            # 使用CRF
            if labels is not None:
                # 创建mask（排除padding和特殊标签-100）
                mask = (labels != -100) & (attention_mask == 1)
                # 将-100替换为0（CRF不接受负数标签）
                labels_for_crf = labels.clone()
                labels_for_crf[labels == -100] = 0
                # 计算负对数似然
                loss = -self.crf(logits, labels_for_crf, mask=mask, reduction='mean')

            # CRF解码
            mask = (attention_mask == 1)
            predictions = self.crf.decode(logits, mask=mask)
            # 转换为tensor（解码结果是list of lists）
            max_len = logits.size(1)
            predictions_padded = []
            for pred in predictions:
                pred_padded = pred + [0] * (max_len - len(pred))
                predictions_padded.append(pred_padded)
            predictions = torch.tensor(predictions_padded, device=logits.device)
        else:
            # 不使用CRF
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1)
                )

            predictions = torch.argmax(logits, dim=-1)

        return {
            'loss': loss,
            'logits': logits,
            'predictions': predictions
        }
