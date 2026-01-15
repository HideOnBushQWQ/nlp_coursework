"""
模型基类模块
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn
import json
import os


class BaseNERModel(nn.Module, ABC):
    """NER模型抽象基类"""

    def __init__(self, config: Dict):
        """
        初始化模型基类

        Args:
            config: 模型配置字典
        """
        super().__init__()
        self.config = config
        self.num_labels = config.get('num_labels', 9)

    @abstractmethod
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
        pass

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        推理预测

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            predictions: [batch_size, seq_len]
        """
        self.eval()
        outputs = self.forward(input_ids, attention_mask)
        return outputs['predictions']

    def save_pretrained(self, save_dir: str):
        """
        保存模型

        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_dir, 'model.pt'))

        # 保存配置
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, save_dir: str, device: str = 'cpu'):
        """
        加载模型

        Args:
            save_dir: 模型目录
            device: 设备

        Returns:
            模型实例
        """
        # 加载配置
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 创建模型
        model = cls(config)

        # 加载权重
        model_path = os.path.join(save_dir, 'model.pt')
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

        return model

    def get_num_parameters(self) -> int:
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
