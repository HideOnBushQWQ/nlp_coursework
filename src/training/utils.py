"""
训练工具函数
"""

import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 3, mode: str = 'max', delta: float = 0.0):
        """
        初始化早停机制

        Args:
            patience: 容忍的epoch数
            mode: 'min'表示指标越小越好，'max'表示越大越好
            delta: 最小改进量
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float):
        """
        更新早停状态

        Args:
            score: 当前指标得分
        """
        if self.best_score is None:
            self.best_score = score
            logger.info(f"Initial score: {score:.4f}")
        elif self._is_improvement(score):
            logger.info(f"Score improved: {self.best_score:.4f} -> {score:.4f}")
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("Early stopping triggered!")

    def _is_improvement(self, score: float) -> bool:
        """
        判断是否有改进

        Args:
            score: 当前得分

        Returns:
            是否有改进
        """
        if self.mode == 'min':
            return score < self.best_score - self.delta
        else:  # mode == 'max'
            return score > self.best_score + self.delta


def set_seed(seed: int):
    """
    设置随机种子，保证实验可复现

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def count_parameters(model) -> int:
    """
    统计模型参数量

    Args:
        model: PyTorch模型

    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    """
    保存训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        save_path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device='cpu'):
    """
    加载训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        checkpoint_path: 检查点路径
        device: 设备

    Returns:
        起始epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")
    return epoch
