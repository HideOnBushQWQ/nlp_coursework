"""
训练器模块
"""

from typing import Dict
import os
import json
import logging
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import EarlyStopping
 
logger = logging.getLogger(__name__)


class NERTrainer:
    """NER训练器"""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        label_encoder,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        初始化训练器

        Args:
            model: NER模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            label_encoder: 标签编码器
            config: 训练配置
            device: 设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_encoder = label_encoder
        self.config = config
        self.device = device

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 早停
        self.early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 3),
            mode='max'  # F1-score越大越好
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rate': []
        }

        # 输出目录
        self.save_dir = config['output']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        # 最优模型路径
        self.best_model_path = os.path.join(self.save_dir, 'best_model')

    def _create_optimizer(self) -> AdamW:
        """创建优化器"""
        # 分组参数：预训练模型参数 vs 新增参数
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        
        # 确保数值类型正确（YAML可能读取为字符串）
        learning_rate = self.config['training'].get('learning_rate', 3e-5)
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        
        weight_decay = self.config['training'].get('weight_decay', 0.01)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=1e-8
        )
        return optimizer

    def _create_scheduler(self):
        """创建学习率调度器"""
        num_training_steps = len(self.train_loader) * self.config['training']['num_epochs']
        
        # 确保warmup_ratio是浮点数
        warmup_ratio = self.config['training'].get('warmup_ratio', 0.1)
        if isinstance(warmup_ratio, str):
            warmup_ratio = float(warmup_ratio)
        
        num_warmup_steps = int(num_training_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            # 数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('max_grad_norm', 1.0)
            )

            # 更新参数
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def evaluate(self, data_loader: DataLoader = None) -> Dict[str, float]:
        """
        评估模型

        Args:
            data_loader: 数据加载器，默认使用验证集

        Returns:
            评估指标字典
        """
        if data_loader is None:
            data_loader = self.val_loader

        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                predictions = outputs['predictions']

                total_loss += loss.item()
                num_batches += 1

                # 收集预测和标签（排除padding和特殊token）
                for pred, label, mask in zip(predictions, labels, attention_mask):
                    # 只保留有效位置
                    valid_indices = (label != -100) & (mask == 1)
                    valid_pred = pred[valid_indices].cpu().tolist()
                    valid_label = label[valid_indices].cpu().tolist()

                    if len(valid_pred) > 0:
                        all_predictions.append(valid_pred)
                        all_labels.append(valid_label)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # 计算指标
        from src.evaluation.metrics import compute_metrics
        metrics = compute_metrics(all_predictions, all_labels, self.label_encoder)
        metrics['loss'] = avg_loss

        return metrics

    def train(self):
        """完整训练流程"""
        logger.info("=" * 50)
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Num epochs: {self.config['training']['num_epochs']}")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Learning rate: {self.config['training']['learning_rate']}")
        logger.info(f"Model parameters: {self.model.get_trainable_parameters():,}")
        logger.info("=" * 50)

        best_f1 = 0.0

        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"\n{'='*20} Epoch {epoch + 1}/{self.config['training']['num_epochs']} {'='*20}")

            # 训练
            train_loss = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}")

            # 验证
            val_metrics = self.evaluate()
            val_loss = val_metrics['loss']
            val_precision = val_metrics['precision']
            val_recall = val_metrics['recall']
            val_f1 = val_metrics['f1']

            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Precision: {val_precision:.4f}")
            logger.info(f"Val Recall: {val_recall:.4f}")
            logger.info(f"Val F1: {val_f1:.4f}")

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # 保存最优模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                logger.info(f"✓ New best model! F1: {best_f1:.4f}")
                logger.info(f"Saving to {self.best_model_path}")
                self.model.save_pretrained(self.best_model_path)
                # 保存标签编码器
                self.label_encoder.save(os.path.join(self.best_model_path, 'label_encoder.json'))

            # 早停检查
            self.early_stopping(val_f1)
            if self.early_stopping.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info("\n" + "=" * 50)
        logger.info("Training completed!")
        logger.info(f"Best Val F1: {best_f1:.4f}")
        logger.info("=" * 50)

        # 保存训练历史
        self._save_history()

    def _save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
