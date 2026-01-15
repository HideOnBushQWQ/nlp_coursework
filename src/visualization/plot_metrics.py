"""
指标可视化模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """训练过程可视化器"""

    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        初始化可视化器

        Args:
            style: matplotlib样式
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        sns.set_palette("husl")

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: str = None
    ):
        """
        绘制训练历史

        Args:
            history: 训练历史字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(history['train_loss']) + 1)

        # 损失曲线
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # F1曲线
        axes[0, 1].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2, marker='o')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('F1 Score', fontsize=12)
        axes[0, 1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # Precision和Recall
        if 'val_precision' in history and 'val_recall' in history:
            axes[1, 0].plot(epochs, history['val_precision'], 'b-', label='Precision', linewidth=2)
            axes[1, 0].plot(epochs, history['val_recall'], 'r-', label='Recall', linewidth=2)
            axes[1, 0].plot(epochs, history['val_f1'], 'g-', label='F1', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Score', fontsize=12)
            axes[1, 0].set_title('Precision, Recall, and F1', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)

        # 学习率曲线
        if 'learning_rate' in history:
            axes[1, 1].plot(epochs, history['learning_rate'], 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        plt.show()

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        绘制模型对比柱状图

        Args:
            comparison_df: 对比结果DataFrame
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(comparison_df))
        width = 0.25

        bars1 = ax.bar(x - width, comparison_df['Precision'], width,
                       label='Precision', color='skyblue', edgecolor='black')
        bars2 = ax.bar(x, comparison_df['Recall'], width,
                       label='Recall', color='lightcoral', edgecolor='black')
        bars3 = ax.bar(x + width, comparison_df['F1-Score'], width,
                       label='F1-Score', color='lightgreen', edgecolor='black')

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

        # 添加数值标签
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        plt.show()

    def plot_confusion_matrix(
        self,
        confusion_data: Dict,
        save_path: str = None
    ):
        """
        绘制混淆矩阵

        Args:
            confusion_data: 混淆矩阵数据
            save_path: 保存路径
        """
        # 提取标签
        labels = sorted(set(confusion_data.keys()))

        # 构建矩阵
        matrix = np.zeros((len(labels), len(labels)))
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                matrix[i, j] = confusion_data.get(true_label, {}).get(pred_label, 0)

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.show()

    def plot_entity_distribution(
        self,
        entity_counts: Dict[str, int],
        save_path: str = None
    ):
        """
        绘制实体类型分布

        Args:
            entity_counts: 实体计数字典
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        entities = list(entity_counts.keys())
        counts = list(entity_counts.values())

        bars = ax.bar(entities, counts, color='steelblue', edgecolor='black')
        ax.set_xlabel('Entity Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Entity Type Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Entity distribution plot saved to {save_path}")
        plt.show()
