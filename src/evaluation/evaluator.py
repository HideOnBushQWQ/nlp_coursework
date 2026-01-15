"""
模型评估器和对比器
"""

from typing import Dict, Tuple, List
import pandas as pd
import logging

from .metrics import NERMetrics

logger = logging.getLogger(__name__)


class ModelComparator:
    """多模型性能对比器"""

    def __init__(self, label_encoder):
        """
        初始化对比器

        Args:
            label_encoder: 标签编码器
        """
        self.label_encoder = label_encoder
        self.metrics_calculator = NERMetrics(label_encoder)

    def compare_models(
        self,
        models_results: Dict[str, Tuple[List, List]],
        save_path: str = None
    ) -> pd.DataFrame:
        """
        对比多个模型的性能

        Args:
            models_results: {
                'BERT': (predictions, labels),
                'RoBERTa': (predictions, labels),
                ...
            }
            save_path: 保存结果的路径

        Returns:
            对比结果DataFrame
        """
        results = []

        for model_name, (predictions, labels) in models_results.items():
            logger.info(f"Computing metrics for {model_name}...")
            metrics = self.metrics_calculator.compute(predictions, labels)

            results.append({
                'Model': model_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })

        df = pd.DataFrame(results)

        # 按F1-Score排序
        df = df.sort_values('F1-Score', ascending=False)

        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8')
            logger.info(f"Comparison results saved to {save_path}")

        return df

    def detailed_comparison(
        self,
        models_results: Dict[str, Tuple[List, List]],
        save_path: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        详细对比多个模型在各实体类型上的性能

        Args:
            models_results: 模型结果字典
            save_path: 保存目录

        Returns:
            包含总体和各类实体对比结果的字典
        """
        # 总体对比
        overall_df = self.compare_models(models_results)

        # 各实体类型的对比
        entity_types = ['PER', 'LOC', 'ORG', 'MISC']
        entity_results = {entity_type: [] for entity_type in entity_types}

        for model_name, (predictions, labels) in models_results.items():
            metrics = self.metrics_calculator.compute(predictions, labels)
            report = metrics['report']

            for entity_type in entity_types:
                if entity_type in report:
                    entity_results[entity_type].append({
                        'Model': model_name,
                        'Precision': report[entity_type]['precision'],
                        'Recall': report[entity_type]['recall'],
                        'F1-Score': report[entity_type]['f1-score']
                    })

        # 转换为DataFrame
        entity_dfs = {
            entity_type: pd.DataFrame(results).sort_values('F1-Score', ascending=False)
            for entity_type, results in entity_results.items()
            if results
        }

        # 保存结果
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)

            overall_df.to_csv(
                os.path.join(save_path, 'overall_comparison.csv'),
                index=False,
                encoding='utf-8'
            )

            for entity_type, df in entity_dfs.items():
                df.to_csv(
                    os.path.join(save_path, f'{entity_type}_comparison.csv'),
                    index=False,
                    encoding='utf-8'
                )

            logger.info(f"Detailed comparison results saved to {save_path}")

        return {
            'overall': overall_df,
            **entity_dfs
        }
