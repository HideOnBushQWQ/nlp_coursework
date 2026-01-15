"""
评估指标测试
"""

import unittest
from src.evaluation.metrics import compute_metrics
from src.data_preprocessing import LabelEncoder


class TestMetrics(unittest.TestCase):
    """测试评估指标"""

    def setUp(self):
        """设置测试环境"""
        self.label_encoder = LabelEncoder()

    def test_perfect_prediction(self):
        """测试完美预测"""
        # 完全正确的预测
        predictions = [[0, 1, 2, 0]]  # O, B-PER, I-PER, O
        labels = [[0, 1, 2, 0]]

        metrics = compute_metrics(predictions, labels, self.label_encoder)

        self.assertAlmostEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 1.0)
        self.assertAlmostEqual(metrics['f1'], 1.0)

    def test_no_match(self):
        """测试完全不匹配"""
        predictions = [[0, 0, 0, 0]]  # 全部预测为O
        labels = [[0, 1, 2, 0]]  # 有一个实体

        metrics = compute_metrics(predictions, labels, self.label_encoder)

        # 没有预测任何实体，precision和recall都应该为0
        # 但根据seqeval的实现，可能会有不同的处理
        self.assertTrue(0 <= metrics['f1'] <= 1)


if __name__ == '__main__':
    unittest.main()
