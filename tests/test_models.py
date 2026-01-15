"""
模型模块测试
"""

import unittest
import torch
from src.models import BertNER, RobertaNER, BiLSTMCRF


class TestModelBasics(unittest.TestCase):
    """测试模型基本功能"""

    def test_bert_ner_forward(self):
        """测试BERT模型前向传播"""
        config = {
            'pretrained_model': 'bert-base-uncased',
            'num_labels': 9,
            'dropout': 0.1,
            'use_crf': False  # 不使用CRF以加快测试
        }

        model = BertNER(config)

        # 创建假数据
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 9, (batch_size, seq_len))

        # 前向传播
        outputs = model(input_ids, attention_mask, labels)

        self.assertIsNotNone(outputs['loss'])
        self.assertEqual(outputs['logits'].shape, (batch_size, seq_len, 9))
        self.assertEqual(outputs['predictions'].shape, (batch_size, seq_len))

    def test_bilstm_crf_forward(self):
        """测试BiLSTM-CRF模型前向传播"""
        config = {
            'vocab_size': 1000,
            'embedding_dim': 100,
            'hidden_dim': 128,
            'num_layers': 1,
            'dropout': 0.5,
            'num_labels': 9
        }

        model = BiLSTMCRF(config)

        # 创建假数据
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 9, (batch_size, seq_len))

        # 前向传播
        outputs = model(input_ids, attention_mask, labels)

        self.assertIsNotNone(outputs['loss'])
        self.assertEqual(outputs['predictions'].shape, (batch_size, seq_len))


if __name__ == '__main__':
    unittest.main()
