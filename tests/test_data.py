"""
数据模块测试
"""

import unittest
import tempfile
import os
from src.data_preprocessing import CoNLLDatasetLoader, LabelEncoder, NERDataset


class TestLabelEncoder(unittest.TestCase):
    """测试标签编码器"""

    def setUp(self):
        """设置测试环境"""
        self.encoder = LabelEncoder()

    def test_encode_decode(self):
        """测试编码和解码"""
        label = 'B-PER'
        label_id = self.encoder.encode(label)
        decoded_label = self.encoder.decode(label_id)
        self.assertEqual(label, decoded_label)

    def test_num_labels(self):
        """测试标签数量"""
        # O + 4类实体 * 2(B/I) = 9
        self.assertEqual(self.encoder.num_labels, 9)

    def test_encode_batch(self):
        """测试批量编码"""
        labels = ['O', 'B-PER', 'I-PER', 'O']
        label_ids = self.encoder.encode_batch(labels)
        decoded_labels = self.encoder.decode_batch(label_ids)
        self.assertEqual(labels, decoded_labels)


class TestCoNLLDatasetLoader(unittest.TestCase):
    """测试数据集加载器"""

    def setUp(self):
        """设置测试环境"""
        # 创建临时测试文件
        self.test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
        self.test_file.write("中 B-LOC\n")
        self.test_file.write("国 I-LOC\n")
        self.test_file.write("是 O\n")
        self.test_file.write("\n")
        self.test_file.write("北 B-LOC\n")
        self.test_file.write("京 I-LOC\n")
        self.test_file.close()

    def tearDown(self):
        """清理测试环境"""
        os.unlink(self.test_file.name)

    def test_load(self):
        """测试加载数据"""
        loader = CoNLLDatasetLoader(self.test_file.name)
        sentences, labels = loader.load()

        self.assertEqual(len(sentences), 2)
        self.assertEqual(len(labels), 2)
        self.assertEqual(sentences[0], ['中', '国', '是'])
        self.assertEqual(labels[0], ['B-LOC', 'I-LOC', 'O'])


if __name__ == '__main__':
    unittest.main()
