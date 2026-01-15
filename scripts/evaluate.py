"""
评估脚本
"""

import argparse
import yaml
import json
import logging
import os

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer

from src.data_preprocessing import CoNLLDatasetLoader, NERDataset, LabelEncoder
from src.models import BertNER, RobertaNER
from src.evaluation import NERMetrics, compute_detailed_metrics


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate NER model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--test_file', type=str,
                        help='Test file (default: from config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_file', type=str,
                        help='Output file for detailed results')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # 加载tokenizer
    if config['model']['type'] == 'bert':
        tokenizer = BertTokenizer.from_pretrained(config['model']['pretrained_model'])
        model_class = BertNER
    elif config['model']['type'] == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(config['model']['pretrained_model'])
        model_class = RobertaNER
    else:
        raise ValueError(f"Model type {config['model']['type']} not supported")

    # 加载标签编码器
    label_encoder = LabelEncoder.load(os.path.join(args.model_path, 'label_encoder.json'))

    # 加载测试数据
    test_file = args.test_file or config['data']['test_file']
    logger.info(f"Loading test data from {test_file}")

    loader = CoNLLDatasetLoader(test_file)
    sentences, labels = loader.load()

    test_dataset = NERDataset(
        sentences,
        labels,
        tokenizer,
        label_encoder,
        config['data']['max_length']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # 加载模型
    logger.info(f"Loading model from {args.model_path}")
    model = model_class.from_pretrained(args.model_path, device=device)
    model.eval()

    # 评估
    logger.info("Evaluating...")
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            predictions = model.predict(input_ids, attention_mask)

            # 收集有效的预测和标签
            for pred, label, mask in zip(predictions, batch_labels, attention_mask):
                valid_indices = (label != -100) & (mask == 1)
                valid_pred = pred[valid_indices].cpu().tolist()
                valid_label = label[valid_indices].cpu().tolist()

                if len(valid_pred) > 0:
                    all_predictions.append(valid_pred)
                    all_labels.append(valid_label)

    # 计算指标
    metrics = compute_detailed_metrics(all_predictions, all_labels, label_encoder)

    # 打印结果
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("\nPer-entity metrics:")

    for entity_type in ['PER', 'LOC', 'ORG', 'MISC']:
        if entity_type in metrics['report']:
            entity_metrics = metrics['report'][entity_type]
            print(f"\n{entity_type}:")
            print(f"  Precision: {entity_metrics['precision']:.4f}")
            print(f"  Recall: {entity_metrics['recall']:.4f}")
            print(f"  F1-Score: {entity_metrics['f1-score']:.4f}")
            print(f"  Support: {entity_metrics['support']}")

    # 保存结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to {args.output_file}")


if __name__ == '__main__':
    main()
