"""
推理脚本
"""

import argparse
import json
import os

import torch
from transformers import BertTokenizer, RobertaTokenizer

from src.models import BertNER, RobertaNER, BiLSTMCRF
from src.inference import NERPredictor


def main():
    parser = argparse.ArgumentParser(description='NER Prediction')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['bert', 'roberta', 'bilstm'],
                        help='Model type')
    parser.add_argument('--pretrained_model', type=str,
                        help='Pretrained model name for tokenizer')
    parser.add_argument('--text', type=str,
                        help='Text to predict')
    parser.add_argument('--input_file', type=str,
                        help='Input file with texts (one per line)')
    parser.add_argument('--output_file', type=str,
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # 检查设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载tokenizer
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        model_class = BertNER
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
        model_class = RobertaNER
    else:
        raise ValueError(f"Model type {args.model_type} not supported for prediction yet")

    # 加载预测器
    print(f"Loading model from {args.model_path}...")
    predictor = NERPredictor.from_pretrained(
        args.model_path,
        model_class,
        tokenizer,
        device
    )

    # 预测
    if args.text:
        # 单句预测
        result = predictor.predict(args.text)
        print("\n=== Prediction Result ===")
        print(f"Text: {result['text']}")
        print(f"\nEntities:")
        for entity in result['entities']:
            print(f"  - {entity['text']} ({entity['type']})")

    elif args.input_file:
        # 批量预测
        print(f"Loading texts from {args.input_file}...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"Predicting {len(texts)} texts...")
        results = predictor.predict_batch(texts, batch_size=32)

        # 保存结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output_file}")
        else:
            for i, result in enumerate(results[:5]):  # 只打印前5个
                print(f"\n=== Text {i+1} ===")
                print(f"Text: {result['text']}")
                print(f"Entities: {result['entities']}")
    else:
        print("Error: Please provide either --text or --input_file")


if __name__ == '__main__':
    main()
