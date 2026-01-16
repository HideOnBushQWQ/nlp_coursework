"""
训练脚本
"""

import argparse
import yaml
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, RobertaTokenizerFast

# 确保项目根目录在 Python 搜索路径中，方便直接运行 scripts 下的脚本
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import CoNLLDatasetLoader, NERDataset, LabelEncoder, build_vocab, BiLSTMDataset
from src.models import BertNER, RobertaNER, BiLSTMCRF
from src.training import NERTrainer, set_seed


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(config, tokenizer=None, vocab=None):
    """加载数据"""
    logger.info("Loading data...")

    # 加载训练集
    train_loader_obj = CoNLLDatasetLoader(config['data']['train_file'])
    train_sentences, train_labels = train_loader_obj.load()

    # 加载验证集
    val_loader_obj = CoNLLDatasetLoader(config['data']['dev_file'])
    val_sentences, val_labels = val_loader_obj.load()

    # 创建标签编码器
    label_encoder = LabelEncoder()

    # 创建数据集
    if config['model']['type'] == 'bilstm':
        # BiLSTM需要构建词表
        if vocab is None:
            vocab = build_vocab(train_sentences + val_sentences)

        train_dataset = BiLSTMDataset(
            train_sentences,
            train_labels,
            vocab,
            label_encoder,
            config['data']['max_length']
        )
        val_dataset = BiLSTMDataset(
            val_sentences,
            val_labels,
            vocab,
            label_encoder,
            config['data']['max_length']
        )
    else:
        # BERT/RoBERTa使用tokenizer
        train_dataset = NERDataset(
            train_sentences,
            train_labels,
            tokenizer,
            label_encoder,
            config['data']['max_length']
        )
        val_dataset = NERDataset(
            val_sentences,
            val_labels,
            tokenizer,
            label_encoder,
            config['data']['max_length']
        )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    return train_loader, val_loader, label_encoder


def create_model(config):
    """创建模型"""
    model_type = config['model']['type']

    if model_type == 'bert':
        model = BertNER(config['model'])
    elif model_type == 'roberta':
        model = RobertaNER(config['model'])
    elif model_type == 'bilstm':
        model = BiLSTMCRF(config['model'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Created {model_type} model")
    logger.info(f"Model parameters: {model.get_trainable_parameters():,}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train NER model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: from config)')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置设备
    if args.device:
        config['device'] = args.device
    
    # 确保设备设置正确：如果配置中指定了cuda但不可用，则回退到cpu
    requested_device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if requested_device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    else:
        device = requested_device
    
    logger.info(f"Using device: {device}")

    # 设置随机种子
    set_seed(config.get('seed', 42))

    # 创建输出目录
    os.makedirs(config['output']['save_dir'], exist_ok=True)

    # 准备tokenizer或词表
    tokenizer = None
    vocab = None

    if config['model']['type'] in ['bert', 'roberta']:
        # 加载fast tokenizer（支持 word_ids，用于对齐标签）
        if config['model']['type'] == 'bert':
            tokenizer = BertTokenizerFast.from_pretrained(config['model']['pretrained_model'])
        else:
            tokenizer = RobertaTokenizerFast.from_pretrained(config['model']['pretrained_model'])
        logger.info(f"Loaded fast tokenizer: {config['model']['pretrained_model']}")

    # 加载数据
    train_loader, val_loader, label_encoder = load_data(config, tokenizer, vocab)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # 创建模型
    model = create_model(config)

    # 创建训练器
    trainer = NERTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        label_encoder=label_encoder,
        config=config,
        device=device
    )

    # 开始训练
    trainer.train()

    logger.info("Training finished!")


if __name__ == '__main__':
    main()
