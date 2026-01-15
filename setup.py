"""
项目安装配置
"""

from setuptools import setup, find_packages

setup(
    name="ner_system",
    version="1.0.0",
    description="中英双语命名实体识别系统",
    author="NLP Coursework",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "pytorch-crf>=0.7.2",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "seqeval>=1.2.2",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
    ],
    entry_points={
        "console_scripts": [
            "ner-train=scripts.train:main",
            "ner-evaluate=scripts.evaluate:main",
            "ner-predict=scripts.predict:main",
        ],
    },
)
