import os
from datasets import load_dataset

BASE_DIR = "data/raw"
EN_DIR = os.path.join(BASE_DIR, "english")
ZH_DIR = os.path.join(BASE_DIR, "chinese")

os.makedirs(EN_DIR, exist_ok=True)
os.makedirs(ZH_DIR, exist_ok=True)


def save_conll(sentences, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        for sent, lab in zip(sentences, labels):
            for w, t in zip(sent, lab):
                f.write(f"{w} {t}\n")
            f.write("\n")


# ---------------------------
# English: CoNLL-2003
# ---------------------------
print("Downloading CoNLL-2003 (English)...")
conll = load_dataset("conll2003")

save_conll(
    conll["train"]["tokens"],
    conll["train"]["ner_tags"],
    os.path.join(EN_DIR, "train.txt"),
)
save_conll(
    conll["validation"]["tokens"],
    conll["validation"]["ner_tags"],
    os.path.join(EN_DIR, "dev.txt"),
)
save_conll(
    conll["test"]["tokens"],
    conll["test"]["ner_tags"],
    os.path.join(EN_DIR, "test.txt"),
)

# HuggingFace ner_tags 是数字，这里转成 BIO 标签
label_map = conll["train"].features["ner_tags"].feature.names

def replace_tags(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                w, t = line.split()
                lines.append(f"{w} {label_map[int(t)]}\n")
            else:
                lines.append("\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

for f in ["train.txt", "dev.txt", "test.txt"]:
    replace_tags(os.path.join(EN_DIR, f))


# ---------------------------
# Chinese: MSRA NER
# ---------------------------
print("Downloading MSRA NER (Chinese)...")
msra = load_dataset("msra_ner")

label_names = msra["train"].features["ner_tags"].feature.names
# e.g. ['O', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']

def save_msra(split, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in split:
            for ch, tag_id in zip(item["tokens"], item["ner_tags"]):
                tag = label_names[tag_id]
                f.write(f"{ch} {tag}\n")
            f.write("\n")

# MSRA 只有 train / test
train_dataset = msra["train"]

# 划分 dev
split = train_dataset.train_test_split(test_size=0.1, seed=42)
train_split = split["train"]
dev_split = split["test"]

save_msra(train_split, os.path.join(ZH_DIR, "train.txt"))
save_msra(dev_split,   os.path.join(ZH_DIR, "dev.txt"))
save_msra(msra["test"], os.path.join(ZH_DIR, "test.txt"))