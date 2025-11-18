import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BertTokenizer
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# 静默下载 NLTK 资源
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from model import get_model, evaluate

# ======================
# 配置
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 2
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
MAX_LEN = 512
TEST_RATIO = 0.05 #测试集大小
RANDOM_SEED = 42

MODEL_FILES = [f"best_bert_model({i}).bin" for i in range(1, 8)]
OUTPUT_MERGED_PATH = "fedavg_merged_bert_model.bin"
RESULT_LOG = "merge_evaluation_results.txt"

# 新增：F1 阈值（低于此值的模型将被排除）
MIN_F1_THRESHOLD = 0.70

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ======================
#改进版 FedAvg：支持任意 score（如 F1）
# ======================
def fed_avg_weighted(model_paths, scores, device="cpu"):
    """
    基于任意评分（如 F1）加权的 FedAvg
    """
    if len(model_paths) != len(scores):
        raise ValueError("model_paths 和 scores 长度必须一致")

    valid_pairs = [(p, s) for p, s in zip(model_paths, scores) if os.path.exists(p)]
    if not valid_pairs:
        raise FileNotFoundError("没有有效的模型文件")

    model_paths, scores = zip(*valid_pairs)
    scores = np.array(scores)

    if np.all(scores == 0):
        weights = np.ones_like(scores) / len(scores)
    else:
        weights = scores / scores.sum()

    print("各模型权重分配:")
    for i, (path, score, w) in enumerate(zip(model_paths, scores, weights), 1):
        print(f"  模型 {i} ({os.path.basename(path)}): F1(macro)={score:.4f}, 权重={w:.4f}")

    state_dicts = []
    for path in model_paths:
        sd = torch.load(path, map_location=device)
        state_dicts.append(sd)

    merged_sd = {}
    for key in state_dicts[0]:
        merged_sd[key] = sum(w * sd[key].to(torch.float32) for w, sd in zip(weights, state_dicts))

    return merged_sd


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class IMDBTestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_imdb_test_loader(batch_size=16, max_len=512, test_ratio=1.0, seed=42):
    print(f"正在加载 IMDb 官方测试集 (使用比例: {test_ratio * 100:.1f}%)...")
    dataset = load_dataset("imdb")
    test_df = pd.DataFrame(dataset['test'])
    if test_ratio < 1.0:
        test_df = test_df.sample(frac=test_ratio, random_state=seed).reset_index(drop=True)
    test_df['clean_text'] = test_df['text'].apply(clean_text)
    print(f"最终测试样本数: {len(test_df)}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = IMDBTestDataset(
        texts=test_df['clean_text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def create_imdb_train_subset_loader(batch_size=16, max_len=512, subset_ratio=0.05, seed=42):
    """
    创建 IMDb 训练集的子集 DataLoader（用于近似评估训练性能）
    """
    print(f"加载 IMDb 训练集子集 (比例: {subset_ratio * 100:.1f}%) 用于过拟合分析...")
    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset['train'])

    # 随机采样（与测试集同规模更公平）
    train_df = train_df.sample(frac=subset_ratio, random_state=seed).reset_index(drop=True)
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    print(f"使用 {len(train_df)} 条训练样本")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = IMDBTestDataset(
        texts=train_df['clean_text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def assess_overfitting(model, train_loader, test_loader, device, model_name="模型"):
    """
    评估模型过拟合情况

    Returns:
        dict: {'train_acc', 'test_acc', 'train_f1', 'test_f1', 'acc_gap', 'f1_gap'}
    """
    print(f"\n正在评估 {model_name} 的过拟合风险...")
    model.eval()

    # 训练子集性能
    _, train_acc, train_preds, train_labels = evaluate(model, train_loader, device)
    train_f1 = f1_score(train_labels, train_preds, average='macro')

    # 测试子集性能
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    test_f1 = f1_score(test_labels, test_preds, average='macro')

    acc_gap = train_acc - test_acc
    f1_gap = train_f1 - test_f1

    print(f"{model_name} 训练子集: 准确率={train_acc:.4f}, F1(macro)={train_f1:.4f}")
    print(f"{model_name} 测试子集: 准确率={test_acc:.4f}, F1(macro)={test_f1:.4f}")
    print(f"准确率差距 (Train - Test): {acc_gap:.4f}")
    print(f"F1 差距 (Train - Test): {f1_gap:.4f}")

    overfit = False
    if acc_gap > 0.05 or f1_gap > 0.05:
        print(f"警告：{model_name} 可能存在明显过拟合！")
        overfit = True
    elif acc_gap > 0.02 or f1_gap > 0.02:
        print(f"提示：{model_name} 存在轻微过拟合")
    else:
        print(f"{model_name} 泛化良好，无显著过拟合")

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'acc_gap': acc_gap,
        'f1_gap': f1_gap,
        'is_overfit': overfit
    }
# ======================
# 主程序
# ======================
def main():
    # 创建统一测试 DataLoader
    TEST_DATALOADER = create_imdb_test_loader(
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    # 创建一次训练子集 DataLoader，用于所有模型的过拟合分析
    print("\n" + "=" * 50)
    print("初始化训练子集（用于过拟合分析）...")
    TRAIN_SUB_LOADER = create_imdb_train_subset_loader(
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        subset_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    all_predictions = []
    all_true_labels = []
    model_f1_macros = []
    valid_model_paths = []
    single_model_overfit_results = []  # 存储每个单模型的过拟合指标

    print("\n正在评估独立模型...\n")
    for i, model_path in enumerate(MODEL_FILES, 1):
        if not os.path.exists(model_path):
            print(f"跳过 {model_path}：文件不存在")
            continue

        print(f"\n--- 模型 {i}: {model_path} ---")
        model = get_model(num_labels=NUM_LABELS, model_name=MODEL_NAME)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)

        _, acc, preds, labels = evaluate(model, TEST_DATALOADER, DEVICE)
        cm = confusion_matrix(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')

        print(f"准确率: {acc:.4f} | F1 (macro): {f1_macro:.4f} | F1 (weighted): {f1_weighted:.4f}")
        print("混淆矩阵:")
        print(cm)

        # 评估该单模型的过拟合情况
        overfit_metrics = assess_overfitting(
            model,
            TRAIN_SUB_LOADER,
            TEST_DATALOADER,
            DEVICE,
            model_name=f"模型 {i}"
        )
        single_model_overfit_results.append((model_path, overfit_metrics))

        all_predictions.append(preds)
        all_true_labels.append(labels)
        model_f1_macros.append(f1_macro)
        valid_model_paths.append(model_path)

    if not valid_model_paths:
        raise RuntimeError("无有效模型用于合并")

    # ==============================
    # 过滤低质量模型（基于 F1）
    # ==============================
    filtered_paths = []
    filtered_f1s = []
    for path, f1 in zip(valid_model_paths, model_f1_macros):
        if f1 >= MIN_F1_THRESHOLD:
            filtered_paths.append(path)
            filtered_f1s.append(f1)
        else:
            print(f"排除低性能模型 {os.path.basename(path)} (F1={f1:.4f} < {MIN_F1_THRESHOLD})")

    if not filtered_paths:
        print("所有模型 F1 均低于阈值，回退到全部模型")
        filtered_paths, filtered_f1s = valid_model_paths, model_f1_macros

    # ==============================
    # 执行加权 FedAvg（基于 F1）
    # ==============================
    print(f"\n执行加权 FedAvg（按 F1(macro) 分配权重，阈值={MIN_F1_THRESHOLD})...")
    merged_sd = fed_avg_weighted(filtered_paths, filtered_f1s, device="cpu")
    torch.save(merged_sd, OUTPUT_MERGED_PATH)
    print(f"加权合并完成！已保存至: {OUTPUT_MERGED_PATH}")

    # ==============================
    # 评估合并模型
    # ==============================
    print("\n正在评估合并后的模型...")
    merged_model = get_model(num_labels=NUM_LABELS, model_name=MODEL_NAME)
    merged_model.load_state_dict(torch.load(OUTPUT_MERGED_PATH, map_location=DEVICE))
    merged_model.to(DEVICE)

    _, merged_acc, merged_preds, merged_labels = evaluate(merged_model, TEST_DATALOADER, DEVICE)
    merged_cm = confusion_matrix(merged_labels, merged_preds)
    merged_f1_macro = f1_score(merged_labels, merged_preds, average='macro')
    merged_f1_weighted = f1_score(merged_labels, merged_preds, average='weighted')

    print(f"\n合并模型最终结果:")
    print(f"确率: {merged_acc:.4f} | F1 (macro): {merged_f1_macro:.4f} | F1 (weighted): {merged_f1_weighted:.4f}")
    print("混淆矩阵:")
    print(merged_cm)

    # 评估合并模型的过拟合
    print("\n" + "=" * 50)
    merged_overfit = assess_overfitting(
        merged_model,
        TRAIN_SUB_LOADER,
        TEST_DATALOADER,
        DEVICE,
        model_name="合并模型"
    )

    # ==============================
    # 对比最佳单模型 vs 合并模型
    # ==============================
    best_single_f1 = max(model_f1_macros)
    best_single_acc = max(
        np.mean(np.array(p) == np.array(l))
        for p, l in zip(all_predictions, all_true_labels)
    )

    print(f"\n最佳单模型: 准确率={best_single_acc:.4f}, F1(macro)={best_single_f1:.4f}")
    print(f"合并模型:   准确率={merged_acc:.4f}, F1(macro)={merged_f1_macro:.4f}")



if __name__ == "__main__":
    main()