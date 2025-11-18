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

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from model import get_model, evaluate

# Model Hyperparameter Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 2
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
MAX_LEN = 512
TEST_RATIO = 0.05 # Test set size
RANDOM_SEED = 42

MODEL_FILES = [f"best_bert_model({i}).bin" for i in range(1, 8)]
OUTPUT_MERGED_PATH = "fedavg_merged_bert_model.bin"
RESULT_LOG = "merge_evaluation_results.txt"

# F1 score threshold (models below this value will be excluded)
# Since the minimum accuracy of a binary classification model is 50%, the evaluation threshold should be at least 0.5
# Too low an aggregation model means low-iteration models are synchronized with high-iteration updates,
# which can jeopardize model convergence, so it is adjusted to 0.7 to filter out some poorer models.
MIN_F1_THRESHOLD = 0.70

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



# Weighted Fed_avg algorithm 'Communication-Efficient Learning of Deep Networks from Decentralized Data'
# Since the exact amount of data used for training is not available,
# the vote-based scheme cannot be implemented and is replaced with an F1 score-based scheme
# Bins with higher F1 scores (above the threshold) will be assigned higher weights
def fed_avg_weighted(model_paths, scores, device="cpu"):
    """
    FedAvg based on F1 weighting
    """
    if len(model_paths) != len(scores):
        raise ValueError("The lengths of model_paths and scores must be consistent.")

    valid_pairs = [(p, s) for p, s in zip(model_paths, scores) if os.path.exists(p)]
    if not valid_pairs:
        raise FileNotFoundError("No valid model file")

    model_paths, scores = zip(*valid_pairs)
    scores = np.array(scores)

    if np.all(scores == 0):
        weights = np.ones_like(scores) / len(scores)
    else:
        weights = scores / scores.sum()

    print("Model weight allocation:")
    for i, (path, score, w) in enumerate(zip(model_paths, scores, weights), 1):
        print(f"  Model {i} ({os.path.basename(path)}): F1(macro)={score:.4f}, Weight={w:.4f}")

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
    print(f"Loading IMDb test set (Proportion used: {test_ratio * 100:.1f}%)...")
    dataset = load_dataset("imdb")
    test_df = pd.DataFrame(dataset['test'])
    if test_ratio < 1.0:
        test_df = test_df.sample(frac=test_ratio, random_state=seed).reset_index(drop=True)
    test_df['clean_text'] = test_df['text'].apply(clean_text)
    print(f"Final number of test samples: {len(test_df)}")

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
    Create a subset DataLoader of the IMDb training set (for approximate evaluation of training performance)
    """
    print(f"Load IMDb training dataset subset (Proportion used: {subset_ratio * 100:.1f}%) Used for overfitting analysis...")
    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset['train'])

    # Random sampling (fairer when the same size as the test set)
    train_df = train_df.sample(frac=subset_ratio, random_state=seed).reset_index(drop=True)
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    print(f"Use {len(train_df)} training sample")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = IMDBTestDataset(
        texts=train_df['clean_text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def assess_overfitting(model, train_loader, test_loader, device, model_name="Model"):
    """
    Assessing Model Overfitting

    Returns:
    dict: {'train_acc', 'test_acc', 'train_f1', 'test_f1', 'acc_gap', 'f1_gap'}

    Since model overfitting is defined as performing well on the training set (facing data it has already learned)
    but poorly on the test set (data it has not seen),
    the quantification metric for the degree of overfitting is defined as the difference in accuracy between the training set and the test set.
    """
    print(f"\nAssessing the overfitting risk of  {model_name} ...")
    model.eval()

    # Training subset performance
    _, train_acc, train_preds, train_labels = evaluate(model, train_loader, device)
    train_f1 = f1_score(train_labels, train_preds, average='macro')

    # Test subset performance
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    test_f1 = f1_score(test_labels, test_preds, average='macro')

    acc_gap = train_acc - test_acc
    f1_gap = train_f1 - test_f1

    print(f"{model_name} Training subset: Accuracy={train_acc:.4f}, F1(macro)={train_f1:.4f}")
    print(f"{model_name} Test subset: Accuracy={test_acc:.4f}, F1(macro)={test_f1:.4f}")
    print(f"Accuracy gap (Train - Test): {acc_gap:.4f}")
    print(f"F1 gap (Train - Test): {f1_gap:.4f}")

    overfit = False
    if acc_gap > 0.05 or f1_gap > 0.05:
        print(f"Warning: {model_name} may have significant overfitting!")
        overfit = True
    elif acc_gap > 0.02 or f1_gap > 0.02:
        print(f"Notice: {model_name} has mild overfitting")
    else:
        print(f"{model_name} generalizes well, no significant overfitting")
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'acc_gap': acc_gap,
        'f1_gap': f1_gap,
        'is_overfit': overfit
    }


#main
def main():
    # Create a unified test DataLoader
    TEST_DATALOADER = create_imdb_test_loader(
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    # Create a training subset DataLoader for overfitting analysis of all models
    print("\n" + "=" * 50)
    print("Initializing training subset (for overfitting analysis)...")
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
    single_model_overfit_results = []  # Store the overfitting metrics for each individual model

    print("\nEvaluating the standalone model...\n")
    for i, model_path in enumerate(MODEL_FILES, 1):
        if not os.path.exists(model_path):
            print(f"Skipping {model_path}: file does not exist")
            continue

        print(f"\n--- Model {i}: {model_path} ---")
        model = get_model(num_labels=NUM_LABELS, model_name=MODEL_NAME)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)

        _, acc, preds, labels = evaluate(model, TEST_DATALOADER, DEVICE)
        cm = confusion_matrix(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')

        print(f"Accuracy: {acc:.4f} | F1 (macro): {f1_macro:.4f} | F1 (weighted): {f1_weighted:.4f}")
        print("Confusion Matrix:")
        print(cm)

        # Assess the overfitting of this single model
        overfit_metrics = assess_overfitting(
            model,
            TRAIN_SUB_LOADER,
            TEST_DATALOADER,
            DEVICE,
            model_name=f"Model {i}"
        )
        single_model_overfit_results.append((model_path, overfit_metrics))

        all_predictions.append(preds)
        all_true_labels.append(labels)
        model_f1_macros.append(f1_macro)
        valid_model_paths.append(model_path)

    if not valid_model_paths:
        raise RuntimeError("No valid model available for merging")

    # Filter low-quality models (based on F1)
    filtered_paths = []
    filtered_f1s = []
    for path, f1 in zip(valid_model_paths, model_f1_macros):
        if f1 >= MIN_F1_THRESHOLD:
            filtered_paths.append(path)
            filtered_f1s.append(f1)
        else:
            print(f"Eliminate low-performance models {os.path.basename(path)} (F1={f1:.4f} < {MIN_F1_THRESHOLD})")

    if not filtered_paths:
        print("All models have an F1 score below the threshold, reverting to all models")
        filtered_paths, filtered_f1s = valid_model_paths, model_f1_macros

    # Perform weighted FedAvg (based on F1)
    print(f"\nExecuting weighted FedAvg (assigning weights based on F1(macro), threshold={MIN_F1_THRESHOLD})...")
    merged_sd = fed_avg_weighted(filtered_paths, filtered_f1s, device="cpu")
    torch.save(merged_sd, OUTPUT_MERGED_PATH)
    print(f"Weighted merge completed! Saved to: {OUTPUT_MERGED_PATH}")

    # Evaluate the merged model
    print("\nEvaluating the merged model...")
    merged_model = get_model(num_labels=NUM_LABELS, model_name=MODEL_NAME)
    merged_model.load_state_dict(torch.load(OUTPUT_MERGED_PATH, map_location=DEVICE))
    merged_model.to(DEVICE)

    _, merged_acc, merged_preds, merged_labels = evaluate(merged_model, TEST_DATALOADER, DEVICE)
    merged_cm = confusion_matrix(merged_labels, merged_preds)
    merged_f1_macro = f1_score(merged_labels, merged_preds, average='macro')
    merged_f1_weighted = f1_score(merged_labels, merged_preds, average='weighted')

    print(f"\nFinal result of the merged model:")
    print(f"Accuracy: {merged_acc:.4f} | F1 (macro): {merged_f1_macro:.4f} | F1 (weighted): {merged_f1_weighted:.4f}")
    print("Confusion Matrix:")
    print(merged_cm)

    # Assess overfitting of the combined model
    print("\n" + "=" * 50)
    merged_overfit = assess_overfitting(
        merged_model,
        TRAIN_SUB_LOADER,
        TEST_DATALOADER,
        DEVICE,
        model_name="Combined Model"
    )

    # Comparison of Best Single Model vs Combined Model
    best_single_f1 = max(model_f1_macros)
    best_single_acc = max(
        np.mean(np.array(p) == np.array(l))
        for p, l in zip(all_predictions, all_true_labels)
    )


    print(f"\nBest single model: Accuracy={best_single_acc:.4f}, F1(macro)={best_single_f1:.4f}")
    print(f"Merged model: Accuracy={merged_acc:.4f}, F1(macro)={merged_f1_macro:.4f}")



if __name__ == "__main__":
    main()