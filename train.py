import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from config import *
from data_loader import load_and_preprocess_data, simple_preprocess
from dataset import IMDBDataset
from model import get_model, train_epoch, evaluate
from visualize import set_chinese_font, plot_wordclouds, plot_training_curves, \
                     plot_confusion_matrix, print_classification_report, print_error_samples

def main():
    set_chinese_font()
    print(f"使用设备：{device}")

    # 加载数据
    train_df, test_df = load_and_preprocess_data()

    # 词云分析（小样本）
    sample_size = 1000
    train_sample = train_df.sample(sample_size, random_state=random_state)
    train_sample['clean_for_wc'] = train_sample['text'].apply(simple_preprocess)
    pos_text = ' '.join(train_sample[train_sample['label']==1]['clean_for_wc'])
    neg_text = ' '.join(train_sample[train_sample['label']==0]['clean_for_wc'])
    plot_wordclouds(pos_text, neg_text)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 划分训练/验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['clean_text'].tolist(),
        train_df['label'].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=train_df['label'].tolist()
    )

    # 创建 Dataset 和 DataLoader
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = IMDBDataset(test_df['clean_text'].tolist(), test_df['label'].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型、优化器、调度器
    model = get_model().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练循环
    best_val_accuracy = 0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 10)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(train_loss)
        print(f"训练损失：{train_loss:.4f}")

        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"验证损失：{val_loss:.4f}，验证准确率：{val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("保存最佳模型")

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_accuracies)

    # 测试评估
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\n测试集结果：损失={test_loss:.4f}, 准确率={test_acc:.4f}")

    print_classification_report(test_labels, test_preds)
    plot_confusion_matrix(test_labels, test_preds)

    # 错误样本分析
    test_df_reset = test_df.reset_index(drop=True)
    test_df_reset['pred'] = test_preds
    print_error_samples(test_df_reset)

if __name__ == "__main__":
    main()