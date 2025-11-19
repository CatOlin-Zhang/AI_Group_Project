import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from config import *
from data_loader import load_and_preprocess_data, simple_preprocess
from dataset import IMDBDataset
from model import get_model, train_epoch, evaluate
from visualize import set_chinese_font, plot_wordclouds, plot_training_curves, \
                     plot_confusion_matrix, print_classification_report, print_error_samples, Review_Analyze

def main():
    # Initialize Chinese font for visualization
    set_chinese_font()
    print(f"Using device: {device}")

    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data()
    
    # Perform exploratory data analysis
    Review_Analyze(train_df, test_df)
    
    # Generate word clouds for positive and negative reviews
    sample_size = 1000
    train_sample = train_df.sample(sample_size, random_state=random_state)
    train_sample['clean_for_wc'] = train_sample['text'].apply(simple_preprocess)
    pos_text = ' '.join(train_sample[train_sample['label']==1]['clean_for_wc'])
    neg_text = ' '.join(train_sample[train_sample['label']==0]['clean_for_wc'])
    plot_wordclouds(pos_text, neg_text)

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Split training data into train/validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['clean_text'].tolist(),
        train_df['label'].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=train_df['label'].tolist()
    )

    # Create datasets and data loaders
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = IMDBDataset(test_df['clean_text'].tolist(), test_df['label'].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = get_model().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    best_val_accuracy = 0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 10)

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")

        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model")

    # Plot training history
    plot_training_curves(train_losses, val_losses, val_accuracies)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\nTest Results: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")

    # Generate performance reports and visualizations
    print_classification_report(test_labels, test_preds)
    plot_confusion_matrix(test_labels, test_preds)

    # Analyze misclassified samples
    test_df_reset = test_df.reset_index(drop=True)
    test_df_reset['pred'] = test_preds
    print_error_samples(test_df_reset)

if __name__ == "__main__":
    main()
