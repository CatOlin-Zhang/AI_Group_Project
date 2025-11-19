import torch
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def get_model(num_labels=2, model_name='bert-base-uncased'):
    """
    Initialize a BERT model for sequence classification.
    
    Args:
        num_labels: Number of output labels for classification
        model_name: Pre-trained BERT model name
    Returns:
        Initialized BERT model
    """
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=False,  # Disable attention outputs
        output_hidden_states=False  # Disable hidden states outputs
    )
    return model

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The BERT model to train
        dataloader: Training data loader
        optimizer: Optimizer for gradient updates
        scheduler: Learning rate scheduler
        device: Device to run training on (CPU/GPU)
    Returns:
        Average training loss for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move batch data to specified device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Clear previous gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()  # Update weights
        scheduler.step()  # Update learning rate

    return total_loss / len(dataloader)  # Return average loss

def evaluate(model, dataloader, device):
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: The BERT model to evaluate
        dataloader: Evaluation data loader
        device: Device to run evaluation on (CPU/GPU)
    Returns:
        avg_loss: Average evaluation loss
        accuracy: Prediction accuracy
        predictions: List of predicted labels
        true_labels: List of ground truth labels
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(dataloader, desc="Evaluation"):
            # Move batch data to specified device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels
