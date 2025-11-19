from torch.utils.data import Dataset  # Base class for creating custom datasets in PyTorch
from transformers import BertTokenizer  # BERT tokenizer for text preprocessing (converts text to token IDs)

class IMDBDataset(Dataset):
    """
    A custom PyTorch Dataset class for handling IMDB movie review data.
    This class processes raw text reviews and their corresponding sentiment labels,
    using a BERT tokenizer to convert text into model-friendly input formats (token IDs, attention masks).
    """

    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize the IMDBDataset.

        Args:
            texts (list[str]): List of raw text reviews (e.g., ["Great movie!", "Terrible film..."])
            labels (list[int]): List of corresponding sentiment labels (0 for negative, 1 for positive)
            tokenizer (BertTokenizer): Pre-trained BERT tokenizer instance (e.g., 'bert-base-uncased')
            max_length (int, optional): Maximum sequence length for tokenized text.
                Longer texts will be truncated, shorter ones will be padded. Defaults to 512 (BERT's max limit).
        """
        self.texts = texts  # Store raw text reviews
        self.labels = labels  # Store corresponding sentiment labels
        self.tokenizer = tokenizer  # Store BERT tokenizer for text processing
        self.max_length = max_length  # Store maximum sequence length for consistency

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples (equal to the length of the texts/labels list)
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieve a single sample (text + label) from the dataset by index,
        and process the text into BERT-compatible input formats.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: A dictionary containing:
                - 'input_ids': Tensor of token IDs (after tokenization, padding/truncation)
                - 'attention_mask': Tensor indicating which tokens are real (1) vs padded (0)
                - 'labels': The sentiment label for the sample (0 or 1)
        """
        # Get the raw text and label for the specified index
        text = self.texts[idx]
        label = self.labels[idx]

        # Use BERT tokenizer to process the text into model inputs
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # Add [CLS] at start and [SEP] at end (required for BERT)
            max_length=self.max_length,  # Enforce maximum sequence length
            padding='max_length',  # Pad shorter texts to max_length with [PAD] tokens
            truncation=True,  # Truncate longer texts to max_length
            return_attention_mask=True,  # Return mask to distinguish real vs padded tokens
            return_tensors='pt'  # Return PyTorch tensors (compatible with PyTorch models)
        )

        # Return processed inputs: flatten tensors to remove batch dimension (added by tokenizer)
        return {
            'input_ids': encoding['input_ids'].flatten(),  # Token IDs (shape: [max_length])
            'attention_mask': encoding['attention_mask'].flatten(),  # Attention mask (shape: [max_length])
            'labels': label  # Corresponding sentiment label
        }