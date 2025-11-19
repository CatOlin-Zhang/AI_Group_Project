#!pip install pandas numpy matplotlib seaborn nltk scikit-learn torch transformers datasets
#pip install hf_xet

import pandas as pd  # Library for data manipulation and analysis
import re  # Library for regular expression operations
from datasets import load_dataset  # Function to load datasets from Hugging Face Hub
from nltk.corpus import stopwords  # Module containing stopwords (common uninformative words)
from nltk.stem import WordNetLemmatizer  # Tool for lemmatization (reducing words to their base form)
import nltk  # Natural Language Toolkit for text processing

# Download NLTK resources (only required for the first run)
# stopwords: Contains common English stopwords (e.g., "the", "and")
# wordnet: Lexical database used for lemmatization
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize set of English stopwords for quick lookup
stop_words = set(stopwords.words('english'))
# Initialize lemmatizer to reduce words to their base form (e.g., "running" → "run")
lemmatizer = WordNetLemmatizer()


def simple_preprocess(text):
    """
    Perform basic text preprocessing for NLP tasks.

    Steps include:
    1. Remove HTML tags using regular expressions
    2. Keep only alphabetic characters, replace other characters with spaces
    3. Convert all text to lowercase
    4. Lemmatize each word and remove stopwords
    5. Join processed words back into a single string

    Args:
        text (str): Input text to be preprocessed

    Returns:
        str: Preprocessed text
    """
    # Remove HTML tags (e.g., <br>, <div>)
    text = re.sub(r'<.*?>', '', text)
    # Keep only alphabetic characters (a-z, A-Z), replace others with space
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert all characters to lowercase to ensure case consistency
    text = text.lower()
    # Split text into words, lemmatize each word, and exclude stopwords
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    # Join processed words back into a single string with spaces
    return ' '.join(words)


def clean_text(text):
    """
    Perform text cleaning focused on removing noise and standardizing format.

    Steps include:
    1. Remove HTML tags
    2. Remove URLs (web addresses)
    3. Remove user mentions (e.g., @username)
    4. Normalize whitespace (replace multiple spaces with single space)
    5. Strip leading/trailing whitespace

    Args:
        text (str): Input text to be cleaned

    Returns:
        str: Cleaned text
    """
    # Remove HTML tags (e.g., <br>, <div>)
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs (matches patterns starting with http:// or https://)
    text = re.sub(r'http\S+', '', text)
    # Remove user mentions (matches patterns starting with @ followed by word characters)
    text = re.sub(r'@\w+', '', text)
    # Replace multiple consecutive spaces with a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_data():
    """
    Load the IMDB dataset and apply text cleaning to prepare for analysis.

    Steps include:
    1. Load the IMDB dataset (movie reviews with sentiment labels)
    2. Convert training and test splits to pandas DataFrames
    3. Apply the clean_text function to create a "clean_text" column
    4. Return processed training and test DataFrames

    Returns:
        tuple: (train_df, test_df) where both are pandas DataFrames containing
               original text, labels, and cleaned text
    """
    # Load the IMDB dataset from Hugging Face Hub (contains 'train' and 'test' splits)
    dataset = load_dataset("imdb")

    split = dataset["train"].train_test_split(test_size=0.1, seed=42)

    train_dataset = split["train"]
    val_dataset = split["test"]

    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)
    test_df = pd.DataFrame(dataset['test'])

    # Convert training split to pandas DataFrame for easier manipulation
    #train_df = pd.DataFrame(dataset['train'])
    # Convert test split to pandas DataFrame
    #test_df = pd.DataFrame(dataset['test'])

    # Apply clean_text function to the 'text' column and store results in 'clean_text'
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    val_df['clean_text'] = val_df['text'].apply(clean_text)
    test_df['clean_text'] = test_df['text'].apply(clean_text)

    # Return processed training and test dataframes
    return train_df, test_df, val_df


