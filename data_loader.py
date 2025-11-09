import pandas as pd
import re
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# 下载NLTK资源（只在首次运行时需要）
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def simple_preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data():
    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    train_df['clean_text'] = train_df['text'].apply(clean_text)
    test_df['clean_text'] = test_df['text'].apply(clean_text)

    return train_df, test_df
#TODO 数据增强