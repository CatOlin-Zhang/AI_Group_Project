import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import re
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')


def set_english_font():
    """Set English font for plots"""
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
    sns.set(font_scale=1.2)


def set_chinese_font():
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    sns.set(font_scale=1.2)


def diagnose_tokenization_issues(train_df):
    """
    Diagnose tokenization and preprocessing issues
    """
    print("=== Tokenization Issue Diagnosis ===")

    # Check for HTML tags in original text
    sample_texts = train_df['text'].head(5)

    for i, text in enumerate(sample_texts):
        print(f"\n--- Sample {i+1} ---")
        print(f"Original text preview: {text[:200]}...")

        # Check for HTML tags
        html_tags = re.findall(r'<[^>]+>', text)
        if html_tags:
            print(f"Found HTML tags: {set(html_tags)}")

        # Check for "br" in text
        if 'br' in text.lower():
            print("Text contains 'br'")

        # Show tokenization results - with fallback mechanism
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
            print(f"NLTK tokenization result ({len(tokens)} tokens):")
            print(tokens[:30])  # Show first 30 tokens

            # Check for problematic tokens in results
            problem_tokens = [token for token in tokens if token.lower()
                              in ['br', 'nbsp', 'amp', 'lt', 'gt', 'quot']]
            if problem_tokens:
                print(
                    f"Tokenization contains problematic tokens: {set(problem_tokens)}")
            else:
                print("Tokenization is clean")

        except Exception as e:
            print(f"NLTK tokenization failed: {e}")
            # Use simple space tokenization as fallback
            simple_tokens = text.split()
            print(
                f"Simple space tokenization ({len(simple_tokens)} tokens): {simple_tokens[:20]}...")

            # Check for problems in simple tokenization
            problem_tokens = [token for token in simple_tokens if any(
                problem in token.lower() for problem in ['br', 'nbsp', 'amp'])]
            if problem_tokens:
                print(
                    f"Simple tokenization contains problematic tokens: {set(problem_tokens)}")

    # Count problematic word frequency - using simple tokenization to ensure it runs
    print("\n=== High Frequency Word Analysis ===")
    all_words = []
    for text in train_df['text'].head(100):  # Check first 100 samples
        words = text.split()
        all_words.extend(words)

    from collections import Counter
    word_freq = Counter(all_words)

    # Check for problematic words
    problem_patterns = ['br', 'nbsp', '&', '<', '>', 'quot', 'amp', 'lt', 'gt']
    problem_words = {}

    for word, freq in word_freq.most_common(50):  # Check top 50 frequent words
        for pattern in problem_patterns:
            if pattern in word.lower():
                problem_words[word] = freq
                break

    if problem_words:
        print(f"Found {len(problem_words)} types of problematic words:")
        for word, freq in sorted(problem_words.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{word}': {freq} times")
    else:
        print("No common problematic words found")

    # Show top 20 frequent words
    print(f"\nTop 20 frequent words:")
    for word, freq in word_freq.most_common(20):
        print(f"  {word}: {freq} times")

    return word_freq


def clean_text_for_wordcloud(text):
    """
    Clean text specifically for word clouds, remove HTML tags and meaningless words
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove common HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)

    # Remove specific meaningless words
    unwanted_words = ['br', 'nbsp', 'amp', 'lt', 'gt', 'quot']
    for word in unwanted_words:
        text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def plot_wordclouds(positive_text, negative_text):
    """Generate word clouds for positive and negative reviews"""
    # Clean text
    positive_text_cleaned = clean_text_for_wordcloud(positive_text)
    negative_text_cleaned = clean_text_for_wordcloud(negative_text)

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    wc_pos = WordCloud(width=800, height=400,
                       background_color='white').generate(positive_text_cleaned)  # Use cleaned text
    plt.imshow(wc_pos)
    plt.title('Positive Reviews Word Cloud')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    wc_neg = WordCloud(width=800, height=400,
                       background_color='white').generate(negative_text_cleaned)  # Use cleaned text
    plt.imshow(wc_neg)
    plt.title('Negative Reviews Word Cloud')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_training_progress(train_losses, val_losses, val_accuracies, model_name="Model"):
    """Plot enhanced training progress with multiple subplots"""
    set_english_font()

    plt.figure(figsize=(15, 5))

    # Subplot 1: Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'{model_name} - Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Accuracy curve
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
    plt.title(f'{model_name} - Accuracy Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses, val_losses, val_accuracies):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='g')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Test Set Confusion Matrix')
    plt.show()


def print_classification_report(y_true, y_pred):
    """Print classification report"""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
          target_names=['Negative', 'Positive']))


def analyze_error_patterns(test_df, num_samples=5):
    """Analyze error patterns in predictions"""
    errors = test_df[test_df['label'] != test_df['pred']].copy()

    print("=== Error Analysis ===")
    print(f"Total errors: {len(errors)}")
    print(f"Error rate: {len(errors)/len(test_df):.4f}")

    # Error type analysis
    false_positives = errors[errors['label'] == 0].copy()
    false_negatives = errors[errors['label'] == 1].copy()

    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")

    # Use cleaned text for length calculation
    errors['cleaned_text'] = errors['text'].apply(clean_text_for_wordcloud)
    correct_samples = test_df[test_df['label'] == test_df['pred']]
    correct_samples['cleaned_text'] = correct_samples['text'].apply(
        clean_text_for_wordcloud)

    # Analyze length characteristics of error samples
    error_lengths = errors['cleaned_text'].apply(lambda x: len(x.split()))
    correct_lengths = correct_samples['cleaned_text'].apply(
        lambda x: len(x.split()))

    print("\nLength Statistics:")
    print(f"Error samples avg length: {error_lengths.mean():.1f} words")
    print(f"Correct samples avg length: {correct_lengths.mean():.1f} words")

    # Print some error samples
    print(f"\nFirst {min(num_samples, len(errors))} error samples:")
    for i in range(min(num_samples, len(errors))):
        real = 'Positive' if errors['label'].iloc[i] == 1 else 'Negative'
        pred = 'Positive' if errors['pred'].iloc[i] == 1 else 'Negative'
        cleaned_text = errors['cleaned_text'].iloc[i]
        print(f"\nError Sample {i+1}:")
        print(f"  True: {real}, Predicted: {pred}")
        print(f"  Text: {cleaned_text[:200]}...")

    return false_positives, false_negatives


def print_error_samples(test_df, num_samples=3):
    """Print misclassified samples"""
    errors = test_df[test_df['label'] != test_df['pred']].copy()
    print("\nMisclassified Samples:")
    # Clean error sample text
    errors['cleaned_text'] = errors['text'].apply(clean_text_for_wordcloud)

    for i in range(min(num_samples, len(errors))):
        real = 'Positive' if errors['label'].iloc[i] == 1 else 'Negative'
        pred = 'Positive' if errors['pred'].iloc[i] == 1 else 'Negative'
        cleaned_text = errors['cleaned_text'].iloc[i]
        print(f"\nSample {i+1}:")
        print(f"True Label: {real}")
        print(f"Predicted Label: {pred}")
        print(f"Review Text: {cleaned_text[:300]}...")


def plot_review_length_analysis(train_df, test_df):
    """
    Enhanced review length analysis - integrates best practices from both codes
    """
    set_english_font()

    train_df['cleaned_text'] = train_df['text'].apply(clean_text_for_wordcloud)
    test_df['cleaned_text'] = test_df['text'].apply(clean_text_for_wordcloud)

    # Calculate length in characters (after cleaning)
    train_df['char_length'] = train_df['cleaned_text'].apply(lambda x: len(x))
    test_df['char_length'] = test_df['cleaned_text'].apply(lambda x: len(x))

    # Calculate length in words (after cleaning)
    train_df['word_length'] = train_df['cleaned_text'].apply(
        lambda x: len(x.split()))
    test_df['word_length'] = test_df['cleaned_text'].apply(
        lambda x: len(x.split()))

    plt.figure(figsize=(15, 10))

    # Subplot 1: Character length distribution
    plt.subplot(2, 3, 1)
    sns.histplot(train_df['char_length'], kde=True, bins=50)
    plt.axvline(train_df['char_length'].mean(), color='r', linestyle='--',
                label=f'Mean: {train_df["char_length"].mean():.0f} chars')
    plt.title('Character Length Distribution')
    plt.xlabel('Character Count')
    plt.legend()

    # Subplot 2: Word length distribution
    plt.subplot(2, 3, 2)
    sns.histplot(train_df['word_length'], kde=True, bins=50)
    plt.axvline(train_df['word_length'].mean(), color='r', linestyle='--',
                label=f'Mean: {train_df["word_length"].mean():.0f} words')
    plt.title('Word Length Distribution')
    plt.xlabel('Word Count')
    plt.legend()

    # Subplot 3: Character length by sentiment with log scale
    plt.subplot(2, 3, 3)
    sns.boxplot(x='label', y='char_length', data=train_df)
    plt.yscale('log')  # Logarithmic scale for better visualization
    plt.title('Character Length by Sentiment (Log Scale)')
    plt.xlabel('Sentiment (0=Negative, 1=Positive)')
    plt.ylabel('Character Count')

    # Subplot 4: Word length by sentiment
    plt.subplot(2, 3, 4)
    sns.boxplot(x='label', y='word_length', data=train_df)
    plt.title('Word Length by Sentiment')
    plt.xlabel('Sentiment (0=Negative, 1=Positive)')
    plt.ylabel('Word Count')

    # Subplot 5: Comparison histogram
    plt.subplot(2, 3, 5)
    positive_words = train_df[train_df['label'] == 1]['word_length']
    negative_words = train_df[train_df['label'] == 0]['word_length']
    plt.hist(positive_words, bins=30, alpha=0.7,
             label='Positive', color='green')
    plt.hist(negative_words, bins=30, alpha=0.7, label='Negative', color='red')
    plt.title('Word Length: Positive vs Negative')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend()

    # Subplot 6: Statistics summary
    plt.subplot(2, 3, 6)
    stats_text = f"""Statistics Summary (Cleaned Text):
    
Character Length:
- Positive: {train_df[train_df['label']==1]['char_length'].mean():.0f} chars
- Negative: {train_df[train_df['label']==0]['char_length'].mean():.0f} chars

Word Length:
- Positive: {train_df[train_df['label']==1]['word_length'].mean():.0f} words  
- Negative: {train_df[train_df['label']==0]['word_length'].mean():.0f} words

Total Samples: {len(train_df)}"""

    plt.text(0.1, 0.5, stats_text, fontsize=10,
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    plt.title('Statistical Summary (Cleaned)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def demo_model_predictions(sample_texts):
    """Demo model predictions - interactive functionality"""
    set_english_font()

    predictions = []
    confidences = []

    for text in sample_texts:
        # Simplified demo - in real scenario, call actual model
        pred = "Positive" if len(
            text) % 2 == 0 else "Negative"  # Mock prediction
        confidence = 0.85  # Mock confidence

        predictions.append(pred)
        confidences.append(confidence)

    # Create prediction visualization
    plt.figure(figsize=(12, 6))
    colors = ['green' if p == 'Positive' else 'red' for p in predictions]

    plt.bar(range(len(sample_texts)), confidences, color=colors, alpha=0.7)
    plt.xticks(range(len(sample_texts)), [
               f'Sample {i+1}' for i in range(len(sample_texts))])
    plt.ylabel('Prediction Confidence')
    plt.title('Model Prediction Demo')
    plt.ylim(0, 1)

    # Add prediction labels
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        plt.text(i, conf + 0.02, pred, ha='center',
                 va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def run_enhanced_visualizations(train_df, val_df, test_df,
                                train_losses=None, val_losses=None, val_accuracies=None,
                                y_true=None, y_pred=None, model_name="Model"):
    """
    Enhanced visualization main function
    """
    set_english_font()

    print(f"=== {model_name} Enhanced Visualization Analysis ===")

    # 0. Tokenization analysis
    print("\n0. Tokenization and Cleaning Diagnosis...")
    diagnose_tokenization_issues(train_df)

    # 1. Enhanced data exploration (integrated version)
    print("\n1. Enhanced Data Exploration...")
    plot_review_length_analysis(train_df, test_df)

    # 2. Word cloud analysis
    print("\n2. Word Cloud Analysis...")
    positive_texts = ' '.join(
        train_df[train_df['label'] == 1]['text'].tolist())
    negative_texts = ' '.join(
        train_df[train_df['label'] == 0]['text'].tolist())
    plot_wordclouds(positive_texts, negative_texts)

    # 3. Training process analysis
    if train_losses and val_losses and val_accuracies:
        print("\n3. Training Process Analysis...")
        plot_training_progress(train_losses, val_losses,
                               val_accuracies, model_name)

    # 4. Performance evaluation and error analysis
    if y_true is not None and y_pred is not None:
        print("\n4. Enhanced Performance Evaluation...")
        plot_confusion_matrix(y_true, y_pred)
        print_classification_report(y_true, y_pred)

        # Error pattern analysis
        test_df_copy = test_df.copy()
        test_df_copy['pred'] = y_pred

        # Use clean_text if available, otherwise use text
        text_column = 'clean_text' if 'clean_text' in test_df_copy.columns else 'text'
        test_df_copy['display_text'] = test_df_copy[text_column]

        false_positives, false_negatives = analyze_error_patterns(test_df_copy)
        print_error_samples(test_df_copy)

        return false_positives, false_negatives


def Review_Analyze(train_df, test_df):
    # Calculate comment length (in characters)
    train_df['length'] = train_df['text'].apply(lambda x: len(x))
    test_df['length'] = test_df['text'].apply(lambda x: len(x))

    plt.figure(figsize=(12, 5))
    # Training set length distribution
    plt.subplot(1, 2, 1)
    sns.histplot(train_df['length'], kde=True, bins=50)
    plt.axvline(train_df['length'].mean(), color='r', linestyle='--',
                label=f"mean value:{train_df['length'].mean():.0f}")
    plt.title('Training Set Review Length')
    plt.xlabel('Character Count')
    plt.legend()

    # Length comparison for different sentiments
    plt.subplot(1, 2, 2)
    sns.boxplot(x='label', y='length', data=train_df)
    # Logarithmic scale to reduce the impact of extreme values
    plt.yscale('log')
    plt.title('Comparison of Review Lengths for Different Sentiment')
    plt.xlabel('Sentiment(0=negtive,1=positive)')
    plt.ylabel('Character Number')

    plt.tight_layout()
    plt.show()
