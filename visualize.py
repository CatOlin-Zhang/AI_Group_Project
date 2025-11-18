import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix

def set_chinese_font():
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    sns.set(font_scale=1.2)

def plot_wordclouds(positive_text, negative_text):
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    wc_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    plt.imshow(wc_pos)
    plt.title('Positive Review WordCloud')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    wc_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
    plt.imshow(wc_neg)
    plt.title('Negative Review WordCloud')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train_losses')
    plt.plot(val_losses, label='val_losses')
    plt.title('train_validation_line')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Verification_accuracy', color='g')
    plt.title('Verification Accuracy Line')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['negative', 'positive'],
                yticklabels=['negative', 'positive'])
    plt.xlabel('pridected label')
    plt.ylabel('true label')
    plt.title('TestSet Confusion Matrix')
    plt.show()

def print_classification_report(y_true, y_pred):
    print("\n分类报告：")
    print(classification_report(y_true, y_pred, target_names=['负面', '正面']))

def print_error_samples(test_df, num_samples=3):
    errors = test_df[test_df['label'] != test_df['pred']]
    print("\n错误预测案例：")
    for i in range(min(num_samples, len(errors))):
        real = '正面' if errors['label'].iloc[i] == 1 else '负面'
        pred = '正面' if errors['pred'].iloc[i] == 1 else '负面'
        print(f"\n样本 {i+1}：")
        print(f"真实标签：{real}")
        print(f"预测标签：{pred}")
        print(f"评论内容：{errors['clean_text'].iloc[i][:300]}...")
def Review_Analyze(train_df , test_df):
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