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
    plt.title('正面评论关键词词云')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    wc_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
    plt.imshow(wc_neg)
    plt.title('负面评论关键词词云')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='验证准确率', color='g')
    plt.title('验证准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'],
                yticklabels=['负面', '正面'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('测试集混淆矩阵')
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
#TODO 更多可视化选择