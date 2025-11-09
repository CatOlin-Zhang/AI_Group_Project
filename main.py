import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 下载NLTK资源
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
sns.set(font_scale=1.2)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 方法1：使用datasets库加载
from datasets import load_dataset
dataset = load_dataset("imdb")
# 转换为DataFrame
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# 简单文本预处理（用于关键词分析）
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def simple_preprocess(text):
    # 移除HTML标签和特殊字符
    text = re.sub(r'<.*?>', '', text)  # 移除HTML标签
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # 保留字母
    text = text.lower()  # 小写化
    words = text.split()
    # 去停用词和词形还原
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# 对部分样本进行预处理（全量处理耗时较长）
sample_size = 1000
train_sample = train_df.sample(sample_size, random_state=42)
train_sample['clean_text'] = train_sample['text'].apply(simple_preprocess)

# 分别提取正负情感的关键词
positive_text = ' '.join(train_sample[train_sample['label']==1]['clean_text'])
negative_text = ' '.join(train_sample[train_sample['label']==0]['clean_text'])

# 绘制词云
from wordcloud import WordCloud

plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.imshow(wordcloud_pos)
plt.title('正面评论关键词词云')
plt.axis('off')

plt.subplot(1, 2, 2)
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
plt.imshow(wordcloud_neg)
plt.title('负面评论关键词词云')
plt.axis('off')

plt.tight_layout()
plt.show()

def clean_text(text):
    """轻量级文本清洗，保留Transformer需要的信息"""
    text = re.sub(r'<.*?>', '', text)  # 移除HTML标签（数据集中常见）
    text = re.sub(r'http\S+', '', text)  # 移除URL
    text = re.sub(r'@\w+', '', text)  # 移除@提及
    text = re.sub(r'\s+', ' ', text).strip()  # 合并多余空格
    return text

# 清洗文本
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)

# 加载BERT基础模型的tokenizer
model_name = 'bert-base-uncased'  # 小写化的BERT基础模型
tokenizer = BertTokenizer.from_pretrained(model_name)

# 测试tokenizer
sample_text = train_df['clean_text'].iloc[0]
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"原始文本：{sample_text[:100]}...")
print(f"分词结果：{tokens[:10]}...")
print(f"token ID：{token_ids[:10]}...")

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length  # BERT基础模型最大长度为512

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 编码文本
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # 添加[CLS]和[SEP]
            max_length=self.max_length,
            padding='max_length',  # 填充到max_length
            truncation=True,  # 截断长文本
            return_attention_mask=True,
            return_tensors='pt'  # 返回PyTorch张量
        )

        # 转换为单维度张量（移除批次维度）
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 划分验证集（从训练集中取10%）
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['clean_text'].tolist(),
    train_df['label'].tolist(),
    #TODO 数据集切分(默认0.1)
    test_size=0.1,
    random_state=42,
    stratify=train_df['label'].tolist()  # 保持情感分布一致
)

# 创建数据集
train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
val_dataset = IMDBDataset(val_texts, val_labels, tokenizer)
test_dataset = IMDBDataset(test_df['clean_text'].tolist(), test_df['label'].tolist(), tokenizer)

# 创建数据加载器（支持批次处理和打乱）
batch_size = 16  # 根据GPU内存调整，16适合12GB显存
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载带分类头的BERT模型
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # 二分类（正面/负面）
    output_attentions=False,  # 不返回注意力权重
    output_hidden_states=False  # 不返回隐藏状态
)

# 移动模型到GPU（如果可用）
model = model.to(device)

# 优化器参数
learning_rate = 2e-5  # BERT推荐学习率（远小于传统神经网络）
epsilon = 1e-8  # AdamW的数值稳定性参数

# 定义优化器
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    eps=epsilon
)

# 学习率调度器（线性预热后线性衰减）
#TODO 全局训练轮次(默认3轮)
epochs = 3  # 预训练模型收敛快，3-5轮足够
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # 预热步数
    num_training_steps=total_steps
)

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()  # 训练模式
    total_loss = 0

    for batch in tqdm(dataloader, desc="训练"):
        # 提取批次数据并移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 重置梯度
        model.zero_grad()

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 计算损失
        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播和参数更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        scheduler.step()

    # 平均损失
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()  # 评估模式
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():  # 禁用梯度计算
        for batch in tqdm(dataloader, desc="评估"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 累计损失
            total_loss += outputs.loss.item()

            # 获取预测结果（取logits的最大值索引）
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)

            # 真实标签
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    return avg_loss, accuracy, predictions, true_labels

# 训练模型
best_val_accuracy = 0
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 10)

    # 训练
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    train_losses.append(train_loss)
    print(f"训练损失：{train_loss:.4f}")

    # 验证
    val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"验证损失：{val_loss:.4f}，验证准确率：{val_acc:.4f}")

    # 保存最佳模型
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'best_bert_model.bin')
        print("保存最佳模型")

# 绘制训练曲线
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

# 加载最佳模型权重
model.load_state_dict(torch.load('best_bert_model.bin'))

# 在测试集上评估
test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
print(f"\n测试集结果：")
print(f"损失：{test_loss:.4f}，准确率：{test_acc:.4f}")

# 分类报告（精确率、召回率、F1分数）
print("\n分类报告：")
print(classification_report(test_labels, test_preds, target_names=['负面', '正面']))

# 绘制混淆矩阵
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['负面', '正面'],
            yticklabels=['负面', '正面'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('测试集混淆矩阵')
plt.show()

# 获取错误预测的样本
test_df['pred'] = test_preds
errors = test_df[test_df['label'] != test_df['pred']]

# 打印几个错误案例
print("\n错误预测案例：")
for i in range(3):
    print(f"\n样本 {i+1}：")
    print(f"真实标签：{'正面' if errors['label'].iloc[i] == 1 else '负面'}")
    print(f"预测标签：{'正面' if errors['pred'].iloc[i] == 1 else '负面'}")
    print(f"评论内容：{errors['clean_text'].iloc[i][:300]}...")



