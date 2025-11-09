# 交互式检测
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification

# ----------------------------
# 配置（需与训练时一致）
# ----------------------------
MODEL_NAME = 'bert-base-uncased'
MODEL_PATH = 'best_bert_model.bin'
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 文本预处理函数（与训练一致）
# ----------------------------
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)          # 移除HTML标签
    text = re.sub(r'http\S+', '', text)        # 移除URL
    text = re.sub(r'@\w+', '', text)           # 移除@提及
    text = re.sub(r'\s+', ' ', text).strip()   # 合并空格
    return text

# ----------------------------
# 加载模型和tokenizer
# ----------------------------
print("正在加载模型和tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"模型已加载到设备：{DEVICE}")

# ----------------------------
# 推理函数
# ----------------------------
def predict_sentiment(text: str) -> str:
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return "正面" if prediction == 1 else "负面"

# ----------------------------
# 交互式循环
# ----------------------------
print("\n 欢迎使用 IMDB 情感分析系统！")
print("请输入一段英文电影评论，模型将判断其情感倾向。")
print("输入 'quit' 退出。\n")

while True:
    user_input = input("请输入评论: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("再见！")
        break
    if not user_input:
        print("输入不能为空，请重试。")
        continue

    try:
        result = predict_sentiment(user_input)
        print(f"预测结果：{result}\n")
    except Exception as e:
        print(f"推理出错：{e}\n")