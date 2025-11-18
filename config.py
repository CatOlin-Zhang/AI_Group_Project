import torch
# Hyperparameter Configuration
# device set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Path and Parameters
model_name = 'bert-base-uncased'
max_length = 512
batch_size = 16
learning_rate = 2e-5
epsilon = 1e-8
epochs = 3
test_size = 0.1
random_state = 42
best_model_path = 'best_bert_model.bin'

# 可视化中文字体
matplotlib_font = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
