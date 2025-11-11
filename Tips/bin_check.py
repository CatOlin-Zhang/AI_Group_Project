import torch

sd1 = torch.load("best_bert_model.bin", map_location="cpu")
sd2 = torch.load("best_bert_model(2).bin", map_location="cpu")

assert sd1.keys() == sd2.keys(), "参数名不一致！"
for k in sd1:
    assert sd1[k].shape == sd2[k].shape, f"参数 {k} 形状不一致！"
print("两个模型结构完全兼容，可以合并。")