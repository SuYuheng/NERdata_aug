from transformers import BertTokenizer, BertForMaskedLM
import torch

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('../pre_model/bert-uncased')
model = BertForMaskedLM.from_pretrained('../pre_model/bert-uncased')
model.eval()  # 设置为评估模式
# 准备输入，句子中有两个掩码
text = "HummingBad also has the [MASK] to [MASK] code into Google Play"
# 对输入进行编码，将文本转换为模型需要的格式
input_ids = tokenizer.encode(text, return_tensors="pt")

# 预测所有tokens的分数
with torch.no_grad():  # 关闭梯度计算
    outputs = model(input_ids)
    predictions = outputs[0]

# 找到所有[MASK]的位置
mask_token_indices = torch.where(input_ids == tokenizer.mask_token_id)[1]

# 存储所有掩码位置的预测结果
all_predictions = []
for mask_index in mask_token_indices:
    mask_token_logits = predictions[0, mask_index, :]
    top_1_token = torch.topk(mask_token_logits, 2, dim=0).indices.tolist()  # 正确处理维度
    predicted_words = [tokenizer.decode([token]) for token in top_1_token]
    all_predictions.extend(predicted_words)

print(all_predictions)