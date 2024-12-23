from transformers import BertTokenizer, AutoTokenizer
import re

# 载入预训练的BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-cased')

# 为IP地址、URL和HASH值添加特殊标记
special_tokens_dict = {'additional_special_tokens': ['<IP>', '<URL>', '<HASH>', '<DOMAIN>']}
tokenizer.add_special_tokens(special_tokens_dict)

# 定义正则表达式
ip_regex = r'\b\d{1,3}(\.\d{1,3}){3}\b'
url_regex = r'http[s]?://\S+'
hash_regex = r'\b([a-fA-F\d]{32}|[a-fA-F\d]{40}|[a-fA-F\d]{64})\b'


# 自定义分词器
def custom_tokenize(text):
    # # 替换IP地址
    # text = re.sub(ip_regex, '<IP>', text)
    # # 替换URL
    # text = re.sub(url_regex, '<URL>', text)
    # # 替换HASH值
    # text = re.sub(hash_regex, '<HASH>', text)
    return tokenizer.tokenize(text)


# 测试文本
sample_text = "Access attempt from IP 192.168.1.100 was denied. For more details, visit https://example.com/security. Transaction ID: 9e107d9d372bb6826bd81d3542a419d6."
tokenized_text = custom_tokenize(sample_text)
print(tokenized_text)
print(tokenizer.convert_tokens_to_ids(tokenized_text))
print(len(tokenizer))
