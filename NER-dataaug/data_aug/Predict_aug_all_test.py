import codecs
import random
import re
from random import shuffle
import random

from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer
import torch

# random.seed(1)
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet

# 加载预训练的BERT模型和tokenizer
model_name = '../pre_model/latesy'
model = BertForMaskedLM.from_pretrained(model_name)
# 读取tokenizer分词模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 为IP地址、URL和HASH值添加特殊标记
special_tokens_dict = {'additional_special_tokens': ['<IP>', '<URL>', '<HASH>', '<DOMAIN>']}
tokenizer.add_special_tokens(special_tokens_dict)

ip_regex = r'\b\d{1,3}(\.\d{1,3}){3}\b'
ip_regex_1 = r'\b\d{1,3}(?:\[.\]\d{1,3}){3}\b'
domain_regex = r'\b[\w-]+(?:\[.\][\w-]+)+\b'
url_regex = r'http[s]?://\S+'
url_regex_1 = r'HXXP[S]?://\S+'
hash_regex = r'\b([a-fA-F\d]{32}|[a-fA-F\d]{40}|[a-fA-F\d]{64})\b'


def custom_tokenize(text):
    # print(text)
    # 替换IP地址
    text = re.sub(ip_regex, '<IP>', text)
    text = re.sub(ip_regex_1, '<IP>', text)
    # 替换URL
    text = re.sub(url_regex, '<URL>', text)
    text = re.sub(url_regex_1, '<URL>', text)
    # 替换域名
    text = re.sub(domain_regex, '<DOMAIN>', text)
    # 替换HASH值
    text = re.sub(hash_regex, '<HASH>', text)
    return tokenizer.tokenize(text)


def predict_mask(text):
    # tokenized_text = custom_tokenize(text)
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
    tokenized_tuple = enumerate(tokenized_text)
    masked_index = []
    for i, x in tokenized_tuple:
        if x == '[MASK]':
            masked_index.append(i)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    # 创建tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model.eval()

    predicted_token = []

    with torch.no_grad():
        for index in masked_index:
            try:
                outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            except:
                print(text)
                return "error"
            predictions = outputs[0][0, index].topk(1)  # 获取预测的前5个可能的token及其对应的概率
            predicted_token.extend(tokenizer.convert_ids_to_tokens(predictions.indices.tolist()))
    return predicted_token


def data_collect_o(url, max_length=64):
    sentences = []  # 存储所有段落的列表
    labels = []  # 存储所有段落标签的列表

    current_paragraph = []  # 当前段落文本
    paragraph_labels = []  # 当前段落的标签
    current_sentence = []  # 当前句子文本
    sentence_labels = []  # 当前句子的标签

    with open(url, 'r', encoding='utf8') as reader:
        for line in reader:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) > 1:
                    word, label = parts[0], parts[1]
                    # 先将单词添加到当前句子
                    current_sentence.append(word)
                    sentence_labels.append(label)
            else:
                # 空行表示句子结束
                if current_sentence:
                    # 判断加入当前句子后是否超过最大长度
                    current_sentence.append("")
                    sentence_labels.append("sep")
                    if len(current_paragraph) + len(current_sentence) > max_length:
                        # 如果超过，保存当前段落并开始新段落
                        sentences.append(current_paragraph)
                        labels.append(paragraph_labels)
                        current_paragraph = []
                        paragraph_labels = []
                    # 添加当前句子到段落
                    current_paragraph.extend(current_sentence)
                    paragraph_labels.extend(sentence_labels)
                    # 清空当前句子以准备下一个
                    current_sentence = []
                    sentence_labels = []

        # 保存最后一个段落，如果有
        if current_paragraph:
            sentences.append(current_paragraph)
            labels.append(paragraph_labels)

    return sentences, labels


def synonym_replacement(words, per, label):
    sentence = words
    random_word_index = []
    new_label = label.copy()
    temp_sentence = sentence.copy()
    for w in (range(len(temp_sentence))):
        if label[w] == "O":
            if random.random() <= per:
                temp_sentence[w] = '[MASK]'
                random_word_index.append(w)
    temp_str = " ".join(temp_sentence)
    # print(temp_sentence)
    if predict_mask(temp_str) != "error":
        re_words = predict_mask(temp_str)
        random_word_index.sort()
        for i in range(len(random_word_index)):
            temp_sentence[random_word_index[i]] = re_words[i]
    # print("replaced", random_word, "with", re_word)
    # print(re_words)
    # print(temp_sentence)
    # new_sentence = [synonym if word == random_word else word for word in new_sentence]

    # print(sentence)
    # print(new_sentence)
    # print(new_label)
    return temp_sentence, new_label


# if __name__ == '__main__':
#     print("atomic number ".split())

if __name__ == '__main__':
    re_words, new_words, new_labels = [], [], []
    url = "../data/Merge/train.txt"
    sentences, labels = data_collect_o(url)
    with open("../data/2/train.txt", 'a', encoding='utf-8') as f:
        for i in range(len(sentences)):
            new_word, new_label = synonym_replacement(sentences[i], 0.35, labels[i])  # 设置增广的比例
            for p in range(len(new_word)):
                if new_label[p] == "sep":
                    f.writelines('\n')
                else:
                    f.writelines(new_word[p] + '\t' + new_label[p] + '\n')

# if __name__ == '__main__':
#     print(predict_mask('She ordered a [MASK] coffee and a [MASK] sandwich, while her friend chose a [MASK] tea and a [MASK] salad.'))
