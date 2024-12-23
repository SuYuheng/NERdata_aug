import codecs
import random
from random import shuffle
import random

from transformers import BertForMaskedLM, BertTokenizer
import torch

# random.seed(1)
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet

# 加载预训练的BERT模型和tokenizer
model_name = '../pre_model/bert-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)


def predict_mask(text):
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
            predictions = outputs[0][0, index].topk(1)  # 获取预测的前5个可能的token及其对应的概率
            predicted_token.extend(tokenizer.convert_ids_to_tokens(predictions.indices.tolist()))
    return predicted_token


def data_collect_o(url):
    temp_s = []
    label_s = []
    with open(url, 'r', encoding='utf8') as reader:
        for index, line in enumerate(reader):
            line = line.strip()
            if line:
                parts = line.split('\t')
                # print(parts)
                org_words.append(parts[0])  # 原文内容
                label_s.append(parts[1])  # 标签内容
                temp_s.append(parts[0])  # 获取分句内容
            if not line:
                sentences.append(temp_s)  # 将分句加入句子列表
                sentences.append("sep")
                # print(temp_s)
                temp_s = []
                labels.append(label_s)
                labels.append("sep")
                label_s = []
            if parts[1] == "O":
                re_words.append(parts[0])  # 提取非实体部分
        if temp_s:
            sentences.append(temp_s)  # 将最后一句加入句子列表
            labels.append(label_s)
    # print(sentences,re_words)


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
    re_words, org_words, sentences, labels, new_words, new_labels = [], [], [], [], [], []
    url = "../data/Merge/train.txt"
    data_collect_o(url)
    # new_word = synonym_replacement(sentences[3], 0.5)
    with open("../data/pre_all_base/0.4/train.txt", 'a', encoding='utf-8') as f:
        for i in range(len(sentences)):
            if sentences[i] == "sep":
                f.writelines('\n')
                continue
            # print(len(sentences[i]))
            # print(len(labels[i]))
            new_word, new_label = synonym_replacement(sentences[i], 0.4, labels[i])  # 设置增广的比例
            # new_words.extend(new_word)
            # new_labels.extend(new_label)
            # print(len(new_words), len(new_labels), len(org_words))
            for p in range(len(new_word)):
                f.writelines(new_word[p] + '\t' + new_label[p] + '\n')

# if __name__ == '__main__':
#     print(predict_mask('She ordered a [MASK] coffee and a [MASK] sandwich, while her friend chose a [MASK] tea and a [MASK] salad.'))