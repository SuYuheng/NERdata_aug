import codecs
import random
from random import shuffle

from transformers import BertTokenizer, BertForMaskedLM

random.seed(1)
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet
# 加载预训练的BERT模型和tokenizer
model_name = '../pre_model/bert-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def data_collect(url):
    temp_s = []
    with open(url, 'r', encoding='utf8') as reader:
        lines = reader.readlines()
        for index in range(len(lines)):
            line = lines[index].strip()
            if not line:
                insert_entity(temp_s)
                words.append(temp_s)
                temp_s = ["sep"]
                continue
            parts = line.split('\t')
            b_type = parts[1].split('-')
            # print(b_type)
            if b_type[0] == "B":
                insert_entity(temp_s)
                words.append(temp_s)
                temp_s = [b_type[1], parts[0]]
            elif b_type[0] == "I":
                temp_s.append(parts[0])
            elif b_type[0] == "O":
                insert_entity(temp_s)
                words.append(temp_s)
                temp_s = [b_type[0], parts[0]]
    del words[0]
    # print(words)


def insert_entity(entity):
    if not entity:
        return
    if entity[0] == "Organization":
        en_org.append(entity)
    elif entity[0] == "Malware":
        en_mal.append(entity)
    elif entity[0] == "Vulnerability":
        en_vul.append(entity)
    elif entity[0] == "Indicator":
        en_ind.append(entity)
    elif entity[0] == "System":
        en_sys.append(entity)

def get_synonyms(word):
    synonyms = []
    list1 = []
    if word[0] == "Organization":
        list1 = en_org
    elif word[0] == "Malware":
        list1 = en_mal
    elif word[0] == "Vulnerability":
        list1 = en_vul
    elif word[0] == "Indicator":
        list1 = en_ind
    elif word[0] == "System":
        list1 = en_sys
    for i in list1:
        if i not in synonyms:
            synonyms.append(i)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


# if __name__ == '__main__':
#     en_org, en_mal, en_ind, en_vul, en_sys = [], [], [], [], []
#     data_collect("../data/Merge/train.txt")
#     print(en_mal)
#     print(en_org)
#     print(en_vul)
#     print(en_ind)
#     print(en_sys)


# if __name__ == '__main__':
#     words, new_words, new_labels = [], [], []
#     en_org, en_mal, en_ind, en_vul, en_sys = [], [], [], [], []
#     url = "../data/Merge/test.txt"
#     data_collect(url)
#     temp_list = get_synonyms(['Malware', 'Windows'])
#     with open("./entity/Malware.txt", 'a', encoding='utf-8') as f:
#         for entity in temp_list:
#             if len(entity) == 2:
#                 f.writelines(entity[1] + '\n')
#             else:
#                 del entity[0]
#                 res = " ".join(entity)
#                 f.writelines(res+'\n')

# if __name__ == '__main__':
#     print([tokenizer.convert_tokens_to_ids(x) for x in tokenizer.tokenize("rocke")])

if __name__ == '__main__':
    x = set()
    with open("./entity/Malware.txt", 'r', encoding='utf-8') as f:
        entities = f.readlines()
        for entity in entities:
            x.update([tokenizer.convert_tokens_to_ids(x) for x in tokenizer.tokenize(entity)])
    with open("./entity/Organization.txt", 'r', encoding='utf-8') as f:
        entities = f.readlines()
        for entity in entities:
            x.update([tokenizer.convert_tokens_to_ids(x) for x in tokenizer.tokenize(entity)])
    with open("./entity/System.txt", 'r', encoding='utf-8') as f:
        entities = f.readlines()
        for entity in entities:
            x.update([tokenizer.convert_tokens_to_ids(x) for x in tokenizer.tokenize(entity)])
    with open("./entity/Vulnerability.txt", 'r', encoding='utf-8') as f:
        entities = f.readlines()
        for entity in entities:
            x.update([tokenizer.convert_tokens_to_ids(x) for x in tokenizer.tokenize(entity)])
    print(x)
    print(len(x))
    with open("./entity/stop_id.txt", 'a', encoding='utf-8') as f:
        for item in x:
            f.writelines(item.__str__()+'\n')
    # print(en_org)
    # print(en_vul)
    # print(en_ind)
    # print(en_sys)
    # aft_words = synonym_replacement(0.7)  # 设置二项分布的p值，实现非真实二项分布，由时间种子生成的随机数的大小进行概率控制
    # for item in aft_words:
    #     if item[0] == "O":
    #         new_words.append(item[1])
    #         new_labels.append(item[0])
    #     elif item[0] == "sep":
    #         new_words.append(item[0])
    #         new_labels.append(item[0])
    #     else:
    #         new_labels.append("B-"+item[0])
    #         new_words.append(item[1])
    #         for t in range(2, len(item)):
    #             new_labels.append("I-"+item[0])
    #             new_words.append(item[t])
    # # print(new_words)
    # # print(new_labels)
    # # print(len(new_words))
    # # print(len(new_labels))
    # with open("../data/CynerN2/train.txt", 'a', encoding='utf-8') as f:
    #     for i in range(len(new_words)):
    #         if new_words[i] == "sep":
    #             f.writelines('\n')
    #         else:
    #             f.writelines(new_words[i]+'\t'+new_labels[i]+'\n')
