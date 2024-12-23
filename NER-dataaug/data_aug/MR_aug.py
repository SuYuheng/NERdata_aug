import codecs
import random
from random import shuffle

random.seed(1)
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet


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


def synonym_replacement(p):
    temp_words = words.copy()
    for n in range(len(temp_words)):
        random.seed()
        if temp_words[n][0] != "O" and temp_words[n][0] != "sep" and random.random() < p:
            synonyms = get_synonyms(temp_words[n])
            # print(temp_words[n])
            # print(synonyms)
            synonym = random.choice(synonyms)
            temp_words[n] = synonym
    return temp_words


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
#     data_collect("../data/Cyner/train-temp")
#     print(en_mal)
#     print(en_org)
#     print(en_vul)
#     print(en_ind)
#     print(en_sys)


if __name__ == '__main__':
    words, new_words, new_labels = [], [], []
    en_org, en_mal, en_ind, en_vul, en_sys = [], [], [], [], []
    url = "../data/pre_all/train.txt"
    data_collect(url)
    aft_words = synonym_replacement(0.7)  # 设置二项分布的p值，实现非真实二项分布，由时间种子生成的随机数的大小进行概率控制
    for item in aft_words:
        if item[0] == "O":
            new_words.append(item[1])
            new_labels.append(item[0])
        elif item[0] == "sep":
            new_words.append(item[0])
            new_labels.append(item[0])
        else:
            new_labels.append("B-"+item[0])
            new_words.append(item[1])
            for t in range(2, len(item)):
                new_labels.append("I-"+item[0])
                new_words.append(item[t])
    # print(new_words)
    # print(new_labels)
    # print(len(new_words))
    # print(len(new_labels))
    with open("../data/new_new/train.txt", 'a', encoding='utf-8') as f:
        for i in range(len(new_words)):
            if new_words[i] == "sep":
                f.writelines('\n')
            else:
                f.writelines(new_words[i]+'\t'+new_labels[i]+'\n')
