import codecs
import random
from random import shuffle

random.seed(1)
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet


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
    new_sentence = sentence.copy()
    new_label = label.copy()
    random_word_list = list(set([sentence[p] for p in range(len(sentence)) if label[p] == 'O']))
    # print(random_word_list)
    random.shuffle(random_word_list)
    n = max(1, int(per * len(words)))
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        # print(synonyms)
        if len(synonyms) >= 1:
            while True:
                synonym = random.choice(list(synonyms))
                if synonym != "":
                    break
            new_part = synonym.split()
            # print(new_part)
            # print(synonym)
            # print(len(new_part))
            for k in range(len(new_sentence)):
                if new_sentence[k] == random_word:
                    new_sentence[k] = new_part[0]
                    for j in range(1, len(new_part)):
                        new_sentence.insert(k + j, new_part[j])
                        new_label.insert(k + j, 'O')
                    break
            # new_sentence = [synonym if word == random_word else word for word in new_sentence]
            print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break
    # print(sentence)
    # print(new_sentence)
    # print(new_label)
    return new_sentence, new_label


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


# if __name__ == '__main__':
#     print("atomic number ".split())

if __name__ == '__main__':
    re_words, org_words, sentences, labels, new_words, new_labels = [], [], [], [], [], []
    url = "../data/Merge/train.txt"
    data_collect_o(url)
    # new_word = synonym_replacement(sentences[3], 0.5)
    with open("../data/new_1/0.2/train.txt", 'a', encoding='utf-8') as f:
        for i in range(len(sentences)):
            if sentences[i] == "sep":
                f.writelines('\n')
                continue
            # print(len(sentences[i]))
            # print(len(labels[i]))
            new_word, new_label = synonym_replacement(sentences[i], 0.2, labels[i])  # 设置增广的比例
            # new_words.extend(new_word)
            # new_labels.extend(new_label)
            # print(len(new_words), len(new_labels), len(org_words))
            for p in range(len(new_word)):
                f.writelines(new_word[p] + '\t' + new_label[p] + '\n')
