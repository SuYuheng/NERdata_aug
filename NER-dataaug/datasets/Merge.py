import os

import random

if __name__ == '__main__':

    txt_files = []

    files = os.listdir('./suguguai')

    for file in files:
        txt_files.append(file)
    print(txt_files)

    random.shuffle(txt_files)

    for txt_file in txt_files:
        with open('./suguguai/'+txt_file, 'r', encoding='utf8') as reader:
            lines = reader.readlines()
            with open('dataset_1/train.txt', 'a', encoding='utf-8') as writer:
                writer.writelines(lines)