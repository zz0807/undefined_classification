
'''
对训练集分词并导出分词结果
分词后的文件放在train_word_seg下 文件名与训练集train中的文件名相同
'''

import jieba
import string
import os

#标点符号的映射表
intab = string.punctuation + "。！？，“”﻿"
outtab = " " * len(intab)
trantab = str.maketrans(intab, outtab)

path = "train/"
dirs = os.listdir(path)

print("正在分词")
for x in dirs:
    with open("train/"+x, "r", encoding= 'utf-8') as file:
        new_file = open("train_word_seg/" + os.path.split(file.name)[1], "w" ,encoding= 'utf-8')
        for line in file.readlines():
             temp = line.strip().translate(trantab) #过滤掉标点符号
             l_seg = "/".join(jieba.cut(temp))
             new_file.write(l_seg +'\n')
        new_file.close()



