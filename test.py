import os
from gensim.models import Word2Vec
import string
import jieba
import numpy as np

model = Word2Vec.load("weibodata_vectorB.gem")

print("模型导入成功")

def to_vec(list):
    result = np.array([])
    for value in list:
        if value in model.wv.vocab:#如果词在词向量里
            temp = model[value]
            result = np.append(result, temp, axis=0)
    return result


#标点符号的映射表
intab = string.punctuation + "。！？，“”﻿"
outtab = " " * len(intab)
trantab = str.maketrans(intab, outtab)

# 打开文件
path = "train/"
dirs = os.listdir( path )

line = "今天天气好晴朗"
l = line.strip().translate(trantab)  # 过滤掉标点符号
l2 = " ".join(jieba.cut(l)).split()
#l2 = to_vec(l2)
print(l2)


