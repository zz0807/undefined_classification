'''
读出train_word_seg下分好词的文件，将句子转换为词向量
'''
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import string
import os

model = Word2Vec.load("weibodata_vectorB.gem")
print('词向量模型导入完成')
def to_vec(list):
    result = []
    count = 0 #记录句子的词向量个数，小于50则补零，大于50则切掉
    for value in list:
        if(count == 50):
            break
        if value in model.wv.vocab:#如果词在词向量里
            temp = model[value] #得到该词的词向量
            temp = temp.tolist()
            result = result + temp
            count += 1
    i = count
    while(i<50):
        result = result + [0.0]*50
        i = i + 1
    return result

path = "train_word_seg/"
dirs = os.listdir(path)

all_sentences = []
for x in dirs:
    with open("train_word_seg/"+x, "r", encoding= 'utf-8') as file:
         for line in file.readlines():
             line_vec = to_vec(line.strip().split('/'))
             all_sentences.append(line_vec)

all_sentences = np.array(all_sentences)
#聚类
keams = KMeans(n_clusters=6).fit(all_sentences)
print(keams.cluster_centers_)


