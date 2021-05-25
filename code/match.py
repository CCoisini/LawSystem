# -*- coding: UTF-8 -*-
from gensim.similarities import WmdSimilarity
import jieba
from wmd_process import corpus
from wmd_process import corpus_LD
from wmd_process import documents
from wmd_process import documents_LD
from gensim.models import Word2Vec
 
#停用词载入
stopwords = []
# stopword = open('data/qs_stopwords.txt','r',encoding='utf-8')
# for line in stopword:
#     stopwords.append(line.strip())
 
# 加载模型
model = Word2Vec.load("code/model/word2vec_news.model")
 
# 初始化WmdSimilarity
num_best = 5
instance = WmdSimilarity(corpus, model, num_best=5)
instance_LD = WmdSimilarity(corpus_LD, model, num_best=5)
 
# print(20 * '*', '测试', 40 * '*')

def law(data,class_name):
    # print(data,class_name)
    # sent = input('输入查询语句： ')
    sent_w = list(jieba.cut(data))
    query = [w for w in sent_w if not w in stopwords]
    result = []

    #在相似性类中的“查找”query
    if(class_name == '工伤事故'):
        sims = instance[query]
        for i in range(num_best):
            result.append(documents[sims[i][0]])
    else:
        sims = instance_LD[query]
        for i in range(num_best):
            result.append(documents_LD[sims[i][0]])
    return result

    # for i in range(num_best):
    #     print('sim = %.4f' % sims[i][1],documents_LD[sims[i][0]])
    #     print()


