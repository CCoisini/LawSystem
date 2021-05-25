# -*- coding: UTF-8 -*-
from gensim.models import Word2Vec
from wmd_process import corpus
 
print(20 * '*', '训练模型', 40 * '*')
model = Word2Vec(corpus, workers=3, size=100,min_count=1)
model.save("model/word2vec_news.model")
print("训练完成")