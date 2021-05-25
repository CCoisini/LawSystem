# -*- coding: UTF-8 -*-
import jieba
import pandas as pd
 
# f = open('data/关键词-工伤保险条例.csv', encoding='utf-8')
path = open('data/关键词-工伤保险条例.csv', encoding = 'utf-8')
path_LD = open('data/关键词-劳动法.csv', encoding = 'utf-8')

data = pd.read_csv(path)
data_LD = pd.read_csv(path_LD)

data = data.replace(r'\s', '', regex=True)
data = data.drop_duplicates() # 按全字段进行数据去重

data_LD = data_LD.replace(r'\s', '', regex=True)
data_LD = data_LD.drop_duplicates() # 按全字段进行数据去重

law_data = data['法条']
law_data = law_data.tolist()

LD_data = data_LD['法条']
LD_data = LD_data.tolist()

corpus = []
corpus_LD = []
documents = []
documents_LD = []
 
#加载停用词
stopwords = []
stopword = open('data/qs_stopwords.txt','r',encoding='utf-8')
for line in stopword:
    stopwords.append(line.strip())
 
#工伤保险条例
for i in range(len(law_data)):
    each = law_data[i].replace('\n', '').replace(' ', '').strip()
    # print(each)
    documents.append(each)
    each = list(jieba.cut(each))
    text = [w for w in each if not w in stopwords]
    corpus.append(text)

#劳动法条例
for i in range(len(LD_data)):
    each = LD_data[i].replace('\n', '').replace(' ', '').strip()
    # print(each)
    documents_LD.append(each)
    each = list(jieba.cut(each))
    text = [w for w in each if not w in stopwords]
    corpus_LD.append(text)


# print('工伤保险条例匹配总数',len(corpus))
# print('劳动法条例匹配总数',len(corpus_LD))

