import pandas as pd
import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib
from gensim import corpora
from gensim import models
from gensim import similarities

#陈代码-贝叶斯分类器
#需要修改的内容：1）停用词表的文件位置——在40行与80行中均有   2）jiaohu函数中"questions_train.csv"文件所在的位置——jiaohu函数第一行    均根据实际文件保存位置改

def remove_punctuation(line):   #去停顿
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line
def train_bayes(filepath,dataName,className,modelfile): #用于训练模型的文件名、预分类的列名、类别列名、模型文件
    df=pd.read_csv(filepath)
    df["class_id"]=df[className].factorize()[0]
    class_id_match = df[[className, 'class_id']].drop_duplicates().sort_values('class_id').reset_index(drop=True)
    #print(class_id_match)
    classToID = dict(class_id_match.values)
    idToClass = dict(class_id_match[['class_id', className]].values)
    #print(class_id_match)
    
    #去停顿、停用词，并分词
    df['clean_data'] = df[dataName].apply(remove_punctuation) 
    #df.sample(10)
    stopwords = [line.strip() for line in open('data/baidu_stopwords.txt', 'r', encoding='utf-8').readlines()]  
    df['cut_data'] = df['clean_data'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
    #df.head()
    
    #计算TF-IDF值
    tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
    features = tfidf.fit_transform(df.cut_data)
    labels = df.class_id
    #print(features.shape)
    #print('-----------------------------')
    #print(features)
    
    #寻找该类别下关联最大的2个词语
    N = 5
    for c_name, c_id in sorted(classToID.items()):
        features_chi2 = chi2(features, labels == c_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        #print("# '{}':".format(c_name))
        #print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
        #print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
        
    #模型训练 
    X_train, X_test, y_train, y_test = train_test_split(df['cut_data'], df['class_id'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    #模型保存
    joblib.dump(clf, modelfile)
    
    return idToClass,count_vect

    #模型预测函数，返回预测的类别
def myPredict(modelfile,sec,count_vect,idToClass):   #模型文件名、句子、
    model=joblib.load(modelfile)
    stopwords = [line.strip() for line in open('data/baidu_stopwords.txt', 'r', encoding='utf-8').readlines()]
    format_sec=" ".join([w for w in list(jieba.cut(remove_punctuation(sec))) if w not in stopwords])
    pred_class_id=model.predict(count_vect.transform([format_sec]))
    #print(pred_class_id)
    return idToClass[pred_class_id[0]]


#谢代码-相似度匹配
#要修改的地方：1）停用词表位置——第91行    2）第181行一堆数据文件的位置
# 1.1 历史比较文档的分词
def cut_contents_words(q_list):
    cut_words_list = []
    # 读取停用词表
    with open('data/baidu_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords =[word.replace('\n', '') for word in stopwords]
        # print(stopwords[:20])
    # 处理数据库中的问题数据
    contents = [c for c in q_list if c != '\n']    
    # 对每个句子进行处理
    for line in contents:
        # 对句子进行分词
        words_list = [word for word in jieba.cut(line)]
        # 去停用词
        words_list = [word for word in words_list if word not in stopwords]
        cut_words_list.append(words_list)     
    return contents, cut_words_list
def calSimilarity(q_question,cut_words_list, question_test_cut):
    # 2 制作语料库
    # 2.1 获取词袋
    dictionary = corpora.Dictionary(cut_words_list)
    # 2.2 制作语料库
    # 历史文档的二元组向量转换
    corpus = [dictionary.doc2bow(doc) for doc in cut_words_list]
    # 测试文档的二元组向量转换
    question_test_vec = dictionary.doc2bow(question_test_cut)
    # 3 相似度分析
    # 3.1 使用TF-IDF模型对语料库建模
    tfidf = models.TfidfModel(corpus)
    # 获取测试文档中，每个词的TF-IDF值
    tfidf[question_test_vec]
    # 3.2 对每个目标文档，分析测试文档的相似度
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
    sims = index[tfidf[question_test_vec]]
    # 根3.3 据相似度排序
    sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])    
    return sims_sorted
def matchTestQuestion(q_question,contents, cut_words_list):    
    # 存放输入问题与数据库中问题的相似度值的列表
    similarity_list = []
    # 存放相似度值对应的数据库中的问题列表
    simQ_list = []    
    # 测试文档的分词
    question_test = q_question
    question_test_cut = [word for word in jieba.cut(question_test)]
    # print(question_test_cut[:5])   
    # 返回相似度排序
    sims_sorted = calSimilarity(q_question,cut_words_list, question_test_cut)
    # 返回相似度前10高的问题和相似度
    i = 0
    for j, s in enumerate(sims_sorted):
        if i < 10:
            index = s[0]
            sim = s[1]
            similarity_list.append(sim)
            simQ_list.append(contents[index])
            # print("(sim = {0}) {1}".format(sim, contents[index]))
            i += 1
        else:
            break           
    # 创建{相似度： 问题}字典
    sim_q_dict = dict(zip(similarity_list, simQ_list))         
    return sim_q_dict
def read_match(q_question,filePath):
    # 读取问题回答对数据
    data = pd.read_csv(filePath)   
    # 将问题和回答数据转换为列表，并输出内容
    questions = data['questions'].tolist()
    answers = data['answers'].tolist()
    
    # 创建{问题：回答}字典
    q_a_dict = dict(zip(questions, answers))
    
    # 对问题数据进行分词
    contents, cut_words_list = cut_contents_words(questions)
    
    sim_q_dict = matchTestQuestion(q_question,contents, cut_words_list)
    sqa_list=[]
    for key in sim_q_dict.keys():
        sim = float(key)
        sim_q = sim_q_dict[key]
        sim_a = q_a_dict[sim_q]
        r_list=[sim,sim_q,sim_a]
        sqa_list.append(r_list)
    return sqa_list



def jiaohu(sec):
    idToClass,count_vect=train_bayes('data/questions_train.csv','questions','classes',"classify.model")  #第一个的文件路径按照实际情况修改
    # sec= input("请输入问题：")   #此处连接输入语句
    class_name=myPredict('classify.model',sec,count_vect,idToClass)
    # print("您的问题属于：",class_name)   #此处输出不输出都行
    fileName='data/'+class_name+'.csv'     #此处根据实际情况改变位置
    list_=read_match(sec,fileName)
    related_list=[]
    return class_name,list_

def getClass(sec):
    idToClass,count_vect=train_bayes('data/questions_train.csv','questions','classes',"classify.model")  #第一个的文件路径按照实际情况修改
    class_name=myPredict('classify.model',sec,count_vect,idToClass)
    return class_name

    
'''
    for col in list_:
        #print(col[0])
        if col[0]-0.3<0:
            list_.remove(col)
    if len(list_)>0:  #有相关问题
        print("为您找到最相关的问题：")    #输出第一个问题与答案
        #list_[0].remove(list_[0][0])
        #most_relate=list_[0]     #需要的话，这两行保存最相关的问题
        print(list_[0][1])
        print("解答：",list_[0][2])
        
        if len(list_)>1:    #有其他相关问题
            print("-------------------")
            print("为您找到更多相关问题和答案：")
            for i in range(1,len(list_)):
                list_[i].remove(list_[i][0])
                #related_list.append(list_[i])   #如果需要的话，related_list里面保存了所有相关问题
                print(list_[i]) 
                print("-------------------")   
            
        choice=input("请问是否需要为您返回相关法条？y/n")
        if choice=="y":
            print("将为您返回法条")
            #返回法条语句
        else:
            print("谢谢使用！")
    
    else:   #没有相关问题
        print("将为您返回法条")
        #返回法条语句
'''
    

   

# jiaohu()