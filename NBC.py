#coding: utf-8
__author__ = 'LiNing'
import os
import time
import jieba
import jieba.analyse
import nltk
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


def TextProcessing(folder_path):

    folder_list = os.listdir(folder_path)
    train_set = []
    test_set = []
    all_words = {}
    class_list = []

    # 类间循环
    for i in range(len(folder_list)):
        new_folder_path = os.path.join(folder_path, folder_list[i])
        files = os.listdir(new_folder_path)
        class_list.append(folder_list[i].decode('utf-8'))

        # 判断指标
        j = 1
        N = 100 # 每类text最多取100个样本
        # 类内循环
        for file in files:
            if j > N:
                break
            fp = open(os.path.join(new_folder_path, file), 'r')
            raw = fp.read()
            # print raw

            # jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False) # 精确模式，返回的结构是一个可迭代的genertor，jieba分词可以改进词典库
            # jieba.disable_parallel() # 关闭并行分词模式

            # for w in word_cut: # for循环得到每一个词语
            #     print w
            word_list = list(word_cut) # genertor转化为list，每个词unicode格式
            # print word_list

            # # jieba关键词提取
            # # tags = jieba.analyse.extract_tags(raw, 10)
            # tags = jieba.analyse.textrank(raw, 10)
            # print tags

            # 划分训练集和测试集
            nFile = min([len(files), N])
            if j > 0.3*nFile:
                train_set.append((word_list, class_list[i]))
                # 针对训练集的每个text分好的词序列进行循环，统计词频放入all_words
                for word in word_list:
                    if all_words.has_key(word):
                        all_words[word] += 1
                    else:
                        all_words[word] = 1
            else:
                test_set.append((word_list, class_list[i]))
            fp.close()
            # print "Folder ", i+1, " -file- ", j
            j += 1

    # 所有出现过的词数目
    # print "allwords: \n", all_words.keys()
    print "all_words length in dict: ", len(all_words.keys())

    # key函数利用词频进行降序排序
    all_words_list = sorted(all_words.items(), key=lambda all_word:all_word[1], reverse=True) # 内建函数sorted参数需为list

    return train_set, test_set, all_words_list


def words_dict_use_stopwords(deleteN, all_words_list): # word_features是选用的word-词典
    # 生成stopwords_list
    stopwords_file = open('stopwords_cn.txt', 'r')
    stopwords_list = []  
    for line in stopwords_file.readlines():
        stopwords_list.append(line.strip().decode('utf-8'))
    stopwords_file.close()
    
    # dict_name = "dict_stopwords_"+str(deleteN)+".txt"
    # dict = open(dict_name, 'w')
    n = 1  
    word_features = []  
    for t in range(deleteN, len(all_words_list), 1):  
        if n > 1000:
            break
        # print all_words_list[t][0]
        if (all_words_list[t][0] not in stopwords_list) and (not all_words_list[t][0].isdigit()): #不在stopwords_list中并且不是数字
            # dict.writelines(str(all_words_list[t][0]))
            # dict.writelines(' ')
            n += 1
            word_features.append(all_words_list[t][0])
    # dict.close()

    return word_features  


def text_features(text, word_features):  
    text_words = set(text) # 找出text中的不同元素(单词)
    features = {}  
    for word in word_features: # 根据word_feature词典生成每个text的feature (True or False: NaiveBayesClassifier的特征)
        features['contains(%s)' % word] = (word in text_words)
    return features



def TextFeatures(train_set, test_set, word_features):
    # 根据每个text分词生成的word_list生成feature
    # print train_set
    # print test_set
    # train_data是tuple list[({}, ),({}, ),({}, )]，每个text的特征dict和类别组成一个tuple
    train_data = [(text_features(d, word_features), c) for (d, c) in train_set] # (d, c)代表(已经分词的文件, 类别)
    test_data = [(text_features(d, word_features), c) for (d, c) in test_set] 
    # print train_data
    # print test_data
    print "train number: ", len(train_data), "\ntest number: ", len(test_data)     
    return train_data, test_data


 
def TextClassifier(train_data, test_data):
    # 朴素贝叶斯分类器
    classifier = nltk.classify.NaiveBayesClassifier.train(train_data)
    test_accuracy = nltk.classify.accuracy(classifier, test_data)
    return test_accuracy

 
if __name__ == '__main__':

    print "start"
    
    ## 文本预处理
    folder_path = 'Databases\SogouC\Sample'        
    train_set, test_set, all_words_list = TextProcessing(folder_path)
    
    ## 文本特征提取和分类
    deleteNs = range(100, 1000, 20)
    test_accuracy_use = []
    for deleteN in deleteNs:
        # use stopwords
        word_features = words_dict_use_stopwords(deleteN, all_words_list)
        train_data, test_data = TextFeatures(train_set, test_set, word_features)
        test_accuracy = TextClassifier(train_data, test_data)
        test_accuracy_use.append(test_accuracy)
    
    # 结果评价
    # 不同deleteNs下的test_accuracy
    plt.figure()
    plt.plot(deleteNs, test_accuracy_use)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.savefig('result.png')

    print "finished"
