import re
import numpy as np
from nltk.corpus import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

corpus_root = "G:/ExperimenT/Project ECHO/input/CasePresentation1/"
file_pattern = r".*\.txt"
ptb = PlaintextCorpusReader(corpus_root, file_pattern)
list_file = ptb.fileids()

j = 0
corpus = []
index = ["" for x in range(len(list_file))]
label = np.zeros((len(list_file), 1))
new_string = []
for fn in list_file:
    index[j] = fn
    
    # get the label
    ground_truth = fn.split('.')[0].split('_')[3]
    label[j] = int(ground_truth)
    j = j + 1
    
    corpus_ = [line.strip() for line in open("G:/ExperimenT/Project ECHO/input/CasePresentation1/" + fn, 'r')]
    
    for i in range(len(corpus_ )):
        corpus_ [i] = corpus_ [i].lower()
        corpus_ [i] = re.sub(r'\W',' ',corpus_ [i])
        corpus_ [i] = re.sub(r'\s+',' ',corpus_ [i])
        corpus_ [i] = re.sub(r'(^\s+|\s+$)','',corpus_ [i]) #移除每行文字前面或後面可能多個的空白
        corpus_ [i] = re.sub(r'\[!@#$%^&*]/g','',corpus_ [i])
        corpus_ [i] = re.sub(r'\d','',corpus_ [i])
        corpus_ [i] = re.sub(r'(^\s+|\s+$)','',corpus_ [i]) #移除每行文字前面或後面可能多個的空白
        corpus_ [i] = re.sub(r'(^| ).( |$)','',corpus_ [i]) 
        
    
    sent_str_ = ""
    for i in corpus_:
        sent_str_ += str(i) + "-"
    sent_str_ = sent_str_[:-1]
    #print (sent_str_)
    
    sent_str_ = word_tokenize(sent_str_)
    sent_str_without_sw = [word for word in sent_str_ if not word in stopwords.words()]

    new_string = " ".join(sent_str_without_sw)
    #print(corpus_without_sw)
    corpus.append(new_string)
    
vectorizer=CountVectorizer()   #詞頻矩陣
transformer=TfidfTransformer() #tf-idf weight
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
word=vectorizer.get_feature_names() #取得bag-of-words 中的所有詞語
weight=tfidf.toarray()    # 抽取 tf-idf weight
for i in range(len(weight)):#印出每類文件的tf-idf 權重
        print (u"-------输出第",i,u"類文件的tf-idf權重------")
        for j in range(len(word)):  
            print (word[j],weight[i][j])  


np.savetxt("G:/ExperimenT/Project ECHO/input/tfidf_index.csv", index, fmt='%s', delimiter=",")    
np.savetxt("G:/ExperimenT/Project ECHO/input/tfidf_matrix.csv", weight, delimiter=',', fmt='%f')
np.savetxt("G:/ExperimenT/Project ECHO/input/tfidf_label.csv", label.astype(int), fmt='%i', delimiter=",")  


