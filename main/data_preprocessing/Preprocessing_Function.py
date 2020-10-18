import string
#import nltk
from nltk import word_tokenize
#nltk.download()
#from nltk.corpus import stopwords 
#from nltk.tokenize import TweetTokenizer


f_stopword = open("D:/Chi/Homework/data/stopword.txt")
stopword = f_stopword.read().splitlines()

def RemoveStopwords(text, stopword):
    return ' '.join([word for word in word_tokenize(text) if word not in stopword])

def RemovePunctuation(text, punctuation):
    return ' '.join([word for word in word_tokenize(text) if word not in punctuation])

def RemoveNum(text):
    return ''.join([i for i in text if not i.isdigit()])

def DataAfterPreprocessing(data):
    # 1. Unified Uppercase to Lowercase (將文檔的大寫統一變小寫)
    data['text_lower'] = data['text'].apply(lambda x: x.lower())
    # 2. Remove stopwords
    data['text_no_sw'] = data['text_lower'].apply(lambda x: RemoveStopwords(x, stopword))
    # 3. Remove Number
    data['text_no_num'] = data['text_no_sw'].apply(lambda x: RemoveNum(x))    
    # 4. Remove Punctuation (移除標點符號)
    punctuation = set(string.punctuation)
    data['text_no_pun'] = data['text_no_num'].apply(lambda x: RemovePunctuation(x, punctuation))
    data['text_no_pun'] = data['text_no_pun'].apply(lambda x: x.replace("***",'')) 
    data['text_no_pun'] = data['text_no_pun'].apply(lambda x: x.replace("``",''))
    data['text_no_pun'] = data['text_no_pun'].apply(lambda x: x.replace("--",''))
    data['text_no_pun'] = data['text_no_pun'].apply(lambda x: x.replace("//",''))

    return data