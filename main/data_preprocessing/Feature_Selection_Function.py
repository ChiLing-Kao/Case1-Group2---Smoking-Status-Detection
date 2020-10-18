from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# (1) Bag of Words 詞袋模型- Tf_idf (Term Frequency-Inverse Document Frequency)
def TFID(data):
    Tfid = TfidfVectorizer(use_idf= False,sublinear_tf= True,norm= 'l1',
                             ngram_range= (1, 2),min_df= 3,max_features=7000,max_df=0.8)
    Tfid_x_train = Tfid.fit_transform(data['text'])
    df = Tfid_x_train.toarray()
    return df

# (2) N-gram 語言模型
#rgram_range = (1,1) #表示 unigram
#rgram_range = (2,2) #表示 bigram
#rgram_range = (3,3) #表示 thirgram
def NGRAM(data,rgram_range):
    Ngram = CountVectorizer(rgram_range)  
    Ngram_x_train = Ngram.fit_transform(data['text'])                         
    df = Ngram_x_train.toarray()
    return df                  