import os
import numpy as np
import re    
import pandas as pd         

# 1. Preprocessing
from sklearn import model_selection
from Preprocessing_Function import DataAfterPreprocessing
from Feature_Selection_Function import TFID, NGRAM

#2. Model
from sklearn.ensemble import GradientBoostingClassifier

##3. Result
#from sklearn.metrics.classification import accuracy_score

#%% 1. Read the train and test data
# (1) Read train dataset and labels from the file names
train_path = "D:/Chi/Homework/data/train/" 
file_names = os.listdir(train_path) 

text = []
labels = []
for i in range(0,40):
    f = open(train_path + file_names[i], "r")
    words = f.read().splitlines()
    words.remove("") # remove "" (第一行的空格)
    word2text = " ".join(words) 
    # CURRENT SMOKER = 1
    if not re.search('CURRENT SMOKER',file_names[i]) == None:
        labels.append(1)
        text.append(word2text)
    # NON-SMOKER = 0    
    elif not re.search('NON-SMOKER',file_names[i]) == None:
        labels.append(0)
        text.append(word2text)
    # PAST SMOKER = -1    
    elif not re.search('PAST SMOKER',file_names[i]) == None:
        labels.append(-1)
        text.append(word2text)
    #  UNKNOWN = 2
    elif not re.search('UNKNOWN',file_names[i]) == None:
        labels.append(2)
        text.append(word2text)

# (2) Read test dataset 

test_path = "D:/Chi/Homework/data/test/" 
file_names = os.listdir(test_path) 

test_text = []
for i in range(0,40):
    f = open(test_path + file_names[i], "r")
    words = f.read().splitlines()
    words.remove("") # remove "" (第一行的空格)
    word2text = " ".join(words) 
    test_text.append(word2text)


# (3) Data Preprocessing for train and test data
# a. For train data
smoke_dict = {"text": text,"label": labels}
smoke_df = pd.DataFrame(smoke_dict)
smoke_df['text'][0]
smoke_df_df = DataAfterPreprocessing(smoke_df)
smoke_df_df = smoke_df_df[['text_no_pun','label']]
smoke_df_df.columns = ['text', 'label']
# b. For test data
smoke_dict_test = {"text": test_text}
smoke_df_test = pd.DataFrame(smoke_dict_test)
smoke_df_df_test = DataAfterPreprocessing(smoke_df_test)
smoke_df_df_test = smoke_df_df_test[['text_no_pun']]
smoke_df_df_test.columns = ['text']

# (4) Word to Vector
all_data = pd.concat([smoke_df_df,smoke_df_df_test],axis = 0)
all_data_vec = pd.DataFrame(NGRAM(all_data, rgram_range = (2,2)))
#all_data_vec = pd.DataFrame(TFID(all_data))

# (5) Split back to train and test
X_train = pd.DataFrame(all_data_vec.iloc[:40])
Y_train = all_data.iloc[0:40]['label'].astype(int)
X_test = pd.DataFrame(all_data_vec.iloc[40:])

# (6) Input Train and Test Data
train = pd.concat([X_train, Y_train], axis = 1)
test = X_test

#%% 2. Basice Machine Learning Model
GB_model = GradientBoostingClassifier(criterion = 'friedman_mse', learning_rate =  0.1, loss = 'deviance', max_depth = 5, max_features = 'log2')

#%% 3-1 Model1: One vs One
#1. Data
#(1) Let the label turn into 4 categories   
CS_Rdf = train.loc[train['label'] == 1]       #label = 1,  CURRENT SMOKER 
NS_Rdf = train.loc[train['label'] == 0]       #label = 0,  NON-SMOKER 
PS_Rdf = train.loc[train['label'] == -1]      #label = -1, PAST SMOKER 
UN_Rdf = train.loc[train['label'] == 2]       #label = 2,  UNKNOWN 

#(2) Let the label turn into 1 vs 1 data type  
CS_NS_Rdf = pd.concat([CS_Rdf,NS_Rdf],axis=0)
CS_PS_Rdf = pd.concat([CS_Rdf,PS_Rdf],axis=0)
CS_UN_Rdf = pd.concat([CS_Rdf,UN_Rdf],axis=0)
NS_PS_Rdf = pd.concat([NS_Rdf,PS_Rdf],axis=0)
NS_UN_Rdf = pd.concat([NS_Rdf,UN_Rdf],axis=0)
PS_UN_Rdf = pd.concat([PS_Rdf,UN_Rdf],axis=0)

# (3) split into train and test
CS_NS_text, CS_NS_label = CS_NS_Rdf.drop('label', axis=1), CS_NS_Rdf['label']
CS_PS_text, CS_PS_label = CS_PS_Rdf.drop('label', axis=1), CS_PS_Rdf['label']
CS_UN_text, CS_UN_label = CS_UN_Rdf.drop('label', axis=1), CS_UN_Rdf['label']
NS_PS_text, NS_PS_label = NS_PS_Rdf.drop('label', axis=1), NS_PS_Rdf['label']
NS_UN_text, NS_UN_label = NS_UN_Rdf.drop('label', axis=1), NS_UN_Rdf['label']
PS_UN_text, PS_UN_label = PS_UN_Rdf.drop('label', axis=1), PS_UN_Rdf['label']
test_text = test

# 2. Models and Evaluation
# (1) Model
model = GB_model
# (2) Predict 
# (2-1) 1-1 Predict   
# a. Current Smoker & Non Smoker
CS_NS_model = model.fit(CS_NS_text, CS_NS_label)   #1,0
Y_pred_1 = CS_NS_model.predict(test_text)
Y_pred_1_prob = np.round(CS_NS_model.predict_proba(test_text),5)
# b. Current Smoker & Past Smoker
CS_PS_model = model.fit(CS_PS_text, CS_PS_label)   #1,-1
Y_pred_2 = CS_PS_model.predict(test_text)
Y_pred_2_prob = np.round(CS_PS_model.predict_proba(test_text),5)
# c. Current Smoker & Unknown
CS_UN_model = model.fit(CS_UN_text, CS_UN_label)   #1,2
Y_pred_3 = CS_UN_model.predict(test_text)
Y_pred_3_prob = np.round(CS_UN_model.predict_proba(test_text),5)
# d. Non Smoker & Past Smoker
NS_PS_model = model.fit(NS_PS_text, NS_PS_label)   #0,-1
Y_pred_4 = NS_PS_model.predict(test_text)
Y_pred_4_prob = np.round(NS_PS_model.predict_proba(test_text),5)
# e. Non Smoker & Unknown
NS_UN_model = model.fit(NS_UN_text, NS_UN_label)   #0,2
Y_pred_5 = NS_UN_model.predict(test_text)
Y_pred_5_prob = np.round(NS_UN_model.predict_proba(test_text),5)
# f. Past Smoker & Unknown
PS_UN_model = model.fit(PS_UN_text, PS_UN_label)   #-1,2
Y_pred_6 = PS_UN_model.predict(test_text)
Y_pred_6_prob = np.round(PS_UN_model.predict_proba(test_text),5)

# (2-1) Accuracy with Most number of Probability
# a. Probility of Prediction for All Samples
prob = np.zeros((len(test),4))

prob[:,0] = Y_pred_1_prob[:,0] #1
prob[:,1] = Y_pred_1_prob[:,1] #0

prob[:,0] = prob[:,0] + Y_pred_2_prob[:,0] #1
prob[:,2] = prob[:,2] + Y_pred_2_prob[:,1] #-1

prob[:,0] = prob[:,0] + Y_pred_3_prob[:,0] #1
prob[:,3] = prob[:,3] + Y_pred_3_prob[:,1] #2

prob[:,1] = prob[:,1] + Y_pred_4_prob[:,0] #0
prob[:,2] = prob[:,2] + Y_pred_4_prob[:,1] #-1

prob[:,1] = prob[:,1] + Y_pred_5_prob[:,0] #0
prob[:,3] = prob[:,3] + Y_pred_5_prob[:,1] #2

prob[:,2] = prob[:,2] + Y_pred_6_prob[:,0] #-1
prob[:,3] = prob[:,3] + Y_pred_6_prob[:,1] #2

# b. Predict Labels
multi_predict = []
for i in range(len(test)):
    label_len = len(np.where(prob[i,:] == np.max(prob[i,:]))[0])
    if (label_len == 1):
        label = np.where(prob[i,:] == np.max(prob[i,:]))[0][0]
        multi_predict.append(label)
    else:
        num = np.random.randint(0, label_len)
        label = np.where(prob[i,:] == np.max(prob[i,:]))[0][num]
        multi_predict.append(label)           

# c. Turn into right labels
for i in range(len(test)):
    if (multi_predict[i] == 0):
        multi_predict[i] = 1
    elif (multi_predict[i] == 1):
        multi_predict[i] = 0
    elif (multi_predict[i] == 2):
        multi_predict[i] = -1
    else:
        multi_predict[i] = 2

print("Model1 Predict:",multi_predict)   

#%% Model 2 :One vs Rest
#(1) Let the label turn into 1 vs all data type
#(1-1) CURRENT SMOKER v.s Non-CURRENT SMOKER
CS_Rdf = train.loc[train['label'] == 1]     
Non_CS_Rdf = train.loc[train['label'] != 1]   
Non_CS_Rdf['label'] = np.zeros((len(Non_CS_Rdf),),int)

#(1-2) NON-SMOKER v.s Non-NON-SMOKER
NS_Rdf = train.loc[train['label'] == 0]     
NS_Rdf['label'] = NS_Rdf['label'].replace([0], 1)
Non_NS_Rdf = train.loc[train['label'] != 0]   
Non_NS_Rdf['label'] = np.zeros((len(Non_NS_Rdf),),int)

#(1-3) PAST SMOKER v.s Non-PAST SMOKER
PS_Rdf = train.loc[train['label'] == -1]  
PS_Rdf['label'] = PS_Rdf['label'].replace([-1], 1) 
Non_PS_Rdf = train.loc[train['label'] != -1]  
Non_PS_Rdf['label'] = np.zeros((len(Non_PS_Rdf),),int)

#(1-4) UNKNOWN v.s Non- UNKNOWN
UN_Rdf = train.loc[train['label'] == 2] 
UN_Rdf['label'] = UN_Rdf['label'].replace([2], 1)  
Non_UN_Rdf = train.loc[train['label'] != 2] 
Non_UN_Rdf['label'] = np.zeros((len(Non_UN_Rdf),),int)

#(2) Let the label turn into 1 vs 1 data type  
CSN_Rdf = pd.concat([CS_Rdf,Non_CS_Rdf],axis=0)
NSN_Rdf = pd.concat([NS_Rdf,Non_NS_Rdf],axis=0)
PSN_Rdf = pd.concat([PS_Rdf,Non_PS_Rdf],axis=0)
UNN_Rdf = pd.concat([UN_Rdf,Non_UN_Rdf],axis=0)

# (3) split into train and test
CSN_text, CSN_label = CSN_Rdf.drop('label', axis=1), CSN_Rdf['label']
NSN_text, NSN_label = NSN_Rdf.drop('label', axis=1), NSN_Rdf['label']
PSN_text, PSN_label = PSN_Rdf.drop('label', axis=1), PSN_Rdf['label']
UNN_text, UNN_label = UNN_Rdf.drop('label', axis=1), UNN_Rdf['label']
test_text = test
   
# 2. Models and Evaluation
# (1) Model    
model2 = GB_model
# (2) Predict 
# (2-1) 1-1 Predict
# a. CURRENT SMOKER v.s Non-CURRENT SMOKER
CSN_model = model2.fit(CSN_text, CSN_label)   
Y_pred_1_1 = CSN_model.predict(test_text)
Y_pred_1_1_prob = np.round(CSN_model.predict_proba(test_text),5)
Y_pred_1_1_prob
# b. NON-SMOKER v.s Non-NON-SMOKER
NSN_model = model2.fit(NSN_text, NSN_label)   
Y_pred_2_1 = NSN_model.predict(test_text)
Y_pred_2_1_prob = np.round(NSN_model.predict_proba(test_text),5)
Y_pred_2_1_prob
# c. PAST SMOKER v.s Non-PAST SMOKER
PSN_model = model2.fit(PSN_text, PSN_label)   
Y_pred_3_1 = PSN_model.predict(test_text)
Y_pred_3_1_prob = np.round(PSN_model.predict_proba(test_text),5)
Y_pred_3_1_prob
# d. UNKNOWN v.s Non- UNKNOWN
UNN_model = model2.fit(UNN_text, UNN_label)   
Y_pred_4_1 = UNN_model.predict(test_text)
Y_pred_4_1_prob = np.round(UNN_model.predict_proba(test_text),5)
Y_pred_4_1_prob

# (2) Accuracy with Most number of Probability
# a. Probility of Prediction for All Samples
prob = np.zeros((len(test),4))

prob[:,0] = Y_pred_1_1_prob[:,1] 
prob[:,1] = Y_pred_2_1_prob[:,1] 
prob[:,2] = Y_pred_3_1_prob[:,1]
prob[:,3] = Y_pred_4_1_prob[:,1]


# b. Predict Labels
multi_predict_1 = []
for i in range(len(test)):
    label_len = len(np.where(prob[i,:] == np.max(prob[i,:]))[0])
    predict_1 = np.c_[Y_pred_1_1,Y_pred_2_1,Y_pred_3_1,Y_pred_4_1]
    if np.sum(predict_1[i]) == 1:
        label = np.where(predict_1[i] == 1)[0][0]
        multi_predict_1.append(label)
    else:
        label = np.where(prob[i,:] == np.max(prob[i,:]))[0][0]
        multi_predict_1.append(label)

# c. Turn into right labels
for i in range(len(test)):
    if (multi_predict_1[i] == 0):
        multi_predict_1[i] = 1
    elif (multi_predict_1[i] == 1):
        multi_predict_1[i] = 0
    elif (multi_predict_1[i] == 2):
        multi_predict_1[i] = -1
    else:
        multi_predict_1[i] = 2

print("Model2 Predict:", multi_predict_1)   

#%% Model 3 : Multiclass  Classfication
model3 = GB_model
model3.fit(X_train, Y_train)
multi_predict_2 = model3.predict(test_text)
    
print("Model3 Predict:", multi_predict_2)   

#%% Model Ensemble : Voting
Model_list = [multi_predict,multi_predict_1,multi_predict_2]
Voting_result = np.zeros((len(test),4))
Accuracy_voting = []
Voting_result = np.zeros((len(test),4))
for model in range(3):
    for l in range(len(test)):            
        if Model_list[model][l] == 1:
            Voting_result[l][0] += 1
        elif Model_list[model][l] == 0:
            Voting_result[l][1] += 1
        elif Model_list[model][l] == -1:
            Voting_result[l][2] += 1           
        else:
            Voting_result[l][3] += 1         

Voting_predict = []
for i in range(len(test)):
   if (np.max(Voting_result[i,:]) != 1) :
       voting_label = np.where(Voting_result[i,:] == np.max(Voting_result[i,:]))[0][0]
       Voting_predict.append(voting_label)
   else:
       voting_label = Model_list[0][l]
       Voting_predict.append(voting_label)

for j in range(len(test)):
    if (Voting_predict[j] == 0):
        Voting_predict[j] = 1
    elif (Voting_predict[j] == 1):
        Voting_predict[j] = 0
    elif (Voting_predict[j] == 2):
        Voting_predict[j] = -1
    else:
        Voting_predict[j] = 2

print("Ensemble Model Predict:",Voting_predict)   

#%% Ourout the predict result
ROOT_DIR = "D:/Chi/Homework/" 
file = open(ROOT_DIR + 'Case1_{2}.txt',"a")
for j in range(len(test)):
    if (Voting_predict[j] == 1):
        Voting_predict[j] = "CURRENT SMOKER"
    elif (Voting_predict[j] == 0):
        Voting_predict[j] = 'NON-SMOKER'
    elif (Voting_predict[j] == -1):
        Voting_predict[j] = 'PAST SMOKER'
    else:
        Voting_predict[j] = 'UNKNOWN'
    
    file.write("".join(Voting_predict[j]))
    file.write('\n')
file.close()