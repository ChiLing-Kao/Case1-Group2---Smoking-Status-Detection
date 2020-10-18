import warnings
import pandas as pd
import os, sys, csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import LeaveOneOut 
from imblearn.ensemble import EasyEnsembleClassifier 

from imblearn import FunctionSampler
from sklearn.model_selection import KFold
import time
from sklearn.metrics import f1_score



os.chdir("G:/ExperimenT/Project ECHO/")
sys.path.append(os.getcwd() + "/method")
input_path   = './input/tfidf_matrix_label.csv'
output_path  = './output/200201018_DT_cv_' + str(int(time.time())) + '.csv'
output_file = open(output_path, 'w',newline='')
csvHeader  = ['acc', 'fold']
csvCursor  = csv.DictWriter(output_file, fieldnames=csvHeader)
csvCursor  = csv.writer(output_file)
csvCursor.writerow(csvHeader)
warnings.filterwarnings('ignore')

def func(X, y):
   return X, y




if __name__ == '__main__':

    #不插補     
    df = pd.read_csv(input_path)
    #df = shuffle(df)
    
    #random.shuffle(df)
    y = df['outcome']
    le = LabelEncoder()
    y1= le.fit_transform(y)
    
    df = df.drop(columns=['outcome'])
   
    X = pd.DataFrame(df)
    
    
    for k in range(0,10,1):
    
        n = 30
        loo = LeaveOneOut()
        s = 0
        n_fold = 10
        
        #for train_index, test_index in loo.split(X):
        kf = KFold(n_splits=n_fold,shuffle=True)
        for train_index , test_index in kf.split(X):
            
             
            _X_train, _X_test = X.loc[train_index], X.loc[test_index]
            _y_train, _y_test = y.loc[train_index], y1[test_index]
            
            X_train_bootstrap = _X_train    
            t_0 = X_train_bootstrap[_y_train==0].sample(n,replace=True, random_state=1001)
            t_1 = X_train_bootstrap[_y_train==1].sample(n,replace=True, random_state=1001)
            t_2 = X_train_bootstrap[_y_train==2].sample(n,replace=True, random_state=1001)
            t_3 = X_train_bootstrap[_y_train==3].sample(n,replace=True, random_state=1001)
            X_trainConcat = pd.concat([t_0,t_1,t_2,t_3])
            t_0_y_train = pd.Series(np.zeros((n,), dtype=int))
            t_1_y_train = pd.Series(np.ones((n,), dtype=int)*1)
            t_2_y_train = pd.Series(np.ones((n,), dtype=int)*2)
            t_3_y_train = pd.Series(np.ones((n,), dtype=int)*3)
            
            y_trainConcat = pd.concat([t_0_y_train, t_1_y_train, t_2_y_train, t_3_y_train])
        
            sc = StandardScaler()
            sc.fit(X_trainConcat)
            X_train_std = sc.transform(X_trainConcat)
            X_test_std = sc.transform(_X_test)
               
            ### EasyEnsemble
            ### Create an ensemble sets by iteratively applying random under-sampling
            ee = EasyEnsembleClassifier(random_state=1001)
            sampler = FunctionSampler(func=func)
            X_ee, y_ee = sampler.fit_resample(X_train_std, y_trainConcat)
            
            tree = DecisionTreeClassifier(criterion='entropy', 
                                  max_depth=None,
                                  random_state=1)
            
            clf_labels = ['DT']
            all_clf = [tree]
            
            
            for clf , label in zip(all_clf, clf_labels):
                
                y_pred = clf.fit(X_ee,
                             y_ee).predict(X_test_std)
                
               
                
                print ('f1_score=', f1_score(_y_test, y_pred, average='weighted'))
                csvCursor.writerow((f1_score(_y_test, y_pred, average='weighted'), s))
                
                
            s = s + 1
           
    output_file.close()