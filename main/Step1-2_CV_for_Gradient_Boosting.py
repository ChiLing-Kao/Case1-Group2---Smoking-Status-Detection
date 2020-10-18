# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 03:01:27 2020

@author: Indi
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:32:02 2020
# Ensemble for PA


@author: Indi
"""
import warnings
import pandas as pd
import os, sys, csv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.ensemble import EasyEnsembleClassifier 
from imblearn import FunctionSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
import time
import pickle
from sklearn.metrics import accuracy_score, accuracy_score, f1_score
import optuna




os.chdir("G:/ExperimenT/Project ECHO/")
sys.path.append(os.getcwd() + "/method")
input_path   = './input/tfidf_matrix_label.csv'
output_path  = './output/200201018_GradientBoosting_cv_' + str(int(time.time())) + '.csv'
output_file = open(output_path, 'w',newline='')
csvHeader  = ['acc', 'fold', 'best_trail']
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
        #loo = LeaveOneOut()
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
            
            
            gbc = GradientBoostingClassifier()
            pipe5 = Pipeline([['sc', StandardScaler()],
                              ['clf', gbc]])
            
            clf_labels = ['GradientBoosting']
            all_clf = [pipe5]
            clf_optimal= []
            
            for clf , label in zip(all_clf, clf_labels):
                
                def objective(trial):
                    
                    #Uniform Parameter : A uniform distribution in the linear domain.
                    subsample = trial.suggest_discrete_uniform("subsample", 0.1, 1.0, 0.05)
                    clf = GradientBoostingClassifier(subsample=subsample, random_state=0)
                    
                    clf.fit(X_ee, y_ee)
                    
                    # Save a trained model to a file.
                    with open('{}.pickle'.format(str(k) + '_' + str(trial.number)), 'wb') as fout:
                        pickle.dump(clf, fout)
                    #return 1.0 - accuracy_score(y_valid, clf.predict(X_valid))
                    return clf.score(X_test_std, _y_test)
                     
                
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=10)
                
                
                with open('{}.pickle'.format(str(k) + '_' + str(study.best_trial.number)), 'rb') as fin:
                    best_clf = pickle.load(fin)
                    
                y_pred = best_clf.predict(X_test_std)
            s = s + 1
           
    output_file.close()