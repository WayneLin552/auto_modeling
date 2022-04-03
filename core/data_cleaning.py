# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:09:54 2022

@author: LinMouwei
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
# from scipy import stats
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
# from log.log_settings import Logger
import warnings
warnings.filterwarnings('ignore')
# from sklearn.cluster import DBSCAN

def to_type(df):
    """
    更改数据类型
    """
    drop = []
    length = len(df.dtypes)
    for i in range(length):
        if df.dtypes[i] == 'object':          
            #df = pd.concat([df, pd.get_dummies(df[df.columns[i]])], axis = 1)
            df = df.join(pd.get_dummies(df[df.columns[i]]))
            drop.append(df.columns[i])
    
    df = df.drop(drop,axis=1)
    return df

def clean_missing_data(df,rate=0.03):
    '''
    接着处理缺失数据，默认缺失3%以内的数据直接删除，3%以外的填补
    ---------
    df ：输入原始数据
    rate : 缺失值比值，默认0.03，超过0.03不低于0.80不删除，否则删除
    输出填补或删除缺失值后的数据
    '''
    deltmp = []
    for i in range(len(df.columns)):
        threshold = len(df)*rate
        if df[df.columns[i]].isnull().sum() == 0:
            continue
        elif df[df.columns[i]].isnull().sum()<threshold:
            df = df[df[df.columns[i]].notnull()]
        elif df[df.columns[i]].isnull().sum()>len(df)*0.80:
            #缺失超过90%，直接删除整个列
            deltmp.append(df.columns[i])
        else:
            X = df.drop(df.columns[i],axis=1)
            # print(len(X.columns))
            Y = df.loc[:,df.columns[i]]
            # print(len(Y.columns))
            #X_0 = SimpleImputer(missing_values=np.nan,strategy='constant').fit_transform(X)
            X_0 = Imputer(missing_values='NaN',strategy='most_frequent').fit_transform(X)
            y_train = Y[Y.notnull()]
            y_test = Y[Y.isnull()]
            x_train = X_0[y_train.index-1,:]
            x_test = X_0[y_test.index-1,:]
            rfc = RandomForestRegressor(random_state=0,n_estimators=100,max_depth=3,n_jobs=-1)
            rfc = rfc.fit(x_train,y_train)
            y_predict = rfc.predict(x_test)
            df.loc[Y.isnull(),df.columns[i]] = y_predict
    df = df.drop(deltmp,axis=1)
    return df


    # log_file_dir = dirname + '/log/resume_extractor.log'
    # log = Logger(log_file_dir,level = 'info')        
    # log.logger.info('Program Start...')    
    # log.logger.info('Try to update resumes date, at '+ time.ctime(time.time()))


def clean_error_data(df,rate=0.01):
    '''
    清洗异常值，用IQR方法，并储存每一列数据的图片到./result路径下
    Parameters
    ----------
    df : 输入原始数据.
    rate : 异常值比值，默认0.01，超过0.01不删除，否则删除
    Returns
    -------
    清洗掉异常值后的原始数据.

    '''
    length = len(df.columns)
    threshold = rate*len(df)
    for i in range(length):
        if len(set(df[df.columns[i]])) <= 2 and df[df.columns[i]].isnull().sum()==0:
            f,ax = plt.subplots(figsize=(10,5))
            fig = sns.countplot(df[df.columns[i]],ax=ax)
            # fig.get_figure()
            plt.savefig(BASE_DIR+'/result/'+str(df.columns[i])+'.png')
            
            # print(df.columns[i])
            continue #判断这一列是否为label或one-hot列，label一般没有null
        else:
            # clf = DBSCAN(eps=0.3, min_samples=10).fit(df[df.columns[i]])
            f,[ax1,ax2] = plt.subplots(1,2,figsize=(12,5))
            fig = sns.distplot(df[df.columns[i]],ax=ax1)
            fig = sns.boxplot(df[df.columns[i]],ax=ax2)
            # fig.get_figure()
            plt.savefig(BASE_DIR+'/result/'+str(df.columns[i])+'.png')
            q1 = df[df.columns[i]].quantile(0.25)
            q3 = df[df.columns[i]].quantile(0.75)
            iqr = q3-q1
            fence_low  = q1-1.5*iqr #极端异常点
            fence_high = q3+1.5*iqr
            if ((df[df.columns[i]] < fence_low) | (df[df.columns[i]] > fence_high)).sum() < threshold:  
                df = df[(df[df.columns[i]] >= fence_low) & (df[df.columns[i]] <= fence_high)]
           # df = df[(df[df.columns[i]] <= fence_high)]
    return df

if __name__=='__main__':
    train_data = pd.read_csv(BASE_DIR+'/data/cs-training.csv',encoding='utf-8',index_col=0)
    test_data = pd.read_csv(BASE_DIR+'/data/cs-test.csv',encoding='utf-8',index_col=0)
    print(train_data.shape)
    print(train_data.info())
    print('-----------------------')
    print(test_data.shape)
    print(test_data.info())
    print('-----------------------')
    train_data = to_type(train_data)
    train_data = clean_missing_data(train_data)
    train_data = clean_error_data(train_data)
    print(train_data.shape)
    print(train_data.info())
    print('-----------------------')
    pass