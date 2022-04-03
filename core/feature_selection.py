# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:12:13 2022

@author: LinMouwei
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from math import isnan
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
#from log.log_settings import Logger
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
#from data_cleaning import to_type,clean_missing_data,clean_error_data
import json

def get_x_y_train_data(df,label_name):
    '''
    从原始数据中分隔出x_train和y_train
    --------
    df : 原始数据
    label_name : y_train所在列的名称【一定不能输错，区分大小写!】
    --------
    输出x_train,y_train 格式为DataFrame
    '''
    if len(df.columns)==0:
        return 'error'
    for name in list(df.columns):
        if str(label_name).replace(' ', '') == str(name).replace(' ',''):
            y_train = df[name]
            x_train = df.drop([name],axis=1)
            with open(BASE_DIR+'/model/label','w+',encoding='utf-8') as f:
                f.write(label_name+'\n')
            return x_train,y_train
    print('Column name not found, please type a correct column name, letter case sensitive!')
    return 'error'


def var_filtering(x_train):
    '''
    方差过滤，去除掉方差为0的列
    Parameters
    ----------
    x_train : 数据清洗后的训练用自变量数据

    Returns
    -------
    方差过滤后的训练用自变量数据

    '''
    if len(x_train.columns)==0:
        print('error')
        return 'error'
    for name in list(x_train.columns):
        if np.std(x_train[name])==0:
            print("%-*s %f" % (40,name+' varience:', 0.0))
            # print(name, 'varience: 0.0')
            x_train = x_train.drop([name],axis = 1)
        else:
            print("%-*s %f" % (40,name+' varience:',np.std(x_train[name])))
            # print(name,'varience:', np.std(x_train[name]))
    return x_train



def spearman_filtering(x_train,y_train):
    '''
    对象为连续变量
    spearman过滤，去掉p值大于0.05的特征
    Parameters
    ----------
    x_train : 训练用的自变量数据
    y_train : 训练用的因变量(labels)数据(0,1数据)
    Returns
    -------
    spearman过滤后的自变量数据

    '''
    if len(x_train.columns)==0:
        print('error')
        return 'error'
    for name in list(x_train.columns):
        if len(set(x_train[name])) <= 2:#说明是离散变量，one-hot转化后变为了0-1向量
            print("%-*s %s" % (40,name,'is discrete variable'))
            continue
        if stats.spearmanr(x_train[name],y_train)[1] > 0.05:
            print("%-*s %f %s" % (40,name+' spearmanr cor:',stats.spearmanr(x_train[name],y_train)[1],'drop!'))
            # print(name, 'spearmanr cor:',stats.spearmanr(x_train[name],y_train)[1],'drop!')
            x_train = x_train.drop([name],axis = 1)
        # elif stats.spearmanr(x_train[name],y_train)[1] == 0: #完全等于零，几乎不可能，或许是离散变量，转为离散变量去求
        #     print("%-*s %f" % (70,name+' may be a discrete variable. Try to use chi2:',stats.chi2_contingency(pd.crosstab(x_train[name],y_train))[1]))
        #     if stats.chi2_contingency(pd.crosstab(x_train[name],y_train))[1] > 0.05:
        #         x_train = x_train.drop([name],axis = 1)
        #         print('p > 0.05, drop!')
        #     else: print('p still < 0.05, keep!')
        else: 
            print("%-*s %f %s" % (40,name+' spearmanr cor:',stats.spearmanr(x_train[name],y_train)[1],'keep!'))
            # print(name, 'spearmanr cor:',stats.spearmanr(x_train[name],y_train)[1],'keep!')
    return x_train

def chi2_filtering(x_train,y_train):
    '''
    对象为离散变量
    卡方过滤，去掉p值大于0.05的特征
    Parameters
    ----------
    x_train : 训练用的自变量数据
    y_train : 训练用的因变量(labels)数据(0,1数据)
    Returns
    -------
    卡方过滤后的自变量数据

    '''   
    if len(x_train.columns)==0:
        print('error')
        return 'error'
    for name in list(x_train.columns):
        if len(set(x_train[name])) > 2:#说明是离散变量，one-hot转化后变为了0-1向量
            print("%-*s %s" % (40,name,'is continuous variable'))
            # print(name,'is a continuous variable.')
            continue
        if stats.chi2_contingency(pd.crosstab(x_train[name],y_train))[1] > 0.05:
            print("%-*s %f%s" % (40,name+' chi2 var:',stats.chi2_contingency(pd.crosstab(x_train[name],y_train))[1],'drop!'))
            #print(name,'chi2 var:',stats.chi2_contingency(np.array([[x_train[name]],[y_train]]))[1],'drop!')
            x_train = x_train.drop([name],axis = 1)
        else:
            print("%-*s %f%s" % (40,name+' chi2 var:',stats.chi2_contingency(pd.crosstab(x_train[name],y_train))[1],'keep!'))
            # print(name,'chi2 var:',stats.chi2_contingency(np.array([[x_train[name]],[y_train]]))[1],'keep!')
    return x_train    
    

def randomforest_filtering(x_train,y_train,threshold = 0.01):
    '''
    随机森林过滤，去掉p值大于0.05的特征
    Parameters
    ----------
    x_train : 训练用的自变量数据
    y_train : 训练用的因变量(labels)数据(0,1数据)
    threshold : importances的阈值，默认0.03，超过保留，否则drop
    Returns
    -------
    卡方过滤后的自变量数据

    '''       
    clf = RandomForestRegressor(n_estimators=150,max_depth=20,n_jobs=-1,random_state=1)
    clf.fit(x_train,y_train)
    importances = clf.feature_importances_
    droptmp = []
    print('importances after filtering by randomforest')
    print('--------------------------------------------')
    key = sorted(list(zip(x_train.columns,importances)),key=lambda x:x[1],reverse=True)
    for f in range(len(key)):
        if key[f][1] < threshold:#np.median(importances): 
            droptmp.append(key[f][0])
        print("%2d %-*s %f" % (f+1,30,key[f][0],key[f][1]))
    x_train = x_train.drop(droptmp,axis=1)
    return x_train


def VIF_filtering(x_train):
    '''
    VIF过滤，值大于5去除
    Parameters
    ----------
    x_train : 训练用的自变量数据
    Returns
    -------
    VIF后的自变量数据

    '''    
    x_train['trunc']=1#添加临时截距
    name = x_train.columns
    x = np.matrix(x_train)
    VIF_list = [variance_inflation_factor(x,i) for i in range(x.shape[1])]
    # print(VIF_list)
    VIF = pd.DataFrame({'feature':name,"VIF":VIF_list})
    for i in range(len(VIF_list)):
        if VIF_list[i] > 5:
            x_train = x_train.drop([name[i]],axis=1)
    print('VIF list')
    print('------------------------------')
    print(VIF)
    if 'trunc' in x_train.columns:
        x_train = x_train.drop(['trunc'],axis=1)
    return x_train

def mono_bin(X,Y,n=10):
    '''
    IV过滤，默认去掉IV值小于0.1的特征
    自动分箱算法，不需要手动设置箱子数,默认从10个箱子开始递减
    Parameters
    ----------
    X : 训练用的自变量数据的某一列
    Y : 训练用的因变量(labels)数据(0,1数据)
    n : 默认初始箱子数，默认从10开始递减
    Returns
    -------
    iv : iv值
    d4 ：分箱结果从小到大排列
    cut ：分箱区间
    woe ： WOE值
    '''  
    r=0 #斯皮尔曼初始值
    badnum=Y.sum() #坏样本数
    goodnum=Y.count()-badnum #好样本数
    while np.abs(r) <1:
        d1 =pd.DataFrame({'X':X,'Y':Y,'Bucket':pd.qcut(X,n,duplicates='drop')})#自动分箱代码，x_train分为n箱
        d2 = d1.groupby('Bucket',as_index = True)#分箱结果聚类
        r,p=stats.spearmanr(d2.mean().X,d2.mean().Y)
        n=n-1
    d3=pd.DataFrame(d2.X.min(),columns=['min'])
    d3['min']=d2.min().X#左边界
    d3['max']=d2.max().X#右边界
    d3['bad']=d2.sum().Y#坏样本数量
    d3['total']=d2.count().Y#总样本
    d3['rate']=d2.mean().Y
    print(d3['rate'])
    print('-----------------------')
    d3['woe']=np.log((d3['bad']/badnum)/((d3['total']-d3['bad'])/goodnum))#woe值
    d3['badattr']=d3['bad']/badnum#每个箱占坏样本总数比值
    d3['goodattr']=(d3['total']-d3['bad'])/goodnum
    iv_tmp = list((d3['badattr']-d3['goodattr'])*d3['woe'])#.sum() #计算iv值
    # print(iv_tmp)
    iv = 0
    for i in iv_tmp:
        if abs(i) != float('inf') and isnan(i) == False:
            iv+=i
    # print(iv)
    d4 = (d3.sort_values(by='min')).reset_index(drop=True) #箱体从大到小排序
    print('分箱结果：')
    print(d4)
    print('IV值：',iv)
    woe=list(d4['woe'])#.round(5))
    cut=[]
    cuts=[]
    woes=[]
    
    for i in range(len(d4['woe'])):
        cut.append(float(d4['max'][i]))
    for i in range(len(woe)):
        if abs(woe[i]) != float('inf') and isnan(woe[i]) == False:
            cuts.append(float(cut[i]))
            woes.append(float(woe[i]))
    cuts.insert(0,float('-inf'))
    print('cut区间：',cuts)
    print('woe结果：',woes)

    return d4,iv,cuts,woes


def self_bin(X,Y,cut):
    #手动分箱
    badnum=Y.sum() #坏样本数
    goodnum=Y.count()-badnum #好样本数
    d1=pd.DataFrame({'X':X,'Y':Y,'Bucket':pd.cut(X,cut)})
    d2=d1.groupby('Bucket',as_index=True)
    d3=pd.DataFrame(d2.X.min(),columns=['min'])
    d3['min']=d2.min().X#左边界
    d3['max']=d2.max().X#右边界
    d3['bad']=d2.sum().Y#坏样本数量
    d3['total']=d2.count().Y#总样本
    d3['rate']=d2.mean().Y
    d3['woe']=np.log((d3['bad']/badnum)/((d3['total']-d3['bad'])/goodnum))#woe值
    d3['badattr']=d3['bad']/badnum#每个箱占坏样本总数比值
    d3['goodattr']=(d3['total']-d3['bad'])/goodnum
    iv_tmp = list((d3['badattr']-d3['goodattr'])*d3['woe'])#.sum() #计算iv值
    iv = 0
    for i in iv_tmp:
        if abs(i) != float('inf') and isnan(i) == False:
            iv+=i
    d4 = (d3.sort_values(by='min')).reset_index(drop=True) #箱体从大到小排序
    woe=list(d4['woe'])#.round(5))
    cuts=[]
    cutss=[]
    woes=[]
    
    for i in range(len(d4['woe'])):
        cuts.append(float(d4['max'][i]))
    for i in range(len(woe)):
        if abs(woe[i]) != float('inf') and isnan(woe[i]) == False:
            cutss.append(float(cuts[i]))
            woes.append(float(woe[i]))
    cutss.insert(0, float('-inf'))
    return d4,iv,woes,cutss





def trans_woe(X,name,woe,cut):
    '''
    在建立模型之前，我们需要将筛选后的变量转换为WoE值，便于信用评分
    X : 训练数据
    name ：某一列名字
    woe : 它的woe
    cut ： 它的cut
    '''
    # woe_name=name+'woe'
    for i in range(len(woe)):       # len(woe) 得到woe里 有多少个数值
        if i==0:
            X[name].loc[(X[name]<=cut[i+1])]=woe[i]  #将woe的值按 cut分箱的下节点，顺序赋值给var的woe_name 列 ，分箱的第一段
        elif (i>0) and  (i<=len(woe)-2):
            X[name].loc[((X[name]>cut[i])&(X[name]<=cut[i+1]))]=woe[i] #    中间的分箱区间   ，，数手指头就很清楚了
        else:
            X[name].loc[(X[name]>cut[len(woe)-1])]=woe[len(woe)-1]   # 大于最后一个分箱区间的 上限值，最后一个值是正无穷
    X[name].loc[(X[name]>cut[-1])]=woe[-1]
    return X


def IV_filtering(x_train,y_train,threshold=0.02,cut=5):
    '''
    x_train : 训练用的自变量数据
    y_train : 训练用的因变量(labels)数据(0,1数据)
    threshold : IV的阈值，默认0.02
    cut : 等距分箱的数目，默认5个
    一般IV小于 0.02: unpredictive；0.02 to 0.1: weak；0.1 to 0.3: medium； 0.3 to 0.5: strong
    返回x_train
    保存woe_cut到./model下
    '''
    d4_tmp=[]
    iv_tmp=[]
    cut_tmp=[]
    woe_tmp=[]
    dic={}
    # ninf = float('-inf')#负无穷大
    # pinf = float('inf')#正无穷大
    if len(x_train.columns)==0:
        print('error')
        return 'error'
    for name in list(x_train.columns):
        print('column name:',name)
        d4,iv,cuts,woe=mono_bin(x_train[name],y_train)
        if iv == 0: #尝试手动分箱
            print('try self-bin')
            d4,iv,woe,cuts = self_bin(x_train[name],y_train,cut)
            print('d4:')
            print(d4)
            print('iv:',iv)
            print('woe:',woe)
            print('cut:',cuts)
            d4_tmp.append(d4)
            iv_tmp.append(iv)
            cut_tmp.append(cuts)
            woe_tmp.append(woe)
        else:
            d4_tmp.append(d4)
            iv_tmp.append(iv)
            cut_tmp.append(cuts)
            woe_tmp.append(woe)
        print('\n')
    name = list(x_train.columns)
    for i in range(len(name)):
        if iv_tmp[i] < threshold:# or iv_tmp[i]>1.0:
            x_train = x_train.drop([name[i]],axis=1)
        else:
            x_train = trans_woe(x_train,name[i],woe_tmp[i],cut_tmp[i])
            dic[name[i]+'woe']=woe_tmp[i]
            dic[name[i]+'cut']=cut_tmp[i]

    if len(x_train.columns)==0:
        print('no suitable columns remain, error')
        return 'error'
    with open(BASE_DIR+'/model/columns','w+',encoding='utf-8') as f:
        print('write columns\' names to file')
        for i in list(x_train.columns):
            f.write(i+'\n')
    js = json.dumps(dic)
    with open(BASE_DIR+'/model/woe_cut','w+') as f:
        f.write(js) 
    # x_train.to_csv(BASE_DIR+'/result/x_train.csv',encoding = 'utf-8')
    return x_train
        
    
        


if __name__=='__main__':
    train_data = pd.read_csv(BASE_DIR+'/data/train_data/cs-training.csv',encoding='utf-8',index_col=0)
    test_data = pd.read_csv(BASE_DIR+'/data/test_data/cs-test.csv',encoding='utf-8',index_col=0)
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
    
    x_train,y_train=get_x_y_train_data(train_data,label_name='SeriousDlqin2yrs')
    x_train=var_filtering(x_train)
    x_train=spearman_filtering(x_train,y_train)
    x_train=chi2_filtering(x_train,y_train)
    x_train=randomforest_filtering(x_train,y_train)
    test1 = VIF_filtering(x_train)
    test2 = IV_filtering(test1,y_train)
    pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    