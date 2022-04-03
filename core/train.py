# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:09:51 2022

@author: LinMouwei
"""
import argparse
import warnings
import time
warnings.filterwarnings('ignore')
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# import seaborn as sns
# from scipy import stats
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout = open(BASE_DIR+'/result/record.txt', mode = 'w+',encoding='utf-8')
sys.path.append(BASE_DIR)
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from log.log_settings import Logger
from data_cleaning import to_type,clean_missing_data,clean_error_data
from feature_selection import get_x_y_train_data,var_filtering,spearman_filtering,chi2_filtering,randomforest_filtering,VIF_filtering,IV_filtering
import pickle


def train_model(x_train,y_train,rate=0.25):
    '''
    训练数据
    rate : 划分测试集的比值，默认0.25，不建议太高
    --------------
    返回训练模型
    评估模型，获得auc，roc并保存到./pics目录下
    保存模型到./model目录下
    '''
    x_train['y_train_concat_temp'] = y_train
    training,testing=train_test_split(x_train,test_size=rate,random_state=1)
    x_train = training.drop(training.columns[-1],axis=1)
    y_train = training.iloc[:,-1]
    x_test = testing.drop(testing.columns[-1],axis=1)
    y_test = testing.iloc[:,-1]
    
    clf = GradientBoostingClassifier()
    
    # clf = LogisticRegression()
    clf.fit(x_train,y_train)
    #对测试集做预测
    score_proba = clf.predict_proba(x_test)
    y_predproba=score_proba[:,1]
    # coe = clf.coef_
    # print(coe)
    print('save model...')
    try:
        pickle.dump(clf, open(BASE_DIR+'/model/pickle_model.dat','wb'))
        print('save model successfully')
    except:
        print('can not save model, error')
    fpr,tpr,threshold = roc_curve(y_test,y_predproba) #计算roc，auc
    auc_score = auc(fpr,tpr)
    plt.figure(figsize=(8,5))  #只能在这里面设置
    fig = plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% auc_score)
    plt.legend(loc='lower right',fontsize=14)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim=([0, 1])
    plt.ylim=([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('TPR',fontsize=16)
    plt.xlabel('FPR',fontsize=16)
    # plt.show()
    plt.savefig(BASE_DIR+'/result/ROC_AUC_score.png')
    print('AUC score:',auc_score)
    
    #KS curve
    fig,ax = plt.subplots()
    pic = ax.plot(1-threshold,tpr,label='tpr')
    pic = ax.plot(1-threshold,fpr,label='fpr')
    pic = ax.plot(1-threshold,tpr-fpr,label='KS')
    pic = plt.xlabel('score')
    pic = plt.title('KS curve')
    pic = ax.legend(loc='upper left')
    plt.savefig(BASE_DIR+'/result/KS_curve.png')
    print('KS score:',max(tpr-fpr))
    return None

def train(filename,label_name,missing_data_rate=0.03,error_data_rate=0.01,randomforest_threshold=0.01,iv_threshold=0.02,iv_cut=5,split_rate=0.25,index_col='Unnamed: 0'):
    '''
    训练数据,统一调用
    filename : 训练用的所有数据文件名
    label_name : 自变量的列名称(0,1 label name)，一定要填对！【区分大小写！】
    missing_data_rate ：缺失数据比值，默认0.03，大于0.03填充，小于直接删除
    error_data_rate : 异常数据比值，默认0.01，小于0.01删除，大于可能是别的情况，保留
    randomforest_threshold ：随机森林选择特征阈值，默认0.01，大于0.01的重要性参数保留
    iv_threshold ：IV选择阈值，默认0.02，大于0.02保留
    iv_bin ：手动分箱数目，默认5
    split_rate ：训练集用于验证的比值，默认0.25，不建议设置太高
    index_col : 作为index_col不参与建模的列名，默认‘Unnamed: 0’，如果数据集没有index，就不考虑。
    --------------
    返回训练模型
    评估模型，获得auc，roc并保存到./result目录下
    保存模型到./model目录下
    保存各个数据预处理信息到./result/record.txt中
    '''
    # 创建日志文件
    log_file_dir = BASE_DIR + '/log/train_log.log'
    log = Logger(log_file_dir,level = 'info')        
    log.logger.info('Program Start...')    
    log.logger.info('Try to read date, at '+ time.ctime(time.time()))
    try:
        train_data = pd.read_csv(BASE_DIR+'/data/train_data/'+filename,encoding='utf-8',index_col=index_col)
        with open(BASE_DIR+'/model/index','w+',encoding='utf-8') as f:
            f.write(index_col)
    except:
        # print('行号', e.__traceback__.tb_lineno)
        # log.logger.error(e)
        log.logger.info('Seems that train_data do not have an index column, please close the program to check again or ignore it!')
        try:
            train_data = pd.read_csv(BASE_DIR+'/data/train_data/'+filename,encoding='utf-8',index_col='Unnamed: 0')
            with open(BASE_DIR+'/model/index','w',encoding='utf-8') as f:
                f.write('Unnamed: 0')
        except:
            train_data = pd.read_csv(BASE_DIR+'/data/train_data/'+filename,encoding='utf-8')
        # sys.exit(1) 
    log.logger.info('Data loaded.')
    print(train_data.shape)
    print(train_data.info())
    print('-----------------------')
    print('\n')
    log.logger.info('Try to clean data...')  
    
    try:
        train_data = clean_missing_data(train_data,rate=missing_data_rate)
        train_data = to_type(train_data)    
        train_data = clean_error_data(train_data,rate=error_data_rate)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)
    log.logger.info('Data cleaned.')  
    print(train_data.shape)
    print(train_data.info())
    print('-----------------------')
    print('\n')
    
    log.logger.info('Try to split data...')  
    try:
        x_train,y_train=get_x_y_train_data(train_data,label_name=label_name)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)
    log.logger.info('Data splited.')  
    print('\n')
    
    try:
        x_train=var_filtering(x_train)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)
    log.logger.info('Data var_filtering.') 
    print('\n')
    try:
        x_train=spearman_filtering(x_train,y_train)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)
    log.logger.info('Data spearman_filtering.') 
    print('\n')
    try:
        x_train=chi2_filtering(x_train,y_train)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)
    log.logger.info('Data chi2_filtering.') 
    print('\n')
    try:
        x_train=randomforest_filtering(x_train,y_train,threshold=randomforest_threshold)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)
    log.logger.info('Data randomforest_filtering.') 
    print('\n')
    try:
        x_train = VIF_filtering(x_train)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)
    log.logger.info('Data VIF_filtering.') 
    print('\n')
    try:
        x_train = IV_filtering(x_train,y_train,threshold=iv_threshold,cut=iv_cut)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)
    log.logger.info('Data IV_filtering.') 
    print('\n')
    try:
        clf = train_model(x_train,y_train,rate=split_rate)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)
    log.logger.info('Data trained.')
    log.logger.info('Result saved to ./result successfully!')
    return clf
    
def find_files(file_dir):
    """迭代查找文件"""
    file_paths = []
    file_names = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            path = os.path.join(root,file)
            rear = os.path.splitext(path)[1]
            if rear in [".csv"]: #本处只考虑csv
                file_paths.append(path)
                file_names.append(file)
    return file_paths, file_names



if __name__=='__main__':
    
    parser = argparse.ArgumentParser('Auto run model.')
    # parser.add_argument('--path', type=str,default=os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]+'/docs_dics/resumes')
    parser.add_argument('--label', type=str,help='你的训练集label列名称，区分大小写和空格，一定要填对！')
    parser.add_argument('--missing_data_rate', type=float,default=0.03,help='缺失率阈值，小于这个值删除。')
    parser.add_argument('--error_data_rate', type=float,default=0.01,help='错误率阈值，小于这个值删除。')
    parser.add_argument('--randomforest_threshold', type=float,default=0.01,help='随机森林筛选特征阈值，小于这个值删除。')
    parser.add_argument('--iv_threshold', type=float,default=0.02,help='IV筛选特征阈值，小于这个值删除。')
    parser.add_argument('--iv_cut', type=int,default=5,help='IV分箱数目，为正整数。')
    parser.add_argument('--split_rate', type=float,default=0.25,help='训练集-测试集分隔比例。')
    parser.add_argument('--index_col', type=str,default='Unnamed: 0',help='作为index_col不参与建模的列名，默认‘Unnamed: 0’，如果数据集没有index，就不考虑。')
    args = parser.parse_args()
    # path = args.path
    # path = os.path.abspath(path)
    missing_data_rate = args.missing_data_rate
    error_data_rate =args.error_data_rate
    randomforest_threshold = args.randomforest_threshold
    iv_threshold = args.iv_threshold
    iv_cut = args.iv_cut
    split_rate = args.split_rate
    index_col = args.index_col
    
    path,name = find_files(BASE_DIR+'/data/train_data/')
    if len(name) == 0 or len(name)>1:
        print('Too many files in train_data fold!')
        sys.exit(1)
    filename = name[0]
    with open(BASE_DIR+'/model/label','r+',encoding='utf-8') as f:
        label_name = f.readline().strip('\n')
    with open(BASE_DIR+'/model/index','r+',encoding='utf-8') as f:
        index_tmp = f.readline().strip('\n')
    if index_tmp and index_tmp != 'Unnamed: 0' and index_col == 'Unnamed: 0':
        index_col = index_tmp
    if args.label:
        clf = train(filename,label_name=args.label,missing_data_rate=missing_data_rate,\
                    error_data_rate=error_data_rate,randomforest_threshold=randomforest_threshold,\
                        iv_threshold=iv_threshold,iv_cut=iv_cut,split_rate=split_rate,index_col=index_col)
    elif label_name == '': 
        print('label name is not defined, please type a label name')
        clf = train(filename,label_name=args.label,missing_data_rate=missing_data_rate,\
                    error_data_rate=error_data_rate,randomforest_threshold=randomforest_threshold,\
                        iv_threshold=iv_threshold,iv_cut=iv_cut,split_rate=split_rate,index_col=index_col)
    else:
        clf = train(filename,label_name=label_name,missing_data_rate=missing_data_rate,\
                    error_data_rate=error_data_rate,randomforest_threshold=randomforest_threshold,\
                        iv_threshold=iv_threshold,iv_cut=iv_cut,split_rate=split_rate,index_col=index_col)
    # pass