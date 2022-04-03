# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:14:25 2022

@author: LinMouwei
"""
import argparse
import warnings
import time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.stdout = open(BASE_DIR+'/result/record.txt', mode = 'w',encoding='utf-8')
sys.path.append(BASE_DIR)
#from sklearn.impute import SimpleImputer
# from train import train
from log.log_settings import Logger
import pickle
from data_cleaning import clean_missing_data,to_type
from feature_selection import trans_woe
import json


# def predict_data_cleaning(test_data):
#     '''
#     预测用数据的数据清洗，不处理异常值，只处理空缺值，用平均数填补

#     Parameters
#     ----------
#     test_data : 预测数据

#     Returns
#     -------
#     数据清洗后的预测数据

#     '''
#     for name in list(test_data.columns):
#         test_data[name] = SimpleImputer(missing_values=np.nan,strategy='mean').fit_transform(test_data[name])
#     return test_data
# np.reshape(a, newshape)

def predict(filename,modelname):
    '''
    预测数据,统一调用
    filename : 预测用的所有数据文件名
    modelname ：使用模型的名称，模型储存在./model目录下
    --------------
    返回预测结果的predict_result.csv文件
    '''
    # 创建日志文件
    log_file_dir = BASE_DIR + '/log/predict_log.log'
    log = Logger(log_file_dir,level = 'info')        
    log.logger.info('Program Start...')    
    log.logger.info('Try to read date, at '+ time.ctime(time.time()))
    try:
        test_data = pd.read_csv(BASE_DIR+'/data/test_data/'+filename,encoding='utf-8',index_col=0)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1) 
    log.logger.info('Data loaded.')
    try:
        # test_data = predict_data_cleaning(test_data)
        test_data = clean_missing_data(test_data,rate=0)
        test_data = to_type(test_data)
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)    
    log.logger.info('Data cleaned.')
    log.logger.info('Try to load model...')
    try:
        pickle_in = open(BASE_DIR+'/model/'+modelname,'rb')
        clf = pickle.load(pickle_in)
        pickle_in.close()
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        log.logger.info('Model not found.')
        sys.exit(1)  
    log.logger.info('Model loaded.')
    
    testcolumns = [i.strip('\n') for i in open(BASE_DIR+'/model/columns', 'r',encoding = 'utf-8') if i]
    
    try:
        # print(testcolumns)
        temp = pd.DataFrame()
        for name in testcolumns:
            if name in list(test_data.columns):
                temp[name] = test_data[name]
                #temp = pd.concat([temp,test_data.loc[:,name]],axis=1)
            else:
                print('Table structure has changed, please re-train your model!')
                sys.exit(1)
        test_data = temp
        with open(BASE_DIR+'/model/woe_cut','r+') as f:
            js = f.read()
            dic = json.loads(js)
            for name in list(test_data.columns):
            test_data = trans_woe(test_data,name,dic[name+'woe'],dic[name+'cut'])        


    
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        log.logger.info('columns not found.')
        sys.exit(1) 
            
    log.logger.info('Try to predict...')
    try:
        result = clf.predict_proba(test_data)
        test_data['predict_label0_rate'] = result[:,0]
        test_data['predict_label1_rate'] = result[:,1]
        test_data.to_csv(BASE_DIR+'/result/predict_result.csv',encoding = 'utf-8')
    except Exception as e:
        print('行号', e.__traceback__.tb_lineno)
        log.logger.error(e)
        sys.exit(1)    
    log.logger.info('End successfully.')
    return result

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
    path,name = find_files(BASE_DIR+'/data/test_data/')
    if len(name) == 0 or len(name)>1:
        print('Too many files in test_data fold!')
        sys.exit(1)
    filename = name[0]
    
    
    parser = argparse.ArgumentParser('predict model.')
    # parser.add_argument('--path', type=str,default=os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]+'/docs_dics/resumes')
    parser.add_argument('--model_name', type=str,default='pickle_model.dat')
    args = parser.parse_args()
    
    result = predict(filename,modelname=args.model_name)
    
    
    pass