# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 23:14:44 2017

@author: Administrator
"""

from LSTM import MYLSTM
from nltk import  word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import  RandomForestClassifier
from sklearn import preprocessing
from sklearn.svm import  SVC
import lightgbm as lgb
from sklearn.cross_validation import  cross_val_score
from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report 
from gensim import corpora
from gensim.models import Word2Vec  
import numpy as np
import os
import nltk
import pandas as pd
import re
import pickle
import random

def clean_text(text,symbol_remove,rex_remove_contain_number):
    rex_remove_number=re.compile(r'[’]')
    text=rex_remove_number.sub('\'',text)
    data_text=word_tokenize(text)
#    removing number
    data_text=[ rex_remove_contain_number.sub('',d.lower()) for d in data_text if d not in symbol_remove ]
#    cleaning data        
#    reovmed_stop_word=[d for  d  in  data_text if d not in  stopwords.words('english') and d != '']
    reovmed_stop_word=[d for  d  in  data_text if d != '']
    return reovmed_stop_word
def clean_paper(paper,symbol_remove,rex_remove_contain_number):
    return rex_remove_contain_number.sub('',paper.lower())
def extract_feature(data,symbol_remove,rex_remove_contain_number):
#    data=clean_paper(data)
     
    data_text=clean_text(data,symbol_remove,rex_remove_contain_number)
    data_df=pd.DataFrame(data_text,columns=['word'])
    data_df['fre']=data_df['word']
    return  data_df.groupby('word').count().to_dict('dict')
def extract_train_feature(data,unique_aword_list,symbol_remove,rex_remove_contain_number):
    sentence_list=nltk.sent_tokenize(data)
    unique_aword_np=np.array(unique_aword_list)
    prefix_words=[]
    prefix_words_number=[]
    print('the number of sentences:%s'%(len(sentence_list)))
    count=0
    for se in  sentence_list:
        count+=1
        word_list=clean_text(se,symbol_remove,rex_remove_contain_number)        
        if count%10==0:
            print(count)
        for index in range(len(word_list)):
            if index==0:
                prefix_words.append((-1,-1,word_list[index]))
            elif index ==1:
                prefix_words.append((-1,word_list[index-1],word_list[index]))            
            else:
                prefix_words.append((word_list[index-2],word_list[index-1],word_list[index]))
    prefix_words_number.append([(np.argwhere(unique_aword_np==wo1),np.argwhere(unique_aword_np==wo2),np.argwhere(unique_aword_np==wo3))   for wo1,wo2,wo3 in  prefix_words])
    return prefix_words_number
def clean_prefix_words_number(prefix_words_number):   
    prefix_words_number_copy=[]
    for w1,w2,w3 in  prefix_words_number[0]:
        if len(w1)>0:            
            temp_list=[w1[0][0]]
        else:
            temp_list=[-1]
        if len(w2)>0:
            temp_list.append(w2[0][0])
        else:
            temp_list.append(-1)
        if len(w3)>0:            
            temp_list.append(w3[0][0])
        else:
            temp_list.append(-1)
        prefix_words_number_copy.append(temp_list)
    return prefix_words_number_copy
def construct_year_prefix_set(vocabulary_order_2_indices):
    vocabulary_order_2_dict={}
    for wo_index1,wo_index2,wo_index3 in vocabulary_order_2_indices:        
        if  wo_index3 not in  vocabulary_order_2_dict:
            vocabulary_order_2_dict[wo_index3]=[wo_index1,wo_index2]
        else :
            vocabulary_order_2_dict[wo_index3]+=[wo_index1,wo_index2]
    return vocabulary_order_2_dict
def get_final_train_set(w_v_model,w_v_size,max_freature_number,year_from,year_to,unique_aword_list,train_set_by_year,words_fre_all_year,is_predict_data=False):
    predict_words=[]
    X=[]
    target=[]
    default=[-1 for i in range(w_v_size)]
    for year in  range(year_from,year_to) :
        if not is_predict_data and (year ==2017 or year<=2010):
            continue
        for ws in  range(len(unique_aword_list)):
            temp=[]
            if ws in  train_set_by_year[year]:
                if len(train_set_by_year[year][ws])<=max_freature_number:  
#                    temp=train_set_by_year[year][ws]
                    temp+=[list(w_v_model.wv[unique_aword_list[wo_index]])  if  unique_aword_list[wo_index] in  w_v_model.wv  else default  for wo_index in  train_set_by_year[year][ws] ]
                    
                else:
#                    temp=train_set_by_year[year][ws][:max_freature_number]
                    temp+=[ w for  w  in   [ list(w_v_model.wv[unique_aword_list[wo_index]]) if  unique_aword_list[wo_index] in  w_v_model.wv  else default  for wo_index in  train_set_by_year[year][ws][:max_freature_number] ]]
            temp+=[default for i in range(max_freature_number-len(temp))]
#            if unique_aword_list[ws] in words_fre_all_year[year]['fre']:
#                temp.append(int(words_fre_all_year[year]['fre'][unique_aword_list[ws]]))
##                temp+=[ w_v_model.wv[unique_aword_list[wo_index]]  for wo_index in  train_set_by_year[year][ws] ]
#            else:
#                temp+=[0]
            if ws in  train_set_by_year[year-1]:
                if len(train_set_by_year[year-1][ws])<=max_freature_number:  
#                    temp+=train_set_by_year[year-1][ws]
                    temp+=[list(w_v_model.wv[unique_aword_list[wo_index]])  if  unique_aword_list[wo_index] in  w_v_model.wv  else default  for wo_index in  train_set_by_year[year-1][ws] ]
                else:
#                    temp+=train_set_by_year[year-1][ws][:max_freature_number]                    
                    temp+=[ list(w_v_model.wv[unique_aword_list[wo_index]])  if  unique_aword_list[wo_index] in  w_v_model.wv  else default for wo_index in  train_set_by_year[year-1][ws][:max_freature_number] ]
            temp+=[default for i in range(2*max_freature_number-len(temp))]
            if ws in  train_set_by_year[year-2]:
                if len(train_set_by_year[year-2][ws])<=max_freature_number:  
#                    temp+=train_set_by_year[year-1][ws]
                    temp+=[list(w_v_model.wv[unique_aword_list[wo_index]])  if  unique_aword_list[wo_index] in  w_v_model.wv  else default  for wo_index in  train_set_by_year[year-2][ws] ]
                else:
#                    temp+=train_set_by_year[year-1][ws][:max_freature_number]                    
                    temp+=[ list(w_v_model.wv[unique_aword_list[wo_index]])  if  unique_aword_list[wo_index] in  w_v_model.wv  else default for wo_index in  train_set_by_year[year-2][ws][:max_freature_number] ]
            temp+=[default for i in range(3*max_freature_number-len(temp))]   
            if ws in  train_set_by_year[year-3]:
                if len(train_set_by_year[year-3][ws])<=max_freature_number:  
#                    temp+=train_set_by_year[year-1][ws]
                    temp+=[list(w_v_model.wv[unique_aword_list[wo_index]])  if  unique_aword_list[wo_index] in  w_v_model.wv  else default  for wo_index in  train_set_by_year[year-3][ws] ]
                else:
#                    temp+=train_set_by_year[year-1][ws][:max_freature_number]                    
                    temp+=[ list(w_v_model.wv[unique_aword_list[wo_index]])  if  unique_aword_list[wo_index] in  w_v_model.wv  else default for wo_index in  train_set_by_year[year-3][ws][:max_freature_number] ]
            temp+=[default for i in range(4*max_freature_number-len(temp))]               
#            if unique_aword_list[ws] in  words_fre_all_year[year-1]['fre']:
#                temp.append(int(words_fre_all_year[year-1]['fre'][unique_aword_list[ws]]))
#            else:
#                temp+=[0]     
#            if ws in  train_set_by_year[year-2]:
#                if len(train_set_by_year[year-2][ws])<=max_freature_number:  
#                    temp+=train_set_by_year[year-2][ws]
#                else:
#                    temp+=train_set_by_year[year-2][ws][:max_freature_number]                    
#            temp+=[default for i in range(3*max_freature_number+2-len(temp))]
#            if unique_aword_list[ws] in  words_fre_all_year[year-2]['fre']:
#                temp.append(int(words_fre_all_year[year-2]['fre'][unique_aword_list[ws]]))
#            else:
#                temp+=[0]   
#            if ws in  train_set_by_year[year-3]:
#                if len(train_set_by_year[year-3][ws])<=max_freature_number:  
#                    temp+=train_set_by_year[year-3][ws]
#                else:
#                    temp+=train_set_by_year[year-3][ws][:max_freature_number]                    
#            temp+=[default for i in range(4*max_freature_number+3-len(temp))]
#            if unique_aword_list[ws] in  words_fre_all_year[year-3]['fre']:
#                temp.append(int(words_fre_all_year[year-3]['fre'][unique_aword_list[ws]]))
#            else:
#                temp+=[0]                   
    #        temp+=[total_feature['fre'][unique_aword_list[ws]]]      
    #        if unique_aword_list[ws] in  words_fre_all_year[year]:
    #            temp+=words_fre_all_year[year][unique_aword_list[ws]]
    #        else:
    #            temp+=[0]
            temp_temp=[]
            for  ts in  temp:
                if not isinstance( ts,list):
                    temp_temp.append(ts)
                else:
                    for t in ts:
                        temp_temp.append(t)
            X.append(temp_temp)
            if not is_predict_data:
                if ws in   train_set_by_year[year+1]:
                    target.append(1)
                else:
                    target.append(0)
            else:
                predict_words.append(unique_aword_list[ws])
    return (X,target,predict_words)
    
def get_word_sentence_list(text,symbol_remove,rex_remove_contain_number):            
    sentence_list=nltk.sent_tokenize(text)
    w_s_list=[]
    for s in  sentence_list:
        w_s_list.append(clean_text(s,symbol_remove,rex_remove_contain_number))
    return w_s_list      
symbol_remove=['I',',','.','?','$','%','&','-','--','!',':',';','','-','）','（','￡','．',' ','\'']            
rex_remove_contain_number=re.compile(r'[\d+\.+^-^）ⅢⅡ→．+￡”+“+‘+~–+^_+\\+&,+\'+\-+，+ⅲⅱ^—+\(+\)+\[+\]+]|【d】|【c】|【b】|【a】|\s+|``|\[a\]|\[b\]|\[c\]|\[d\]|')    
#

top_path='''./source/'''
data=''
totl_year_count=0
for dirpath, dirnames, filenames in  os.walk(top_path):
    for filename in  filenames:
        totl_year_count+=1
        with open(top_path+filename,'r') as fp:
            data+=fp.read()
            
w_s_list=get_word_sentence_list(data,symbol_remove,rex_remove_contain_number)
w_v_window=10
w_v_size=2
if not os.path.exists('./w_v_model'):
    w_v_model = Word2Vec(w_s_list, size=w_v_size,window=w_v_window, min_count=5,batch_words=10)
    w_v_model.save('./w_v_model')
else:
    w_v_model=Word2Vec.load('./w_v_model')
total_feature=extract_feature(data,symbol_remove,rex_remove_contain_number)
unique_aword_list=sorted(list(total_feature['fre'].keys()))

#a=extract_train_feature(data,unique_aword_list)
#word_tag=[nltk.pos_tag(i) for  i in  unique_aword_list ]
train_set_by_year={}
words_fre_all_year={}
for dirpath, dirnames, filenames in  os.walk(top_path):
    for year_count,filename in  zip(range(len(filenames)),filenames):
#        if year_count<4:
#            continue
        print(year_count)
        with open(top_path+filename,'r') as fp:
            data=fp.read()
#        w_s_list=get_word_sentence_list(data,symbol_remove,rex_remove_contain_number)
#        model.wv[w_s_list[0]]
        print('handle data')
        if   not  os.path.exists('./pickle/%s_vocabulary_order_2_indices.pkl'%(year_count+2018-totl_year_count)):
            vocabulary_order_2=extract_train_feature(data,unique_aword_list,symbol_remove,rex_remove_contain_number)
            vocabulary_order_2_indices=clean_prefix_words_number(vocabulary_order_2)
            with open('./pickle/%s_vocabulary_order_2_indices.pkl'%(year_count+2018-totl_year_count),'wb') as fp :
                pickle.dump(vocabulary_order_2_indices,fp)                    
        else:
            with open('./pickle/%s_vocabulary_order_2_indices.pkl'%(year_count+2018-totl_year_count),'rb') as fp :
                vocabulary_order_2_indices=pickle.load(fp)
        vocabulary_order_2_dict=construct_year_prefix_set(vocabulary_order_2_indices) 
        train_set_by_year[year_count+2018-totl_year_count]=vocabulary_order_2_dict
        words_fre=extract_feature(data,symbol_remove,rex_remove_contain_number)
        words_fre_all_year[year_count+2018-totl_year_count]=words_fre
current_max=-1
for year in  train_set_by_year:
    for ws in  train_set_by_year[year]:
        if current_max < len(train_set_by_year[year][ws]):
            current_max=len(train_set_by_year[year][ws])
            
#   构建学习集，并评估模型         
X_year={}            
X=[]
target=[]    
max_freature_number=20
X,target,_=get_final_train_set(w_v_model,w_v_size,max_freature_number,2008,2018,unique_aword_list,train_set_by_year,words_fre_all_year,is_predict_data=False)
#    X_year[year]=X.copy()
        
X_arr=np.array(X)

target_arr=np.array(target)

X_arr, X_test, target_arr, y_test = train_test_split(X_arr, target_arr, test_size=0.07, random_state=42)
one_X=X_arr[target_arr==1]
one_target=target_arr[target_arr==1]
sampling_list=list(range(target_arr[target_arr==1].shape[0]))
X_sample_index=[]
for i in  range(target_arr[target_arr==0].shape[0]):
    X_sample_index.append(random.choice(sampling_list))
one_X=one_X[X_sample_index]
zero_X=X_arr[target_arr==0]
one_target=one_target[X_sample_index]
zero_target=target_arr[target_arr==0]
X_arr=np.array(list(one_X)+list(zero_X))
target_arr=np.array(list(one_target)+list(zero_target))
#target_arr=target_arr.reshape((target_arr.shape[1],1))
shuffled_index=list(range(X_arr.shape[0]))
random.shuffle(shuffled_index)
X_arr=X_arr[shuffled_index]
target_arr=target_arr[shuffled_index]

print('start train!')

time_index_size=max_freature_number*w_v_size
#
X_arr=X_arr.reshape((X_arr.shape[0],X_arr.shape[1]//time_index_size,time_index_size))
X_test=X_test.reshape((X_test.shape[0],X_test.shape[1]//time_index_size,time_index_size))
#target_arr=target_arr.reshape((target_arr.shape[0],1,1))
ls=MYLSTM(X_arr.shape[2],X_arr.shape[1],'./NN_model/lstm.h5')
ls.fit(X_arr,target_arr,nb_epoch=360)
ls.save_model()

y_pred=ls.predict(X_test)
#y_pred=y_pred.reshape((y_pred.shape[0],))
th=0.5
y_pred_p=y_pred.copy()
y_pred_p[y_pred>th]=1
y_pred_p[y_pred<=th]=0
print(classification_report(y_test, y_pred_p))




y_pred=ls.predict(X_arr[:3000])
#y_pred=y_pred.reshape((y_pred.shape[0],))
#target_arr=target_arr.reshape((target_arr.shape[0],1))
y_pred_p=y_pred.copy()
y_pred_p[y_pred>th]=1
y_pred_p[y_pred<=th]=0
print(classification_report(target_arr[:3000], y_pred_p))




#输出

predict_X=[]
predict_words=[]
predict_X,_,predict_words=get_final_train_set(w_v_model,w_v_size,max_freature_number,2017,2018,unique_aword_list,train_set_by_year,words_fre_all_year,is_predict_data=True)
predict_X=np.array(predict_X)
#predict_X=predict_X.reshape((predict_X.shape[0],1,predict_X.shape[1]))
predict_X=predict_X.reshape((predict_X.shape[0],predict_X.shape[1]//time_index_size,time_index_size))
output_prob=ls.predict(predict_X)
#output_prob=output_prob.reshape((output_prob.shape[0],1))
y_pred_p=output_prob.copy()
y_pred_p[output_prob>0.5]=1
y_pred_p[output_prob<=0.5]=0
#y_pred_p=y_pred_p.reshape((y_pred_p.shape[0],1))
output_df=pd.DataFrame(predict_words,columns=['word'])
output_df['prob']=output_prob
output_df['result']=y_pred_p

output_df.to_csv('./predict_result.csv',index=False,encoding='utf-8')
print(output_df)
