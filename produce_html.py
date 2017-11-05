# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:51:04 2017

@author: Administrator
"""

from nltk import  word_tokenize
from nltk.corpus import stopwords
import numpy as np
import os
import nltk
import pandas as pd
import re
import requests
import json
from bs4 import BeautifulSoup
def clean_text(text,symbol_remove,rex_remove_contain_number):
    text=text.replace('’','\'')
    rex_remove_number=re.compile(r'^\d+$')
    data_text=word_tokenize(text)
#    removing number
    data_text=[ rex_remove_contain_number.sub('',d.lower()) for d in data_text if d not in symbol_remove  and len(rex_remove_number.findall(d))==0 ]
#    cleaning data        
    reovmed_stop_word=[d for  d  in  data_text if d not in  stopwords.words('english') and d != '']
    return reovmed_stop_word
def extract_feature(data,symbol_remove,rex_remove_contain_number):
#    data=clean_paper(data)
    
    data_text=clean_text(data,symbol_remove,rex_remove_contain_number)
    data_df=pd.DataFrame(data_text,columns=['word'])
    data_df['fre']=data_df['word']
    return  data_df.groupby('word').count()
#def 
def get_word_inter(word):
    print(word)
    output_text=''
    url='http://dict.youdao.com/w/%s/'
    my_header={
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
    }    
    resp=requests.get(url%(word),headers=my_header)
    ht=resp.content.decode('utf-8')
    soup=BeautifulSoup(ht,'html.parser')
    try:
        means=soup.find('div',class_='trans-container').text
    except Exception as e:
        means=''
    try:
        for div  in soup.find_all('div',id='examplesToggle') :
            output_text=str(means)+'\n'+str(div.find('ul').text)
            print(output_text)
            return output_text
    except Exception as e:
        output_text=means
        return output_text
quering_word_filename='''./predict_result.csv'''
#quering_word_filename='''./all_vocabulary.csv'''
if quering_word_filename=='''./predict_result.csv''':
    html_prefix_name='''html_prefix_name_predict_%s'''
else:
    html_prefix_name='''html_prefix_name_%s'''
if not os.path.exists(quering_word_filename):    
    symbol_remove=['I',',','.','?','$','%','&','-','--','!',':',';','','-','）','（','￡','．',' ','\'']            
    rex_remove_contain_number=re.compile(r'[\d+\.+^-^）ⅢⅡ→．+￡”+“+‘+~–+^_+\\+&,+\'+\-+，+ⅲⅱ^—+\(+\)+\[+\]+]|【d】|【c】|【b】|【a】|\s+|``|\[a\]|\[b\]|\[c\]|\[d\]|')            
    top_path='''./source/'''
    data=''
    totl_year_count=0
    for dirpath, dirnames, filenames in  os.walk(top_path):
        for filename in  filenames:
            totl_year_count+=1
            with open(top_path+filename,'r') as fp:
                data+=fp.read()
#sentence_list=nltk.sent_tokenize(data)
    total_feature=extract_feature(data,symbol_remove,rex_remove_contain_number)
    total_feature.to_csv(quering_word_filename,encoding='utf-8')
else:
    total_feature=pd.read_csv(quering_word_filename)
    total_feature.columns=[i.strip('﻿') for   i in  total_feature.columns]
    if len(total_feature.columns)>2:
        total_feature.columns=['word','fre','result']
        total_feature=total_feature.query("result  == 1 ")
    total_feature.index=total_feature.pop('word')
with open('./word.json','r',encoding='utf-8')   as fp:
    word_meaning_json_text=fp.read()
word_meaning_json=json.loads(word_meaning_json_text)
with open('./template/word_template.html','r',encoding='utf-8')   as fp:
    word_tamplalte=fp.read()
def map_wrod_mean(word,word_meaning_json):
    for w in word_meaning_json:
        if w['word'] == word:
            return w['meaning']+'::::::'+w['example_sentence']
total_feature['w']=total_feature.index
total_feature['meaning']=total_feature['w'].map(lambda x :map_wrod_mean(x,word_meaning_json) )            
total_feature=total_feature.dropna()
#temp=total_feature.pop('w')
line_rex=re.compile('\n+')
black_rex=re.compile('\s+')
total_feature['example_sentence']=total_feature['meaning'].map(lambda x :x.split('::::::')[1])
total_feature['example_sentence']=total_feature['example_sentence'].map(lambda x :x.strip())
total_feature['example_sentence']=total_feature['example_sentence'].map(lambda x :line_rex.sub('\n',x))
total_feature['example_sentence']=total_feature['example_sentence'].map(lambda x :black_rex.sub(' ',x))
total_feature['meaning']=total_feature['meaning'].map(lambda x :x.split('::::::')[0])
total_feature['meaning']=total_feature['meaning'].map(lambda x :black_rex.sub(' ',x))



def produce_html(word,freqency,meaning,example_sentence):
    temp_html='''
    <tr><td>%s</td><td>%s</td><td><pre>%s</pre></td><td><pre>%s</pre></td></tr>    
    '''%(word,freqency,meaning,example_sentence)    
    return    temp_html
total_feature=total_feature.sort_values('fre',ascending=False)
html_list=[]

for  i in range(total_feature['meaning'].count()):
    html_list.append(produce_html(total_feature.ix[i].name,total_feature.ix[i]['fre'],total_feature.ix[i]['meaning'],total_feature.ix[i]['example_sentence']))
if  quering_word_filename == '''./predict_result.csv''':    
    file_num=20
else:    
    file_num=50
block_size=len(html_list)//file_num
for i in  range(file_num):    
    start_index=i*block_size
    end_index=i*block_size+block_size
    if i==file_num-1:
        end_index=len(html_list)
    output_html=word_tamplalte.replace('<mytr/>',''.join(html_list[start_index:end_index]))    
    if i>0:
        output_html=output_html.replace('<prepage/>','./'+html_prefix_name%(i-1)+'.html')    
    else:
        output_html=output_html.replace('<prepage/>','javascript:void(0);')    
    if i<file_num-1:
        output_html=output_html.replace('<nextpage/>','./'+html_prefix_name%(i+1)+'.html')    
        output_html=output_html.replace('<nextpage/>','javascript:void(0);')  
    th='''<tr><th>单词</th><th>%s</th><th>翻译</th><th>例句</th></tr>'''
    if  quering_word_filename == '''./predict_result.csv''':
        output_html=output_html.replace('<myth/>',th%('概率'))
        output_html=output_html.replace('<mynav1/>','2018年预测词汇')        
    else:
        output_html=output_html.replace('<myth/>',th%('词频'))  
        output_html=output_html.replace('<mynav1/>','历年真题词汇')
    output_html=output_html.replace('<mynav2/>','第%s页'%(i+1))
    with open('./web/templates/link/'+html_prefix_name%(i)+'.html','w',encoding='utf-8') as fp:
        fp.write(output_html)
    
#total_feature['w']=total_feature.index        
#total_feature['mean']=total_feature['w'].map(lambda x : get_word_inter(x))     
#temp=total_feature.pop('w')
#total_feature.to_csv('./vocabulary_mean.csv')
#with open('./template/word_template.html','r') as fp:    
#    html_tamplate=fp.read()
#html_tamplate.replace(r'<mytr/>')    
    
#unique_aword_list=sorted(list(total_feature['fre'].keys()))    