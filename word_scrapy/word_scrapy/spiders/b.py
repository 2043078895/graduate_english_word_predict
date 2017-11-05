# -*- coding: utf-8 -*-
import scrapy
import pandas as pd
from word_scrapy.items import WordScrapyItem
class BSpider(scrapy.Spider):
    name = 'b'
#    allowed_domains = ['s']
    word_df=pd.read_csv('./all_vocabulary.csv',encoding='utf-8')
    word_df.columns=[i.strip('﻿') for   i in  word_df.columns]
#    word_df['w']=word_df.index
    word_df['url']=word_df['word'].map(lambda x : 'http://dict.youdao.com/w/%s/'%(x) )
    
    start_urls = list(word_df['url'])

    def parse(self, response):
        wsi=WordScrapyItem()
        wsi['word']=response.request.url.split('/')[-2]
        wsi['meaning']=''.join(response.xpath("//div[@id='phrsListTab']//div[@class='trans-container']/ul/*/text()").extract()).strip()
        wsi['example_sentence']=''.join(response.xpath("//div[@id='examplesToggle']//div[@id='bilingual']//*/text()").extract()).strip()
        yield wsi
