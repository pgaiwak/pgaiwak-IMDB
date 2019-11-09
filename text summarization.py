# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:07:02 2018

@author: pgaiw
"""


#creating article summarizer
import bs4 as bs
import urllib.request
import re
import nltk
import numpy as np
#import rouge
#from rouge import Rouge
nltk.download('punkt')
nltk.download('stopwords')
import time
#loading stories
stories = np.load(open('cnn_dataset2.pkl', 'rb'))
print('Loaded Stories %d' % len(stories))

#x = stories.dropna(axis=0, how= 'any', thresh=None, subset = None, inplace = False)

ind = []
for i in range(len(stories)):
    if stories.iloc[i,1] == '':
        ind.append(i)

comp_stories = stories.drop(stories.index[ind])

df1 = comp_stories.iloc[0:9999,:]
df2 = comp_stories.iloc[10000:19999,:]
df3 = comp_stories.iloc[20000:29999,:]
df4 = comp_stories.iloc[30000:39999,:]
df5 = comp_stories.iloc[40000:49999,:]
df6 = comp_stories.iloc[50000:59999,:]
df7 = comp_stories.iloc[60000:69999,:]
df8 = comp_stories.iloc[70000:79999,:]
df9 = comp_stories.iloc[80000:89999,:]
df10 = comp_stories.iloc[90000:92465,:]

rouge_score = []
#getting the data
'''source = urllib.request.urlopen('https://en.wikipedia.org/wiki/List_of_Warriors_characters').read()
#parsing using LXML
soup = bs.BeautifulSoup(source,'lxml')

text = " "

#Fetching text from paragraph tag

for paragraph in soup.find_all('p'):#tag of the article: tags can be different like div, span
    text += paragraph.text'''
new_summary = []
stories['new_summaries'] = 0
#for i in range(len(stories)):
    text = comp_stories.iloc[4,1]
    #preprocessing the text
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'\s+',' ',text)
    clean_text = text.lower()
    clean_text = re.sub(r'\W',' ',clean_text)
    clean_text = re.sub(r'\d',' ',clean_text)
    clean_text = re.sub(r'\s+',' ',clean_text)
    
    #tokenization
    sentences = nltk.sent_tokenize(text)
    stop_words = nltk.corpus.stopwords.words('english')
    
    #creating the histogram using dictionary
    word_count = {}
    for word in nltk.word_tokenize(clean_text):
        if word not in stop_words:
            if word not in word_count.keys():
                word_count[word] = 1
            else:
                word_count[word]+=1
    for key in word_count.keys():
        word_count[key] = word_count[key]/max(word_count.values())
    
    
    #scores of sentences
    sent_score = {}
    for sent in sentences:
        for word in nltk.word_tokenize((sent).lower()):
            if word in word_count.keys():
                if len(sent.split(' ')) <1000:
                    if sent not in sent_score.keys():
                        sent_score[sent] = word_count[word]
                    else:
                        sent_score[sent] += word_count[word]
                    
            
    #Getting the summary: Top n sentences with highest score
                        
    import heapq
    best_sentences = heapq.nlargest(2,sent_score,key=sent_score.get)
    #if best_sentences == ' ':
     #   break
    
    best_sentences = ' '.join(best_sentences)
    '''if len(best_sentences) == 0:
        print(i)
        raise ValueError('Empty sentence')'''
        
    #print(best_sentences)
    #print('\n\n\n')
    
    #stories.iloc[i,2] = best_sentences
    
    #print(best_sentences)
    #len(best_sentences)
    
    r = Rouge()
    rouge_score.append(r.get_scores(best_sentences,stories.iloc[i,0]))
    