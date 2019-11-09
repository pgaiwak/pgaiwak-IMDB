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
nltk.download('stopwords')

#getting the data
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/List_of_Warriors_characters').read()
#parsing using LXML
soup = bs.BeautifulSoup(source,'lxml')

text = " "

#Fetching text from paragraph tag

for paragraph in soup.find_all('p'):#tag of the article: tags can be different like div, span
    text += paragraph.text

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
            if len(sent.split(' ')) <30:
                if sent not in sent_score.keys():
                    sent_score[sent] = word_count[word]
                else:
                    sent_score[sent] += word_count[word]
                
        
#Getting the summary: Top n sentences with highest score
                    
import heapq
best_sentences = heapq.nlargest(5,sent_score,key=sent_score.get)
for sent in best_sentences:
    print(sent)