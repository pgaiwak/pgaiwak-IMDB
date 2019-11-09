# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:48:47 2018

@author: pgaiw
"""

import tweepy
import re
import pickle

from tweepy import OAuthHandler

'''key initialization'''
consumer_key="z7BGfdzCHK8FIB8cdzDUusmEH" 
consumer_secret="4jHk2RGWrTPMFNgoiD1vMX3yp75AqOsnRDiUsXhR7ZJD8vtnmD"
access_token="1060992364146315264-twb6jveHngj4e5OP5IL7Zd6DnRJmRk"
access_secret="LoF4si8SNUDc9vqhMXTOYLcwbqoR0Slyi0AczrSi7Bykv"

auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)

args = ['facebook']
api = tweepy.API(auth,timeout=100)

#fetching tweets

list_tweets = []

query = args[0]
if (len(args) == 1):
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets", lang='en',result_type = 'recent').items(10):
        list_tweets.append(status.text)

with open('tfidfmodel.pickle','rb') as f:
    vect = pickle.load(f)
with open('classifier.pickle','rb') as f:
    classifier = pickle.load(f)
sumx = 0
#preprocessing the tweets
for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    #print(tweet)
    sent = classifier.predict(vect.transform([tweet]).toarray())
    print(tweet,":",sent)
    sumx += sent
tot_pos = 0
tot_neg = 0
