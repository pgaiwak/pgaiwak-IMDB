# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:42:49 2018

@author: pgaiw
"""

import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import pickle
nltk.download('stopwords')

dataset = load_files('IMDB/')
reviews,label = dataset.data, dataset.target

corpus = []
for i in range(len(reviews)):
    rev = re.sub(r'\W',' ',str(reviews[i]))
    rev = rev.lower()
    rev = re.sub(r'\s+[a-z]\s+',' ',rev)
    rev = re.sub(r'^[a-z]\s+',' ',rev)
    rev = re.sub(r'\s',' ',rev)
    rev = re.sub(r'\s\sbr\s\s',' ',rev)
    corpus.append(rev)

#min document freq: term less than 3 docs, dont include
#max_df: if word appears more then 70% of time
#max_fweatures: Maximum number of features
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features = 5000, min_df = 3, max_df=0.7,stop_words = stopwords.words('English'))
BOW = vect.fit_transform(corpus).toarray()
            
from sklearn.feature_extraction.text import TfidfTransformer
tf_idf = TfidfTransformer()
X = tf_idf.fit_transform(BOW).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features = 5000, min_df = 3, max_df=0.7,stop_words = stopwords.words('English'))
BOW = vect.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
review_train,review_test,label_train,label_test = train_test_split(X,label,test_size = 0.2,random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier_1 = LogisticRegression()
classifier_1.fit(review_train,label_train)
pred_labels_1 = classifier_1.predict(review_test)
c1 = LogisticRegression()


from sklearn.svm import LinearSVC
classifier_2 = LinearSVC()
classifier_2.fit(review_train,label_train)
pred_labels_2 = classifier_2.predict(review_test)
c2 = LinearSVC()


from sklearn.naive_bayes import MultinomialNB
classifier_3 = MultinomialNB()
classifier_3.fit(review_train,label_train)
pred_labels_3 = classifier_3.predict(review_test)
c3 = MultinomialNB()



from sklearn.ensemble import RandomForestClassifier, VotingClassifier
classifier_4 = RandomForestClassifier(n_estimators=100)
classifier_4.fit(review_train,label_train)
pred_labels_4 = classifier_4.predict(review_test)
c4 = RandomForestClassifier(n_estimators=100)



from sklearn.tree import DecisionTreeClassifier
classifier_5 = DecisionTreeClassifier()
classifier_5.fit(review_train,label_train)
pred_labels_5 = classifier_5.predict(review_test)
c5 = DecisionTreeClassifier()


pred_labels = []
pred_labels_1 = pred_labels_1.tolist()
pred_labels_2 = pred_labels_2.tolist()
pred_labels_3 = pred_labels_3.tolist()
pred_labels_4 = pred_labels_4.tolist()
pred_labels_5 = pred_labels_5.tolist()

for i in range(len(pred_labels_5)):
    x = pred_labels_1[i] + pred_labels_2[i] + pred_labels_3[i] + pred_labels_4[i] + pred_labels_5[i]
    if x>= 3:
        pred_labels.append(1)
    if x < 3:
        pred_labels.append(0)

eclf1 = VotingClassifier(estimators=[
     ('lr', c1), ('svm', c2), ('mnb', c3),('rf',c4),('dtf',c5)], voting='hard')
eclf1 = eclf1.fit(review_train,label_train)
pred_labels_eclf = eclf1.predict(review_test)

#pred_labels = classifier.predict(review_test)

from sklearn.metrics import confusion_matrix
m = confusion_matrix(label_test,pred_labels_eclf)

#pickling the classifier
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
#pickling the vectorizer
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vect,f)