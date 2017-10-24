import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import random
from collections import Counter
from nltk.stem import WordNetLemmatizer
import math
files=['pos.txt', 'neg.txt']
path='D:/NLP/'
lemmatizer = WordNetLemmatizer()
def get_all_words():
    l=[]
    for fi in files:
        f=open(fi,'r')
        lines=f.readlines()
        for line in lines:
            words=list(word_tokenize(line))
            words=[lemmatizer.lemmatize(word.lower()) for word in words]
            l+=words
        f.close()
    l=Counter(l)
    ret=[i for i in l if(50<l[i]<1000)]
    return ret
def create_map(l):
    dic={}
    idx=0
    for word in l:
        dic[word]=idx
        idx+=1
    return dic
def get_features(l, dic):
    features=[]
    n=len(l)
    for fi in files:
        f=open(fi,'r')
        lines=f.readlines()
        for line in lines:
            words=list(word_tokenize(line))
            words=[lemmatizer.lemmatize(word.lower()) for word in words]
            x=[0]*n
            for word in words:
                if(word in dic): x[dic[word]]+=1
            if fi=='pos.txt':
                features.append([x,[1,0]])
            else:
                features.append([x,[0,1]])
        f.close()
    random.shuffle(features)
    return features
def get_data(f, test_fraction):
    n=len(f)
    n_test=int(n*test_fraction)
    n_train=n-n_test
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    train=f[0:n_train]
    test=f[n_train:]
    for i in train:
        X_train.append(i[0])
        Y_train.append(i[1])
    for i in test:
        X_test.append(i[0])
        Y_test.append(i[1])
    X_train=np.array(X_train).T
    Y_train=np.array(Y_train).T
    X_test=np.array(X_test).T
    Y_test=np.array(Y_test).T
    return (X_train,Y_train,X_test,Y_test)
def create_batches(sz, X_train, Y_train):
    batches=[]
    n=X_train.shape[1]
    num_full_batches=n//sz
    for i in range(num_full_batches):
        batches.append((X_train[:,i*sz:(i+1)*sz],Y_train[:,i*sz:(i+1)*sz]))
    batches.append((X_train[:,num_full_batches*sz:],Y_train[:,num_full_batches*sz:]))
    return batches
def main():
    all_words=get_all_words()
    dic=create_map(all_words)
    return get_data(get_features(all_words,dic), 0.3)
