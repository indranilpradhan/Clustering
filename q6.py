import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import re
import os
from nltk.corpus import stopwords
from collections import Counter
import math

class Cluster:
    def __init__(self, k=5, iteration = 20):
        self.k = k
        self.iteration = iteration
        
    def euclidean_distance(self,x,c):
        sqrsum = 0
        for i in range(len(x)):
            sqrsum = sqrsum + ((x[i] - c[i]) ** 2)
        return math.sqrt(sqrsum)
    
    def cluster(self, testset_path):
        df = pd.DataFrame(columns = ['filename','text', 'label']) 
        base_dir = "/media/indranil/New Volume/second sem/SMAI/Assignment 2/q6/data/dataset/"
        for filename in os.listdir(base_dir):
            path = os.path.join(base_dir, filename)
            with open(path, "r", encoding='latin1') as file:
                text = file.read()
                label = filename[filename.find("_")+1:filename.find(".")]
                df = df.append({'filename':filename,'text': text, 'label':label}, ignore_index=True)
        df.loc[:,"text"] = df.text.apply(lambda x : str.lower(x))
        df.loc[:,"text"] = df.text.apply(lambda x : " ".join(re.findall('[\w]+',x)))
        df["text"] = df['text'].str.replace('[^\w\s]','')
        df.loc[:,"text"] = df.text.apply(lambda x: x.strip())
        tfvect = TfidfVectorizer(stop_words = 'english')
        tfdf = tfvect.fit_transform(df['text'])
        tfdfnorm = normalize(tfdf)
        X_train = tfdfnorm.toarray()
        centroids = X_train[np.random.choice(X_train.shape[0],self.k,replace = False)]
        clusters = np.zeros(X_train.shape[0])
        for it in range(self.iteration):
            cluster_dist_1 = -1
            cluster_dist_2 = -1
            cluster_dist_3 = -1
            cluster_dist_4 = -1
            cluster_dist_5 = -1
            for i in range(X_train.shape[0]):
                cluster_dist_1 = self.euclidean_distance(X_train[i],centroids[0])
                cluster_dist_2 = self.euclidean_distance(X_train[i],centroids[1])
                cluster_dist_3 = self.euclidean_distance(X_train[i],centroids[2])
                cluster_dist_4 = self.euclidean_distance(X_train[i],centroids[3])
                cluster_dist_5 = self.euclidean_distance(X_train[i],centroids[4])
                min_dist = min(cluster_dist_1,cluster_dist_2,cluster_dist_3,cluster_dist_4,cluster_dist_5)
                if(min_dist == cluster_dist_1):
                    clusters[i] = 1
                elif(min_dist == cluster_dist_2):
                    clusters[i] = 2
                elif(min_dist == cluster_dist_3):
                    clusters[i] = 3
                elif(min_dist == cluster_dist_4):
                    clusters[i] = 4
                elif(min_dist == cluster_dist_5):
                    clusters[i] = 5
            np_c1 = []
            np_c2 = []
            np_c3 = []
            np_c4 = []
            np_c5 = []
            
            for i in range(clusters.shape[0]):
                if(clusters[i] == 1):
                    np_c1.append(X_train[i])
            for i in range(clusters.shape[0]):
                if(clusters[i] == 2):
                    np_c2.append(X_train[i])
            for i in range(clusters.shape[0]):
                if(clusters[i] == 3):
                    np_c3.append(X_train[i])
            for i in range(clusters.shape[0]):
                if(clusters[i] == 4):
                    np_c4.append(X_train[i])
            for i in range(clusters.shape[0]):
                if(clusters[i] == 5):
                    np_c5.append(X_train[i])
            np_c11 = np.array(np_c1)
            np_c21 = np.array(np_c2)
            np_c31 = np.array(np_c3)
            np_c41 = np.array(np_c4)
            np_c51 = np.array(np_c5)
            
            centroids[0] = np.mean(np_c11,axis=0)
            centroids[1] = np.mean(np_c21,axis=0)
            centroids[2] = np.mean(np_c31,axis=0)
            centroids[3] = np.mean(np_c41,axis=0)
            centroids[4] = np.mean(np_c51,axis=0)
        
        test_df = pd.DataFrame(columns = ['filename','text']) 
        test_dir = str(testset_path)
        for filename in os.listdir(test_dir):
            test_path = os.path.join(test_dir, filename)
            with open(test_path, "r", encoding='latin1') as file:
                text = file.read()
                test_df = test_df.append({'filename':filename,'text': text}, ignore_index=True)
                
        test_df.loc[:,"text"] = test_df.text.apply(lambda x : str.lower(x))
        test_df.loc[:,"text"] = test_df.text.apply(lambda x : " ".join(re.findall('[\w]+',x)))
        test_df["text"] = test_df['text'].str.replace('[^\w\s]','')
        test_df.loc[:,"text"] = test_df.text.apply(lambda x: x.strip())
                
        X_test = test_df['text']
        tfdf = tfvect.transform(X_test)
        tfdfnorm = normalize(tfdf)
        X_test = tfdf.toarray()
        
        Y_train = np.array(df["label"])
        
        result={}   
        cluster_dist_1 = -1
        cluster_dist_2 = -1
        cluster_dist_3 = -1
        cluster_dist_4 = -1
        cluster_dist_5 = -1
        for i in range(X_test.shape[0]):
            cluster_dist_1 = self.euclidean_distance(X_test[i],centroids[0])
            cluster_dist_2 = self.euclidean_distance(X_test[i],centroids[1])
            cluster_dist_3 = self.euclidean_distance(X_test[i],centroids[2])
            cluster_dist_4 = self.euclidean_distance(X_test[i],centroids[3])
            cluster_dist_5 = self.euclidean_distance(X_test[i],centroids[4])
            min_dist = min(cluster_dist_1,cluster_dist_2,cluster_dist_3,cluster_dist_4,cluster_dist_5)
            if(min_dist == cluster_dist_1):
                result[test_df['filename'][i]] = 1
            elif(min_dist == cluster_dist_2):
                result[test_df['filename'][i]] = 2
            elif(min_dist == cluster_dist_3):
                result[test_df['filename'][i]] = 3
            elif(min_dist == cluster_dist_4):
                result[test_df['filename'][i]] = 4
            elif(min_dist == cluster_dist_5):
                result[test_df['filename'][i]] = 5
                  
        return result   
        