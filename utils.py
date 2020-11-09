import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import neighbors
import sklearn.metrics as sm

def digitize_price(column):

    list = []
    for ind, val in enumerate(column):
        if val[0] == "$":
            list.append(float(val[1:]))
        else:
            list.append(float(val))
        
    arr = np.array(list)
    return pd.Series(arr)

def categorize(column):
    dict = {}
    key = 1.0
    for i in np.unique(column):
        if i not in dict:
            dict[i] = key
            key+=1
    return dict


class evaluate():
    
    def __init__(self, df, kmean, testset, k):
        
        self.df = df
        self.kmean = kmean
        self.k = k
        self.testset = testset
        
    def validate(self, sample, k=1):
        centroids = self.kmean.cluster_centers_

        dist_list = []
        for ind, c in enumerate(centroids):
            dist = np.linalg.norm(c-sample)
            dist_list.append((dist, ind))

        dist_list = sorted(dist_list)
        _, L = dist_list[0] 
        test_frame = self.df[self.df.labels == L]
        a1 = len(test_frame)
        test_frame = test_frame.dropna()
        #print('the number of rows droped :', a1 - len(test_frame))
        x = test_frame[test_frame.columns[:-2]]
        y = test_frame[test_frame.columns[-2]]
        model = neighbors.KNeighborsRegressor(n_neighbors=self.k, weights='distance')
        model.fit(x,y)

        sample = np.array(sample[:-1])
        sample = sample.reshape(1,sample.shape[0])
        result = model.predict(sample)
        
        return result

    def pred(self):
        preds = []
        targets = []
        for _, sample in self.testset.iterrows():
            targets.append(sample['Rating'])
            pred = self.validate(sample[1:], self.k)
            preds.append(pred)
        return np.array(targets), np.array(preds)
    
    def accuracy(self, targets, preds):
        return round(sm.r2_score(targets, preds), 6)*100
       
        