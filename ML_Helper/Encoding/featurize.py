import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix
import scipy
import os
import sys

class Featurizer:
    
    """
    By   : karthikchiru12@gmail.com
    """
    
    def __init__(self, x_train,x_test,x_holdout=[]):
        
        """
        x_train     : training design matrix
        x_test      : testing design matrix
        x_holdout   : holdout design matrix
        """
        self.x_train = x_train
        self.x_test = x_test
        self.x_holdout = x_holdout
        if len(x_holdout) == 0:
            self.holdout_set = False
            assert (x_train.shape[1] == x_test.shape[1]),"Please check if given train, test sets have same number of columns"

        else:
            self.holdout_set = True
            assert (x_train.shape[1] == x_test.shape[1] == x_holdout.shape[1]),"Please check if given train, test, holdout sets have same number of columns"
            
    
    def vectorizeCategoricalVariables(self,features,returnVectorizerObjects = False):
        
        """
        Vectorizes the given categorical features and returns them
        #####################################
        features : names of the features to be vectorized
        """
        
        vecObjs = {}
        x_train_result = []
        x_test_result  = []
        x_holdout_result = []
        
        temp = None
        
        for i in features:
            temp = self.x_train[i].str.replace(" ","_").values
            self.x_train[i] = temp
            temp = self.x_test[i].str.replace(" ","_").values
            self.x_test[i] = temp
            if self.holdout_set!= False:
                temp = self.x_holdout[i].str.replace(" ","_").values
                self.x_holdout[i] = temp       
        
        for i in features:  
            vectorizer = CountVectorizer().fit(self.x_train[i])
            x_train_result.append(vectorizer.transform(self.x_train[i]))
            x_test_result.append(vectorizer.transform(self.x_test[i]))
            if self.holdout_set!= False:
                x_holdout_result.append(vectorizer.transform(self.x_holdout[i]))
                self.x_holdout = self.x_holdout.drop([i],axis=1)
            self.x_train = self.x_train.drop([i],axis=1)
            self.x_test = self.x_test.drop([i],axis=1)
            vecObjs[i] = vectorizer
        
        if returnVectorizerObjects == True:
            return vecObjs, x_train_result,x_test_result,x_holdout_result   
        elif self.holdout_set == False:
            return x_train_result,x_test_result,None
        else:
            return x_train_result,x_test_result,x_holdout_result
        
        
    def normalizeNumericalVariables(self,features,returnNormalizerObjects = False):
        
        """
        Normalizes the given numerical features and returns them
        #####################################
        features : names of the features to be normalized
        """
        
        normObjs = {}
        x_train_result = []
        x_test_result  = []
        x_holdout_result = []
        
        for i in features:
            normalizer = Normalizer().fit(self.x_train[i].values.reshape(-1,1))
            x_train_result.append(normalizer.transform(self.x_train[i].values.reshape(-1,1)))
            x_test_result.append(normalizer.transform(self.x_test[i].values.reshape(-1,1)))
            if self.holdout_set!= False:
                x_holdout_result.append(normalizer.transform(self.x_holdout[i].values.reshape(-1,1)))
                self.x_holdout = self.x_holdout.drop([i],axis=1)
            normObjs[i] = normalizer
            self.x_train = self.x_train.drop([i],axis=1)
            self.x_test = self.x_test.drop([i],axis=1)
        
        if returnNormalizerObjects == True:
            return normObjs, x_train_result,x_test_result,x_holdout_result   
        elif self.holdout_set == False:
            return x_train_result,x_test_result,None
        else:
            return x_train_result,x_test_result,x_holdout_result
    
    def stackFeatures(self,leftOver,categorical,numerical):
        
        """
        Stacks the given numerical,categorical and leftover features
        """
        stackedSet = []
        
        for i in range(leftOver.shape[1]):
            stackedSet.append(leftOver.iloc[:,i].values.reshape(-1,1))
        for i in categorical:
            stackedSet.append(i.todense())
        for j in numerical:
            stackedSet.append(j)
        #stackedSet = csr_matrix(scipy.sparse.hstack(tuple(stackedSet)))
        
        return stackedSet
    
    def featurize(self, numerical, categorical, returnObjs =False):
        
        """
        Encodes the numerical and categorical features and returns the stacked features
        """
        #numerical = self.x_train.select_dtypes(include=np.number).columns.tolist()
        #categorical = self.x_train.select_dtypes(include=np.object).columns.tolist()
        if returnObjs == True:
            vec,tr_vec,ts_vec,hd_vec = self.vectorizeCategoricalVariables(categorical,True)
            norm,tr_norm,ts_norm,hd_norm = self.normalizeNumericalVariables(numerical,True)
            x_train_final = self.stackFeatures(self.x_train,tr_vec,tr_norm)
            x_test_final = self.stackFeatures(self.x_test,ts_vec,ts_norm)
            if self.holdout_set == True:
                x_holdout_final = self.stackFeatures(self.x_holdout,hd_vec,hd_norm)
                return x_train_final,x_test_final,x_holdout_final,vec,norm
            else:
                return x_train_final,x_test_final,vec,norm
        elif returnObjs == False:
            tr_vec,ts_vec,hd_vec = self.vectorizeCategoricalVariables(categorical,False)
            tr_norm,ts_norm,hd_norm = self.normalizeNumericalVariables(numerical,False)
            x_train_final = self.stackFeatures(self.x_train,tr_vec,tr_norm)
            x_test_final = self.stackFeatures(self.x_test,ts_vec,ts_norm)
            if self.holdout_set == True:
                x_holdout_final = self.stackFeatures(self.x_holdout,hd_vec,hd_norm)
                return x_train_final,x_test_final,x_holdout_final
            else:
                return x_train_final,x_test_final
        else:
            raise Exception("Please provide numerical and categorical features")
            
