import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
from math import factorial
import os
import sys

class EDA:
    
    """
    By   : karthikchiru12@gmail.com
    """
    
    def __init__(self,data):
        
        """
        data : dataframe containing the features
        """
        self.data = data
    
    def non_numerical_features_error(self,features):
        
        """
        Returns features whih are not numerical
        """
        message =""""""
        for i in features:
            dtype = str(self.data[i].dtype)
            if dtype not in ['int64','float64']:
                message += "The feature "+i+" is not numerical (or) contains non numerical values.\n"
        return message
    
    def calculate_rows_and_columns(self,m):
        
        """
        Returns rows and columns for subplots given number of features
        """
        if m%2 == 0:
            rows = m//2
            cols = m//(m//2)
        else:
            rows = (m+1)//2
            cols = (m+1)//((m+1)//2)
        return rows, cols


    def count_plot(self,features,width=12,height=16,hue=None,title=""):
        
        """
        Plots a count plot for categorical features with less than 30 categories
        Otherwise prints a value counts for the categorical features with more than 30 categories
        #############################
        input:
            features : feature names
            width    : width of each plot
            height   : height of each plot
            hue      : Semantic variable
            title    : Title of all plots

        output:
            plots the count plots of all given categorical plots
            
        """
        variables_that_can_be_visualized = []
        variables_that_cannot_be_visualized = []
        for i in features:
            if len(self.data[i].value_counts())>15:
                variables_that_cannot_be_visualized.append(i)
            else:
                variables_that_can_be_visualized.append(i)
        # Initializing subplot parameters
        rows = 0
        cols = 0
        m = len(variables_that_can_be_visualized)
        if m!=0:
            if m==1:
                plt.figure(figsize=(width,height))
                plt.title(title+"\n\n",loc='center',fontsize=24)
                sns.countplot(y = self.data[features[0]],hue=self.data[hue] if hue!=None else None)
                return None
            elif m==2:
                fig, axes = plt.subplots(1, 2,figsize=(width,height),constrained_layout=True)
                fig.suptitle(title+"\n\n",fontsize=24)
                axes[0].title.set_text("\nCounts of : "+features[0])
                sns.countplot(y = self.data[features[0]],hue=self.data[hue] if hue!=None else None,ax=axes[0])
                axes[1].title.set_text("\nCounts of : "+features[1])
                sns.countplot(y = self.data[features[1]],hue=self.data[hue] if hue!=None else None,ax=axes[1])
                return None
            else:
                rows,cols = self.calculate_rows_and_columns(m)
                fig, axes = plt.subplots(rows, cols,figsize=(width,height),constrained_layout=True)
                fig.suptitle(title+"\n\n",fontsize=24)

                # plotting all the features
                k = 0
                for i in range(rows):
                    for j in range(cols):
                        if k > m-1:
                            axes[i,j].set_axis_off()
                            break
                        else:
                            axes[i,j].title.set_text("\nCounts of : "+features[k])
                            sns.countplot(y = self.data[features[k]],hue=self.data[hue] if hue!=None else None,ax=axes[i,j])
                            k += 1
                return None
            
        elif len(variables_that_cannot_be_visualized)!=0:
            print("\n")
            print("\t\t\tValue counts of given features :")
            for i in variables_that_cannot_be_visualized:
                print("\n{0} :\n\n{1}\n".format(i,self.data[i].value_counts()))
        
        return None
            
        
    def kde_plot(self,features, width=16, height=14, hue = None,title = ""):

        """
        Plots distribution plots for the given numerical features
        #######################
        input:
            features : feature names
            width    : width of each plot
            height   : height of each plot
            hue      : Semantic variable
            title    : Title of all plots

        output:
            plots the kdeplots of all given numerical features
            
        ######################
        """
        # https://stackoverflow.com/a/54427278
        # Asserting if all are numerical features or not
        assert self.data[features].apply(lambda col: pd.to_numeric(col, errors='coerce').notnull().all()).all(),self.non_numerical_features_error(features)
        
        # Initializing subplot parameters
        rows = 0
        cols = 0
        m = len(features)
        if m==1:
            plt.figure(figsize=(width,height))
            plt.title(title+"\n\n",loc='center',fontsize=24)
            sns.kdeplot(x = self.data[features[0]],hue=self.data[hue] if hue!=None else None)
            return None
        elif m==2:
            fig, axes = plt.subplots(1, 2,figsize=(width,height),constrained_layout=True)
            fig.suptitle(title+"\n\n",fontsize=24)
            axes[0].title.set_text("\nDistribution : "+features[0])
            sns.kdeplot(x = self.data[features[0]],hue=self.data[hue] if hue!=None else None,ax=axes[0])
            axes[1].title.set_text("\nDistribution : "+features[1])
            sns.kdeplot(x = self.data[features[1]],hue=self.data[hue] if hue!=None else None,ax=axes[1])
            return None
        else:
            rows, cols = self.calculate_rows_and_columns(m)
            fig, axes = plt.subplots(rows, cols,figsize=(width,height),constrained_layout=True)
            fig.suptitle(title+"\n\n",fontsize=24)
            # plotting all the features
            k = 0
            for i in range(rows):
                for j in range(cols):
                    if k > m-1:
                        axes[i,j].set_axis_off()
                        break
                    else:
                        axes[i,j].title.set_text("\nDistribution : "+features[k])
                        sns.kdeplot(x = self.data[features[k]],hue=self.data[hue] if hue!=None else None,ax=axes[i,j])
                        k += 1
        return None
    
    def scatter_plot(self,features, width =16, height=14, hue = None,title = ""):

        """
        Plots scatter plots for the given numerical features
        #######################
        input:
            features : feature names
            width    : width of each plot
            height   : height of each plot
            hue      : Semantic variable
            title    : Title of all plots

        output:
            plots the scatter of all given numerical features
            
        ######################
        """
        # https://stackoverflow.com/a/54427278
        # Asserting if all are numerical features or not
        assert self.data[features].apply(lambda col: pd.to_numeric(col, errors='coerce').notnull().all()).all(),self.non_numerical_features_error(features)
        assert len(features) >= 2 , "At least 2 numerical features must be passed for scatter_plot"
        # Initializing subplot parameters
        rows = 0
        cols = 0
        m = len(features)
        if m==2:
            plt.figure(figsize=(width,height))
            plt.title(title+"\n\n",loc='center',fontsize=24)
            sns.scatterplot(data=self.data, x=features[0],y=features[1],hue=self.data[hue] if hue!=None else None)
            return None
        else:
            features_combo = factorial(m)/(factorial(2)*factorial(m-2))
            rows, cols = self.calculate_rows_and_columns(int(features_combo))
            features_combos = []
            for i in permutations(features,2):
            	if i[::-1] not in features_combos:
                	features_combos.append(i)
            fig, axes = plt.subplots(rows, cols,figsize=(width,height),constrained_layout=True)
            fig.suptitle(title+"\n\n",fontsize=24)
            # plotting all the features
            k = 0
            for i in range(rows):
                for j in range(cols):
                    if k > features_combo-1:
                        axes[i,j].set_axis_off()
                        break
                    else:
                        axes[i,j].title.set_text("\nScatter plot : "+str(','.join(features_combos[k])))
                        sns.scatterplot(data=self.data, x=features_combos[k][0],y=features_combos[k][1],hue=self.data[hue] if hue!=None else None,ax=axes[i,j])
                        k += 1
        return None
    
    
    def correlation_plot(self,features, width=16, height=16,title=""):
        
        """
        Plots correlation for the given numerical features
        #######################
        input:
            features : feature names
            width    : width of each plot
            height   : height of each plot
            title    : Title of all plots

        output:
            plots the correlation of all given numerical features
        """
        
        # https://stackoverflow.com/a/54427278
        # Asserting if all are numerical features or not
        assert self.data[features].apply(lambda col: pd.to_numeric(col, errors='coerce').notnull().all()).all(),self.non_numerical_features_error(features)
        assert len(features) >= 2 , "At least 2 numerical features must be passed for correlaton"
        
        corr = self.data[features].corr()
        plt.figure(figsize=(width,height))
        sns.heatmap(corr,annot=True)
        plt.title(title+"\n",loc='center',fontsize=24)
