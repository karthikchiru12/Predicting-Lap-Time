import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


def clean_data(dataframe_to_clean, filename=""):
    """
        Cleans the given dataframe and stores the cleaned data in a new folder
        #############################
        input:
            dataframe_to_clean : the dataframe to clean    
            filename           : the cleaned dataframe is saved using filename
    """

    if not os.path.isfile('Cleaned_data/'+filename):

        dataframe = dataframe_to_clean

        dataframe[' CROSSING_FINISH_LINE_IN_PIT'].fillna(
            'Not Available',inplace=True)
        print("CROSSING_FINISH_LINE_IN_PIT : FIXED")

        dataframe[' S1'] = dataframe[' S1'].fillna("00:00.00")
        for i in range(len(dataframe)):
            if ":" not in str(dataframe.iloc[i, 6]).strip():
                dataframe.iloc[i, 6] = "00:"+str(dataframe.iloc[i, 6])
        dataframe['S1 minutes'] = [str(dataframe.iloc[i, 6]).split(':')[
            0] for i in range(len(dataframe))]
        dataframe['S1 minutes'] = pd.to_numeric(dataframe['S1 minutes'])
        dataframe['S1 seconds'] = [str(dataframe.iloc[i, 6]).split(':')[
            1] for i in range(len(dataframe))]
        dataframe['S1 seconds'] = pd.to_numeric(dataframe['S1 seconds'])
        print("S1                          : FIXED")

        dataframe[' S2'] = dataframe[' S2'].fillna("00:00.00")
        for i in range(len(dataframe)):
            if ":" not in str(dataframe.iloc[i, 8]).strip():
                dataframe.iloc[i, 8] = "00:"+str(dataframe.iloc[i, 8])
        dataframe['S2 minutes'] = [str(dataframe.iloc[i, 8]).split(':')[
            0] for i in range(len(dataframe))]
        dataframe['S2 minutes'] = pd.to_numeric(dataframe['S2 minutes'])
        dataframe['S2 seconds'] = [str(dataframe.iloc[i, 8]).split(':')[
            1] for i in range(len(dataframe))]
        dataframe['S2 seconds'] = pd.to_numeric(dataframe['S2 seconds'])
        print("S2                          : FIXED")

        dataframe[' S3'] = dataframe[' S3'].fillna("00:00.00")
        for i in range(len(dataframe)):
            if ":" not in str(dataframe.iloc[i, 10]).strip():
                dataframe.iloc[i, 10] = "00:"+str(dataframe.iloc[i, 10])
        dataframe['S3 minutes'] = [str(dataframe.iloc[i, 10]).split(':')[
            0] for i in range(len(dataframe))]
        dataframe['S3 minutes'] = pd.to_numeric(dataframe['S3 minutes'])
        dataframe['S3 seconds'] = [str(dataframe.iloc[i, 10]).split(':')[
            1] for i in range(len(dataframe))]
        dataframe['S3 seconds'] = pd.to_numeric(dataframe['S3 seconds'])
        print("S3                          : FIXED")

        dataframe[' KPH'] = dataframe[' KPH'].fillna(0.0)
        print("KPH                         : FIXED")

        for i in range(len(dataframe)):
            if ":" not in str(dataframe.iloc[i, 13]).strip():
                dataframe.iloc[i, 13] = "00:"+str(dataframe.iloc[i, 13])
        dataframe['ELAPSED minutes'] = [str(dataframe.iloc[i, 13]).split(':')[
            0] for i in range(len(dataframe))]
        dataframe['ELAPSED minutes'] = pd.to_numeric(
            dataframe['ELAPSED minutes'])
        dataframe['ELAPSED seconds'] = [str(dataframe.iloc[i, 13]).split(':')[
            1] for i in range(len(dataframe))]
        dataframe['ELAPSED seconds'] = pd.to_numeric(
            dataframe['ELAPSED seconds'])
        print("ELAPSED                     : FIXED")

        dataframe['minutes'] = [str(dataframe.iloc[i, 14]).split(':')[
            0] for i in range(len(dataframe))]
        dataframe['minutes'] = pd.to_numeric(dataframe['minutes'])
        dataframe['seconds'] = [str(dataframe.iloc[i, 14]).split(':')[
            1] for i in range(len(dataframe))]
        dataframe['seconds'] = pd.to_numeric(dataframe['seconds'])
        print("HOUR                        : FIXED")

        dataframe['S1_LARGE'] = dataframe['S1_LARGE'].fillna("00:00.00")
        for i in range(len(dataframe)):
            if ":" not in str(dataframe.iloc[i, 15]).strip():
                dataframe.iloc[i, 15] = "00:"+str(dataframe.iloc[i, 15])
        dataframe['S1 Large minutes'] = [str(dataframe.iloc[i, 15]).split(':')[
            0] for i in range(len(dataframe))]
        dataframe['S1 Large minutes'] = pd.to_numeric(
            dataframe['S1 Large minutes'])
        dataframe['S1 Large seconds'] = [str(dataframe.iloc[i, 15]).split(':')[
            1] for i in range(len(dataframe))]
        dataframe['S1 Large seconds'] = pd.to_numeric(
            dataframe['S1 Large seconds'])
        print("S1 Large                    : FIXED")

        dataframe['S2_LARGE'] = dataframe['S2_LARGE'].fillna("00:00.00")
        for i in range(len(dataframe)):
            if ":" not in str(dataframe.iloc[i, 16]).strip():
                dataframe.iloc[i, 16] = "00:"+str(dataframe.iloc[i, 16])
        dataframe['S2 Large minutes'] = [str(dataframe.iloc[i, 16]).split(':')[
            0] for i in range(len(dataframe))]
        dataframe['S2 Large minutes'] = pd.to_numeric(
            dataframe['S2 Large minutes'])
        dataframe['S2 Large seconds'] = [str(dataframe.iloc[i, 16]).split(':')[
            1] for i in range(len(dataframe))]
        dataframe['S2 Large seconds'] = pd.to_numeric(
            dataframe['S2 Large seconds'])
        print("S2 Large                    : FIXED")

        dataframe['S3_LARGE'] = dataframe['S3_LARGE'].fillna("00:00.00")
        for i in range(len(dataframe)):
            if ":" not in str(dataframe.iloc[i, 17]).strip():
                dataframe.iloc[i, 17] = "00:"+str(dataframe.iloc[i, 17])
        dataframe['S3 Large minutes'] = [str(dataframe.iloc[i, 17]).split(':')[
            0] for i in range(len(dataframe))]
        dataframe['S3 Large minutes'] = pd.to_numeric(
            dataframe['S3 Large minutes'])
        dataframe['S3 Large seconds'] = [str(dataframe.iloc[i, 17]).split(':')[
            1] for i in range(len(dataframe))]
        dataframe['S3 Large seconds'] = pd.to_numeric(
            dataframe['S3 Large seconds'])
        print("S3 Large                    : FIXED")

        dataframe['PIT_TIME'] = dataframe['PIT_TIME'].fillna("00:00.00")
        for i in range(len(dataframe)):
            if ":" not in str(dataframe.iloc[i, 19]).strip():
                dataframe.iloc[i, 19] = "00:"+str(dataframe.iloc[i, 19])
        dataframe['PIT_TIME Large minutes'] = [str(dataframe.iloc[i, 19]).split(':')[
            0] for i in range(len(dataframe))]
        dataframe['PIT_TIME Large minutes'] = pd.to_numeric(
            dataframe['PIT_TIME Large minutes'])
        dataframe['PIT_TIME Large seconds'] = [str(dataframe.iloc[i, 19]).split(':')[
            1] for i in range(len(dataframe))]
        dataframe['PIT_TIME Large seconds'] = pd.to_numeric(
            dataframe['PIT_TIME Large seconds'])
        print("PIT_TIME                    : FIXED")

        dataframe['GROUP'] = dataframe['GROUP'].fillna(0.0)
        print("GROUP                       : FIXED")

        dataframe['POWER'] = dataframe['POWER'].fillna(
            dataframe['POWER'].mean())
        print("POWER                       : FIXED")
        print("\nDONE : If the code works, dont touch it.\n")

        dataframe.to_csv('Cleaned_data/'+filename, index=False)

    else:
        print(filename+"already exists")


def clean_weather_data(dataframe_to_clean, filename=""):
    """
        Cleans the given dataframe and stores the cleaned data in a new folder
        #############################
        input:
            dataframe_to_clean : the dataframe to clean      
            filename           : the cleaned dataframe is saved using filename
    """
    if not os.path.isfile('Cleaned_data/'+filename):

        dataframe = dataframe_to_clean

        dataframe['TIME_UTC_STR'] = pd.to_datetime(dataframe['TIME_UTC_STR'])

        dataframe['AIR_TEMP'] = [str(dataframe.iloc[i, 2]).split(
            ',')[0] for i in range(len(dataframe['AIR_TEMP']))]
        dataframe['AIR_TEMP'] = pd.to_numeric(dataframe['AIR_TEMP'])

        dataframe['TRACK_TEMP'] = [str(dataframe.iloc[i, 3]).split(
            ',')[0] for i in range(len(dataframe['TRACK_TEMP']))]
        dataframe['TRACK_TEMP'] = pd.to_numeric(dataframe['TRACK_TEMP'])

        dataframe['HUMIDITY'] = [str(dataframe.iloc[i, 4]).split(
            ',')[0] for i in range(len(dataframe['HUMIDITY']))]
        dataframe['HUMIDITY'] = pd.to_numeric(dataframe['HUMIDITY'])

        dataframe['PRESSURE'] = [str(dataframe.iloc[i, 5]).split(
            ',')[0] for i in range(len(dataframe['PRESSURE']))]
        dataframe['PRESSURE'] = pd.to_numeric(dataframe['PRESSURE'])

        dataframe['WIND_SPEED'] = [str(dataframe.iloc[i, 6]).split(
            ',')[0] for i in range(len(dataframe['WIND_SPEED']))]
        dataframe['WIND_SPEED'] = pd.to_numeric(dataframe['WIND_SPEED'])

        dataframe['RAIN'] = [str(dataframe.iloc[i, 8]).replace(
            ',', '.') for i in range(len(dataframe['RAIN']))]
        dataframe['RAIN'] = pd.to_numeric(dataframe['RAIN'])
        print("\nDONE : If the code works, dont touch it.\n")

        dataframe.to_csv('Cleaned_data/'+filename, index=False)

    else:
        print(filename+"already exists")


def plot_missing_values_per_feature(dataframe, width=8, height=8, title=""):
    """
        Plots missing values in the given dataframe
        #############################
        input:
            dataframe : the dataframe to plot missing values for
            width     : width of each plot
            height    : height of each plot
            title     : Title of the plot          
    """
    plt.figure(figsize=(width, height))
    sns.barplot(y=list(dataframe.isnull().sum().index),
                x=list(dataframe.isnull().sum().values))
    plt.xlabel("Missing value counts")
    plt.ylabel("Features")
    plt.title("Missing values in {0}\n".format(title), fontsize=20)


def plot_dataset_description_with_target_distribution(dataframe, target, title=""):
    """
        Plots a distribution of target along with the description of numerical features
        #############################
        input:
            dataframe : the dataframe to plot the description for
            target    : target variable
            title     : Title of the plot
    """
    plt.title(title+" "+str(dataframe.shape)+"\n", fontsize=18)
    plot_ax = dataframe[target].plot(kind='kde', legend=True)
    plot_ax.set_xlabel("Target feature"+" : "+target)
    data_desc = dataframe.describe().round(2).T
    table_object = pd.plotting.table(plot_ax, data_desc,
                                     loc='top',
                                     cellLoc='center',
                                     rowColours=["skyblue"]*len(data_desc),
                                     colColours=["skyblue"]*len(data_desc),
                                     bbox=[0.0, -2.50, 1.4, 2]).auto_set_font_size(False)
