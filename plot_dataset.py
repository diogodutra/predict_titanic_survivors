import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#subplot histograms for each feature with separate colored labels of target
#feature and labels are nomenclature specific for machine learning
def hist_labels(df, target, alpha=0.5
    , color_wheel = {0: "#ee4035", 
                   1: "#7bc043", 
                   2: "#0392cf"}
                  ):

    import matplotlib.pyplot as plt
    #%matplotlib inline

    labels = df[target].unique().tolist()
    features = df.drop(columns={target}).columns.tolist()
    total_features = len(features)

    f = plt.figure(figsize=(16,4*total_features))

    for j in range(total_features):
        feature = features[j]
        for i in range(len(labels)):
            y = df.loc[df[target]==labels[i], feature]
            plt.subplot(total_features, 1, j+1)
            plt.hist(y.dropna(), color=color_wheel.get(i), alpha=alpha, label=str(labels[i]))

        plt.gca().set(ylabel='Frequency', xlabel=feature)
        plt.legend();
        
    plt.show()
        

#plot scatter matrix for each feature with separate colored labels of target
#feature and labels are nomenclature specific for machine learning
def scatter_labels(df, target, alpha=0.5
    , color_wheel = {0: "#ee4035", 
                   1: "#7bc043", 
                   2: "#0392cf"}
                  ):

    colors = df[target].map(lambda x: color_wheel.get(x))
    ax = scatter_matrix(df, color=colors, alpha=alpha, figsize=(16, 16), diagonal='hist')
    plt.show()
    

def hist_features(df, label):
    colors = ['DarkRed', 'DarkGreen']
    
    #names = ['<=50K', '>50K']
    names = df[label].unique().tolist()

    n_plots = df.shape[1]-1
    i_plot = 0
    for i in df.columns: #bug, check for axis_y existance in data
        if i!=label:
            plt.figure(i_plot)
            i_plot += 1
            #plt.subplot(n_plots, 3, i_plot)
            #plt.subplot(1, 1, 1)
            x1 = list(df[df[label] == 0][i])
            x2 = list(df[df[label] == 1][i])
            plt.hist([x1, x2], color = colors, label=names)
            plt.legend()
            plt.title(label +' by ' + i)