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