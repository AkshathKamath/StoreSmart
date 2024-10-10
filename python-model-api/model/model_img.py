import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.cluster import KMeans

def Kmeans_model(df):
    A = np.asarray(df['Total amount with Tax'])
    B = np.asarray(df['Quantity'])
    df['Rp'] = A/B
    df['Cp'] = df['Unit price']
    df = df[['Cp','Rp']]

    for i in range(0,1000):
        df.iloc[i,0] = df.iloc[i,0]*np.random.rand()
        df.iloc[i,1] = df.iloc[i,1]*np.random.rand()

    X=df.iloc[:,:].values
    kmeans = KMeans(n_clusters=3,init='k-means++',random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    fig = plt.figure(figsize=(4,6))
    plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=10, c='red')
    plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=10, c='blue')
    plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s=10, c='green')
    plt.legend(['Higher margin items','Low Margin-Low Cost Items','Low Margin-High Cost Items'])
    plt.xlabel('Cost Price')
    plt.ylabel('Retail Price')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf