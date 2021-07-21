import numpy as np
from numpy import random

def RDA(class_cov,train_df,train_folder,unique_labels):
    '''
    RDA : Regularised Discriminant Analysis is a disriminant analysis technique which works by combining the ideas of LDA and QDA.
    Basically a regularised version of both the models.In this we use both pooled covariance(LDA) as well as individual covariance(QDA).
    Inputs:
    class_cov -> Individual Covariance matrices for each class.
    train_df -> Train dataset.
    train_folder -> Number of classes.
    unique_labels-> An array containing all the unique labels.
    Outputs:
    RDA_cov -> A regularised covariance matrix. 
    '''
    np.random.seed(42)
    alpha = np.random.uniform(0,1)
    gamma = np.random.uniform(0,1)
    QDA_cov = np.array(class_cov)
    LDA_cov = (1700-1) * sum(class_cov)/(train_df.shape[0] - len(train_folder))
    # LDA_cov = train_df.iloc[:,:1024].cov()
    # print(LDA_cov.shape)
    QDA_cov_hat = []
    QDA_cov_hat.extend(map(lambda x : (alpha * LDA_cov) + (1 - alpha) * x , QDA_cov))
    QDA_cov_hat = np.array(QDA_cov_hat)
    # D  = (1/len(train_folder)) * np.trace(QDA_cov_hat)
    D = []
    D.extend(map(lambda x : np.mean(np.diag(x)),QDA_cov_hat))
    D = np.array(D)

    RDA_cov = []
    RDA_cov.extend(map(lambda x : gamma * QDA_cov_hat[x] + (1-gamma) * D[x] * np.identity(QDA_cov_hat[0].shape[0]) ,unique_labels))
    RDA_cov = np.array(RDA_cov)
    return RDA_cov