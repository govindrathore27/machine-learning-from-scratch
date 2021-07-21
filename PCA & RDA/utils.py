import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as s
def flatten(img_path):
    '''
    Flattens an image and reshapes into (1,1024) array.
    Input:
    img_path-> image path containing images to be flattened.train_image_path.
    Output:
    list_images-> list of flattened images converted into array of shape (1,1024)
    '''
    list_images = list(map(lambda x : plt.imread(x).reshape(1,1024),img_path))

    return list_images



def post_prob(class_means,class_cov,reduced_df):
    '''
    Inputs:
    class_means-> Mean vector for each class.
    class_cov -> Covariance Matrix for each class.
    reduced_df  -> Test dataset reduced by PCA and RDA.
    Outputs:
    prob_on_class_x -> Posterior probability for test data to compute classes.
    '''
    
    prob_on_class_x = s.multivariate_normal.pdf(reduced_df,class_means,class_cov)
    
    prob_on_class_x = prob_on_class_x.reshape(prob_on_class_x.shape[0],1)
    
    return prob_on_class_x