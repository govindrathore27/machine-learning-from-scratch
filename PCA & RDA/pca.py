import numpy as np
def PCA(cov,preserved_variance=0.97):
    '''
    PCA : Principle Component Analysis is a dimensionality reduction technique which works by proejcting the data in 
    the direction of eigen vectors with highest eigen values obtained by Singular Value Decomposition(SVD).
    Inputs:
    cov-> Covariance matrix on which dimensionality reduction is to be done.
    preserved_variance -> threshold for selection of features.
    Outputs:
    Selected features as an array.
    '''
    
    svd_factorized_mat = np.linalg.svd(cov)
    
    eig_vals = svd_factorized_mat[1]
    
    eig_vecs = svd_factorized_mat[0]
    
    eig_vals_total = np.sum(eig_vals)
    
    eig_vals_sum = 0
    
    indices = []
    
    for i in range(0,eig_vals.shape[0]):
        
        if eig_vals_sum/eig_vals_total > preserved_variance:
            
            break
            
        eig_vals_sum += eig_vals[i]
        
        indices.append(i)
        
    return eig_vecs[:,indices]