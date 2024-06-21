import numpy as np

def estimate_gaussian(X): 
    m, n = X.shape
    mu=X.sum(axis=0)/m
    sum=np.zeros(n)
    for elem in range(m):
        sum=sum + (X[elem] - mu)**2
    var=sum/m
    return mu, var

def multivariate_gaussian(X, mu, var):
    """
    Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """
    
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p

def select_threshold(p_val,y_val):
    bestF1=0.0
    bestEpsilon=0.0
    step_size=(max(p_val) - min(p_val))/1000
    for epsilon in np.arange(min(p_val),max(p_val),step_size):
        predictions=np.where(p_val <= epsilon,1,0)
        tp=np.sum(predictions[y_val==1])
        fp=np.sum(predictions[y_val==0])
        fn=np.sum(y_val[predictions==0])
        precision=tp/(tp + fp)
        recall=tp/(tp+fn)
        F1=(2*precision*recall)/(precision + recall)
        if(F1>bestF1):
            bestF1=F1
            bestEpsilon=epsilon
    return bestEpsilon,bestF1