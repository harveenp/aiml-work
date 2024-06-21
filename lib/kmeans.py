import numpy as np
      
def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    m,n=X.shape    
    for elem in range(m):
        distance=[]
        for cents in range(K):    
            d = np.linalg.norm(X[elem] - centroids[cents])
            distance.append(d)
        idx[elem] = distance.index(min(distance))
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    newCentroids = np.zeros((K, n))
    countAry = np.zeros(K)
    for elem in range(m):
        newCentroids[idx[elem]] = np.add(newCentroids[idx[elem]] , X[elem])
        countAry[idx[elem]] = countAry[idx[elem]] + 1
    for cents in range(K):
        newCentroids[cents] = np.divide(newCentroids[cents],countAry[cents])
    return newCentroids

def run_kMeans(X, initial_centroids, max_iters=10):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids   
    idx = np.zeros(m)
    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids