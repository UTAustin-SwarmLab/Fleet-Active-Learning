
import numpy as np
from sklearn.metrics import pairwise_distances
import random
from copy import deepcopy

class kCenterGreedy:

    def __init__(self, features,base_features, obs_inds, n_iter, n_cache, metric='euclidean'):
        self.features = features
        self.base_features = base_features
        self.inds = obs_inds
        self.n_cache = n_cache
        self.metric = metric
        self.min_distances = None
        self.n_device = len(self.features)
        self.n_obs = self.features[0].shape[0]
        self.already_selected = []
        self.n_iter = n_iter
        dists_w_all = pairwise_distances(np.concatenate(self.features, axis=0), base_features, metric=self.metric)
        self.dists = np.min(dists_w_all, axis=1).reshape(-1,1)

    def distributed_coreset(self):
        """
        Returns a coreset of size n_cache x n_device samples
        
        """

        # Initialize set of caches

        cache_inds = []

        for i in range(self.n_device):

            dists = deepcopy(self.dists[i*(self.n_obs):(i+1)*(self.n_obs)])

            cache_ind = []

            for j in range(self.n_cache):
                  
                  ind = np.argmax(dists)
  
                  cache_ind.append(self.inds[i][ind])
  
                  dists = np.minimum(dists, pairwise_distances(self.features[i], self.features[i][ind].reshape(1,-1), metric=self.metric))
            
            cache_inds.extend(cache_ind)
    
        return cache_inds
    
    def oracle_greedy(self):
        
        "Returns a coreset of size n_cache x n_device samples"

        n_caches = [0 for i in range(self.n_device)]

        mask = np.ones((self.n_obs*self.n_device,1))

        cache_inds = []

        features = np.concatenate(self.features, axis=0)

        inds = [ind for ind in self.inds[0]]
        for i in range(1, self.n_device):
            inds.extend(self.inds[i])
        
        dists = deepcopy(self.dists)

        for j in range(self.n_cache * self.n_device):
            
            ind = np.argmax(dists*mask)

            cache_inds.append(inds[ind])

            device_i = ind // self.n_obs

            n_caches[device_i] += 1

            if n_caches[device_i] == self.n_cache:
                mask[device_i*(self.n_obs):(device_i+1)*(self.n_obs)] = 0

            dists = np.minimum(dists, pairwise_distances(features, features[ind].reshape(1,-1), metric=self.metric))
        
        return cache_inds
    
    def iterative_distributed(self):
        "Returns a coreset of size n_cache x n_device samples"

        # Initialize set of caches

        cache_inds = self.distributed_coreset()

        ind_centers = np.zeros((self.n_device,self.n_cache, self.features[0].shape[1]))

        features = np.concatenate(self.features, axis=0)

        mask = np.zeros((self.n_obs*self.n_device,1))

        for i in range(self.n_device):
            for j in range(self.n_cache):
                ind_centers[i][j] = self.features[i][self.inds[i].index(cache_inds[i*self.n_cache+j])]

        for i in range(self.n_iter):

            for j in range(self.n_device):
                
                mask[j*(self.n_obs):(j+1)*(self.n_obs)] = 1
                
                cache_inds = cache_inds[self.n_cache:]

                centers = np.concatenate(ind_centers[np.arange(self.n_device)!=j],axis=0)

                dists_w_all = pairwise_distances(features, centers, metric=self.metric)

                dists = np.min(dists_w_all, axis=1).reshape(-1,1)

                dists = np.minimum(dists, self.dists)

                for k in range(self.n_cache):

                    ind = np.argmax(dists*mask) - j*self.n_obs

                    cache_inds.append(self.inds[j][ind])

                    dists = np.minimum(dists, pairwise_distances(features, self.features[j][ind].reshape(1,-1), metric=self.metric))  

                    ind_centers[j][k] = self.features[j][ind]
                
                mask[j*(self.n_obs):(j+1)*(self.n_obs)] = 0
        
        return cache_inds
    
    def sample_caches(self, method='Distributed'):
        
        if method == 'Distributed':
            return self.distributed_coreset()
        elif method == 'Oracle':
            return self.oracle_greedy()
        elif method == 'Interactive':
            return self.iterative_distributed()
        else:
            raise ValueError('Method not supported')
          


        








        


