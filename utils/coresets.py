
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
        if base_features.shape[0] == 0:
            self.dists = np.ones((self.n_obs*self.n_device,1))*np.inf
        else:
            dists_w_all = pairwise_distances(np.concatenate(self.features, axis=0), base_features, metric=self.metric)
            self.dists = np.min(dists_w_all, axis=1).reshape(-1,1)

    def distributed_coreset(self):
        """
        Returns a coreset of size n_cache x n_device samples
        
        """

        # Initialize set of caches

        cache_inds = []

        dists = np.zeros((self.n_obs,1))

        for i in range(self.n_device):

            dists[:] = self.dists[i*(self.n_obs):(i+1)*(self.n_obs)]

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
        
        dists = np.zeros((self.n_obs*self.n_device,1))
        dists[:] = self.dists

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

                    if ind < 0:
                        ind = 0
                        while self.inds[j][ind] in cache_inds:
                            ind += 1

                    cache_inds.append(self.inds[j][ind])

                    dists = np.minimum(dists, pairwise_distances(features, self.features[j][ind].reshape(1,-1), metric=self.metric))  

                    ind_centers[j][k] = self.features[j][ind]
                
                mask[j*(self.n_obs):(j+1)*(self.n_obs)] = 0
        
        return cache_inds
    
    def iterative_new(self):
        "Returns a coreset of size n_cache x n_device samples"

        # Initialize set of caches from scratch

        cache_inds = []

        dists = np.zeros((self.n_obs*self.n_device,1))
        dists[:] = self.dists
        mask = np.zeros((self.n_obs*self.n_device,1))

        features = np.concatenate(self.features, axis=0)

        for i in range(self.n_device):
            mask[i*(self.n_obs):(i+1)*(self.n_obs)] = 1

            for j in range(self.n_cache):
                  
                ind = np.argmax(dists*mask) - i*self.n_obs

                if ind < 0:
                    ind = 0
                    while self.inds[i][ind] in cache_inds:
                        ind += 1
                
                cache_inds.append(self.inds[i][ind])
  
                dists = np.minimum(dists, pairwise_distances(features, self.features[i][ind].reshape(1,-1), metric=self.metric))
                
            mask[i*(self.n_obs):(i+1)*(self.n_obs)] = 0

        return cache_inds
        
    def sample_caches(self, method='Distributed'):
        
        if method == 'Distributed':
            return self.distributed_coreset()
        elif method == 'Oracle':
            return self.oracle_greedy()
        elif method == 'Interactive':
            return self.iterative_distributed()
        elif method == "Interactive-New":
            return self.iterative_new()
        else:
            raise ValueError('Method not supported')
          
class FacilityLocation:

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

        self.M = np.zeros((self.n_obs*self.n_device,self.n_obs*self.n_device))
        dists = pairwise_distances(np.concatenate(self.features, axis=0), metric=self.metric)
        self.M = self.distance_function(dists)

        if base_features.shape[0] != 0:
            dists = pairwise_distances(np.concatenate(self.features, axis=0), base_features, metric=self.metric)
            m = self.distance_function(dists)
            self.max_M = np.max(m,axis=1).reshape(-1,1)
        else:
            self.max_M = np.zeros((self.n_obs*self.n_device,1))

    def distance_function(self, x):
        return 1/(1+0.01*x)

    def distributed(self):
        """
        Returns a coreset of size n_cache x n_device samples
        
        """

        # Initialize set of caches

        cache_inds = []

        max_M = np.zeros((self.n_obs,1))

        for i in range(self.n_device):

            max_M[:] = self.max_M[i*(self.n_obs):(i+1)*(self.n_obs)]
            Ms = self.M[i*(self.n_obs):(i+1)*(self.n_obs),i*(self.n_obs):(i+1)*(self.n_obs)]

            cache_ind = []

            for j in range(self.n_cache):
                  
                ind = np.argmax(np.maximum(max_M,Ms).sum(axis=0))

                cache_ind.append(self.inds[i][ind])

                max_M = np.maximum(max_M,Ms[:,ind].reshape(-1,1))
                Ms[:,ind] = 0
                Ms[ind,:] = 0
                  
            cache_inds.extend(cache_ind)
    
        return cache_inds

    def distributed_new(self):

        cache_inds = []

        max_M = np.zeros((self.n_obs*self.n_device,1))

        for i in range(self.n_device):

            max_M[:] = self.max_M
            Ms = self.M[:,i*(self.n_obs):(i+1)*(self.n_obs)]

            cache_ind = []

            for j in range(self.n_cache):
                  
                ind = np.argmax(np.maximum(max_M,Ms).sum(axis=0))

                cache_ind.append(self.inds[i][ind])

                max_M = np.maximum(max_M,Ms[:,ind].reshape(-1,1))
                Ms[:,ind] = 0
                Ms[ind,:] = 0
                  
            cache_inds.extend(cache_ind)
    
        return cache_inds
 
    def centralized(self):

        n_caches = [0 for i in range(self.n_device)]

        mask = np.ones((1,self.n_obs*self.n_device))

        cache_inds = []

        #inds = [ind for ind in self.inds[0]]

        #for i in range(1, self.n_device):
        #    inds.extend(self.inds[i])

        inds = [ind for ind in self.inds[0]]
        for i in range(1, self.n_device):
            inds.extend(self.inds[i])

        max_M = self.max_M

        for j in range(self.n_cache * self.n_device):
            
            ind = np.argmax(np.maximum(max_M,self.M*mask).sum(axis=0))

            cache_inds.append(inds[ind])

            device_i = ind // self.n_obs

            n_caches[device_i] += 1

            if n_caches[device_i] == self.n_cache:
                mask[0,device_i*(self.n_obs):(device_i+1)*(self.n_obs)] = 0

            max_M = np.maximum(max_M,self.M[:,ind].reshape(-1,1))
        
        return cache_inds

    def centralized_new(self):

        n_caches = [0 for i in range(self.n_device)]

        cache_inds = []

        ind_map = np.arange(self.n_device)

        max_M = self.max_M

        for j in range(self.n_cache * self.n_device):
            
            ind = np.argmax(np.maximum(max_M,self.M).sum(axis=0))

            device_i = ind // self.n_obs

            cache_inds.append(self.inds[ind_map[device_i]][ind%self.n_obs])

            n_caches[ind_map[device_i]] += 1

            max_M = np.maximum(max_M,self.M[:,ind].reshape(-1,1))

            if n_caches[ind_map[device_i]] == self.n_cache:
                self.M = np.delete(self.M,slice(device_i*(self.n_obs),(device_i+1)*(self.n_obs)),axis=1)
                ind_map = np.delete(ind_map,device_i)
        
        return cache_inds

    def iterative(self):
        "Returns a coreset of size n_cache x n_device samples"

        # Initialize set of caches from scratch

        cache_inds = []
        max_M  = self.max_M
        for i in range(self.n_device):
            M = self.M[:,i*(self.n_obs):(i+1)*(self.n_obs)]

            for j in range(self.n_cache):
                  
                ind = np.argmax(np.maximum(max_M,M).sum(axis=0))

                if ind < 0:
                    ind = 0
                    while self.inds[i][ind] in cache_inds:
                        ind += 1
                
                cache_inds.append(self.inds[i][ind])
  
                max_M = np.maximum(max_M,M[:,ind].reshape(-1,1))
                

        return cache_inds
  
    def sample_caches(self, method='Distributed'):
        
        if method == 'Distributed':
            return self.distributed()
        elif method == 'Distributed-New':
            return self.distributed_new()
        elif method == 'Oracle':
            return self.centralized()
        elif method == 'Oracle-New':
            return self.centralized_new()
        elif method == 'Interactive':
            return self.iterative()
        elif method == 'Interactive-New':
            return self.iterative()
        else:
            raise ValueError('Method not supported')
       

        

        








        


