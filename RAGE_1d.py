### Adapted from https://github.com/fiezt/Transductive-Linear-Bandit-Code/blob/master/RAGE.py

import numpy as np
import itertools
import logging
import time
from tqdm import tqdm


class RAGE(object):
    def __init__(self, X, theta_star, factor, delta, Z=None):
        
        self.X = X
        if Z is None:
            self.Z = X
        else:
            self.Z = Z
        self.K = len(X)
        self.K_Z = len(self.Z)
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.opt_arm = np.argmax(self.Z@theta_star)
        self.delta = delta
        self.factor = factor
        
    @profile     
    def algorithm(self, seed, var=True, binary=False):
        
        self.var=var
        self.seed = seed
        np.random.seed(self.seed)

        self.active_arms = list(range(len(self.Z)))
        self.arm_counts = np.zeros(self.K)
        self.N = 0
        self.phase_index = 1
                
        while len(self.active_arms) > 1:
            # print(f"Starting new iteration with {len(self.active_arms)} active arms.")
            
            self.delta_t = self.delta/(self.phase_index**2)
            # print(f"Updated delta_t to {self.delta_t} for phase {self.phase_index}.")
            
            self.build_Y()
            # print(f"Built y\n")
            design, rho = self.optimal_allocation()
            # print(f"Optimal allocation design: {design}\n")
            # print(f"Rho value: {rho}\n")
            
            support = np.sum((design > 0).astype(int))
            n_min = 2*self.factor*support
            eps = 1/self.factor
            
            num_samples = max(np.ceil(8*(2**(self.phase_index-1))**2*rho*(1+eps)*np.log(2*self.K_Z**2/self.delta_t)), n_min).astype(int)
            allocation = self.rounding(design, num_samples)
            # print(f"Number of samples required this phase: {num_samples}")
            # print(f"Allocation per arm: {allocation}")
            
            pulls = np.vstack([np.tile(self.X[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])
            
            if not binary:
                rewards = pulls@self.theta_star + np.random.randn(allocation.sum(), 1)
            else:
                rewards = np.random.binomial(1, pulls@self.theta_star, (allocation.sum(), 1))
            
            self.A_inv = np.linalg.pinv(pulls.T@pulls)
            self.theta_hat = self.A_inv@pulls.T@rewards
            # print(f"Updated theta_hat: {self.theta_hat.flatten()}")
            
            self.drop_arms()
            self.phase_index += 1
            self.arm_counts += allocation
            self.N += num_samples
            
            # print(f"Finished phase {self.phase_index-1}")
            # print(f"Current total sample count: {self.N}")
            # print(f"Remaining active arms: {len(self.active_arms)}")
            # print("\n\n")
            
            logging.info('\n\n')
            logging.info('finished phase %s' % str(self.phase_index-1))
            logging.info('design %s' % str(design))
            logging.debug('allocation %s' % str(allocation))
            logging.debug('arm counts %s' % str(self.arm_counts))
            logging.info('round sample count %s' % str(num_samples))
            logging.info('total sample count %s' % str(self.N))
            logging.info('active arms %s' % str(self.active_arms)) 
            logging.info('rho %s' % str(rho))      
            logging.info('\n\n')

        del self.Yhat
        del self.idxs
        del self.X
        del self.Z
        self.success = (self.opt_arm in self.active_arms)
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))
            
    # @profile  
    # def build_Y(self):
        
    #     k = len(self.active_arms)
    #     # print(f"Inside build_Y, number of active arms: {k} \n")
    #     idxs = np.zeros((k*k,2))
    #     Zhat = self.Z[self.active_arms]
    #     Y = np.zeros((k*k, self.d))
    #     rangeidx = np.array(list(range(k)))
        
    #     for i in tqdm(range(k)):
    #         idxs[k*i:k*(i+1),0] = rangeidx
    #         idxs[k*i:k*(i+1),1] = i
    #         Y[k*i:k*(i+1),:] = Zhat - Zhat[i,:] 
        
    #     self.Yhat = Y
    #     self.idxs = idxs

    @profile  
    def build_Y(self):
        
        k = len(self.active_arms)
        # print(f"Inside build_Y, number of active arms: {k} \n")
        idxs = np.zeros((k,2))
        Zhat = self.Z[self.active_arms]
        Y = np.zeros((k, self.d))
        rangeidx = np.array(list(range(k)))
        
        for i in tqdm(range(k)):
            idxs[i:i+1,0] = 0
            idxs[i:i+1,1] = i
            Y[i:i+1,:] = Zhat[i, :] - Zhat[0,:] 
        
        self.Yhat = Y
        self.idxs = idxs
        
    @profile
    def optimal_allocation(self):
        
        design = np.ones(self.K)
        design /= design.sum()  
        
        # max_iter = 5000
        max_iter = 5000
        
        d = self.X.shape[1]
        A_inv = np.zeros((d, d))
        U = np.zeros((d, d))
        D = np.zeros(d)
        V = np.zeros((d, d))
        Ainvhalf = np.zeros((d, d))
        newY = np.zeros((self.Yhat.shape[0], d))
        rho = np.zeros((self.Yhat.shape[0], 1))
        
        for count in tqdm(range(1, max_iter)):
            A_inv[:, :] = np.linalg.pinv(self.X.T@np.diag(design)@self.X)
            # print(f"Inside optimal_allocation, A_inv shape: {A_inv.shape}")   
            U[:, :],D[:],V[:, :] = np.linalg.svd(A_inv)
            Ainvhalf[:, :] = U@np.diag(np.sqrt(D))@V.T
            # print(f"Inside optimal_allocation, Ainvhalf shape: {Ainvhalf.shape}")  
            
            newY[:, :] = (self.Yhat@Ainvhalf)**2
            # print(f"Inside optimal_allocation, newY shape: {newY.shape}") 
            rho[:, :] = newY@np.ones((newY.shape[1], 1))
            # print(f"Inside optimal_allocation, rho shape: {rho.shape}") 
                        
            idx = np.argmax(rho)
            y = self.Yhat[idx, :, None]
            g = ((self.X@A_inv@y)*(self.X@A_inv@y)).flatten()
            g_idx = np.argmax(g)
                        
            gamma = 2/(count+2)
            design_update = -gamma*design
            design_update[g_idx] += gamma
                
            relative = np.linalg.norm(design_update)/(np.linalg.norm(design))
                        
            design += design_update
            
            if count % 100 == 0:
                logging.debug('design status %s, %s, %s, %s' % (self.seed, count, relative, np.max(rho)))
            
            # # Clear large temporary variables
            # del A_inv, U, D, V, Ainvhalf, newY, y, g                
            if relative < 0.01:
                 break
                        
        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0
        
        return design, np.max(rho)
    
    @profile             
    def rounding(self, design, num_samples):
        
        num_support = (design > 0).sum()
        support_idx = np.where(design>0)[0]
        support = design[support_idx]
        n_round = np.ceil((num_samples - .5*num_support)*support)

        while n_round.sum()-num_samples != 0:
            if n_round.sum() < num_samples:
                idx = np.argmin(n_round/support)
                n_round[idx] += 1
            else:
                idx = np.argmax((n_round-1)/support)
                n_round[idx] -= 1

        allocation = np.zeros(len(design))
        allocation[support_idx] = n_round
            
        return allocation.astype(int)
      
    @profile      
    def drop_arms(self):
                
            
        if not self.var:
            active_arms = self.active_arms.copy()
            removes = set()
            scores = self.Yhat@self.theta_hat
            # gap = 2**(-(self.phase_index+2))
            gap = 2**(-(self.phase_index))

            for t,s in tqdm(enumerate(scores)):
                if gap <= s[0]:
                    arm_idx = int(self.idxs[t][1])
                    removes.add(self.active_arms[arm_idx])

            for r in tqdm(removes):
                self.active_arms.remove(r)
            
        else:
            active_arms = self.active_arms.copy()

            for arm_idx in tqdm(active_arms):

                arm = self.Z[arm_idx, :, None]

                for arm_idx_prime in tqdm(active_arms):

                    if arm_idx == arm_idx_prime:
                        continue

                    arm_prime = self.Z[arm_idx_prime, :, None]
                    y = arm_prime - arm

                    if np.sqrt(2*y.T@self.A_inv@y*np.log(2*self.K**2/self.delta_t)) <= y.T@self.theta_hat:
                        self.active_arms.remove(arm_idx)
                        break
     