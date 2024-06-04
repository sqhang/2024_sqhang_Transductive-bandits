## Modified from: https://github.com/fiezt/Transductive-Linear-Bandit-Code/blob/master/RAGE.py
import numpy as np
import itertools
import logging
import time


class RAGE_opt(object):
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
        self.theta_hat_list = []
        
        
    def algorithm(self, seed, var=True, binary=False, sigma=1, stop_arm_count=1, rel_thresh=0.01):
        
        self.var=var
        self.seed = seed
        self.sigma = sigma
        self.stop_arm_count = stop_arm_count
        self.rel_thresh=rel_thresh        
        np.random.seed(self.seed)

        self.active_arms = list(range(len(self.Z)))
        self.arm_counts = np.zeros(self.K)
        self.N = 0
        self.phase_index = 1
        self.theta_hat = None
                
        while len(self.active_arms) > self.stop_arm_count:
                        
            self.delta_t = self.delta/(self.phase_index**2)      
            
            self.build_Y()
            design, rho = self.optimal_allocation()
            support = np.sum((design > 0).astype(int))
            n_min = 2*self.factor*support
            eps = 1/self.factor
            
            # num_samples = max(np.ceil(8*(2**(self.phase_index-1))**2*rho*(1+eps)*np.log(2*self.K_Z**2/self.delta_t)), n_min).astype(int)
            num_samples = max(np.ceil(8*(2**(self.phase_index+1))**2*rho*(1+eps)*np.log(2*self.K_Z**2/self.delta_t) * (self.sigma ** 2)), n_min).astype(int)             
            allocation = self.rounding(design, num_samples)
            
            pulls = np.vstack([np.tile(self.X[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])
            
            if not binary:
                rewards = pulls@self.theta_star + np.random.randn(allocation.sum(), 1) * self.sigma
            else:
                rewards = np.random.binomial(1, pulls@self.theta_star, (allocation.sum(), 1))
            
            self.A_inv = np.linalg.pinv(pulls.T@pulls)
            self.theta_hat = self.A_inv @ pulls.T @ rewards
            # print(f"theta_hat shape: {self.theta_hat.shape}")
            
            self.reference_update()        
            self.drop_arms()
            self.phase_index += 1
            self.arm_counts += allocation
            self.N += num_samples
            self.theta_hat_list.append((self.N, self.theta_hat, len(self.active_arms)))
            
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
        del self.X
        del self.Z
        self.success = (self.opt_arm in self.active_arms)
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))
            
    
    def build_Y(self):
        if self.theta_hat is None:
            # Randomly select a reference point if theta_hat is not initialized
            reference_idx = np.random.choice(self.active_arms)
            # print(f"Random best arm chosed, {reference_idx}\n")
        else:
            active_Z = self.Z[self.active_arms]
            reference_idx = np.argmax(active_Z @ self.theta_hat)
        reference = self.Z[reference_idx]
        Y = self.Z[self.active_arms] - reference
        self.Yhat = Y
        
    def reference_update(self):
        if self.theta_hat is not None:
            active_Z = self.Z[self.active_arms]
            # print(f"active_Z.shape: {active_Z.shape}\n")
            # print(f"theta_hat.shape: {self.theta_hat.shape}")
            # print(f"opt_index: {np.argmax(active_Z @ self.theta_hat)}\n")
            
            self.opt_arm = self.active_arms[np.argmax(active_Z @ self.theta_hat)]
        
    
    def optimal_allocation(self):
        
        design = np.ones(self.K)
        design /= design.sum()  
        
        max_iter = 5000
        
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.X.T@np.diag(design)@self.X)    
            U,D,V = np.linalg.svd(A_inv)
            Ainvhalf = U@np.diag(np.sqrt(D))@V.T
            
            newY = (self.Yhat@Ainvhalf)**2
            rho = newY@np.ones((newY.shape[1], 1))
                        
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
                            
            if relative < self.rel_thresh:
                 break
                        
        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0
        
        return design, np.max(rho)
    
                
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

    def drop_arms(self):
        if self.theta_hat is None:
            return

        presumed_best_arm = self.Z[self.opt_arm, :, None]

        active_arms_matrix = self.Z[self.active_arms, :]

        y = presumed_best_arm.T - active_arms_matrix
        projections = y @ self.theta_hat.flatten()

        if not self.var:
            thresholds = 2 ** (-self.phase_index - 2)
        else:
            quadratic_forms = np.einsum('ij,jk,ik->i', y, self.A_inv, y)
            thresholds = np.sqrt(2 * (self.sigma**2) * np.log(2 * self.K**2 / self.delta_t) * quadratic_forms)
        
        removes = np.array(self.active_arms)[projections >= thresholds]
        self.active_arms = [arm for arm in self.active_arms if arm not in removes]


