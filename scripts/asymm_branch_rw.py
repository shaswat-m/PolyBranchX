# --------------------------------------------------------
# branch_rw.py
# by Shaswat Mohanty, shaswatm@stanford.edu
#
# Objectives
# Utility functions for numerically simulating branching spatial processes (BRW, BMRW, BBM and GBRW)
#
# Cite: (to be added)
# --------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import argparse
import itertools
from functools import partial
from copy import deepcopy
import random
import multiprocessing as mp
'''
 Run using: python3 branch_rw.py --single_analysis  
'''

def weib(x,n,a):
    '''
    Function for Weibull Fitting
    '''
    return (a / n) * (x / n)**(a-1) * np.exp(-(x/n)**a)

def objective_lin(x,a,b):
    '''
    Objective function for a linear fit
    '''
    return a*x+b

def objective_log(x,a,b,c):
    '''
    Objective function for a linear + log fit
    '''
    return a*x+b*np.log(x)+c

class Branched_3D_Walkers():
    def __init__(self, rate = 0.01, jump = 1.0, box = 100.0, num_paths = 2000, bins = 100,
                 box_min = 60, box_max = 200, log_points = 10, pooling = False,
                 lattice = False, purge = True, terminate = True, gaussian = False, plot_ref = False,
                  rho_min = 0.1, rho_max = 0.9, rho_points = 6, stationary = False, density_terminate = False,
                 parallel = False, count_link = False, correlated = False, correlation = 8.0/28.0, 
                 time_discrete = 1.0, dependent = False):
        self.stationary = stationary
        self.rate       = rate
        self.jump_rate  = int(1/rate)
        self.jump       = jump
        self.box        = box
        self.num_paths  = num_paths
        self.bins       = bins
        self.box_min    = box_min
        self.box_max    = box_max
        self.log_points = log_points
        self.prox_limit = 1
        self.criteria   = 1
        self.gaussian   = gaussian
        self.lattice    = lattice
        self.purge      = purge
        self.rho_min    = rho_min
        self.rho_max    = rho_max
        self.rho_points = rho_points
        self.terminate  = terminate
        self.parallel   = parallel
        self.density_terminate = density_terminate
        self.density_threshold = 3 / 4 / np.pi / 0.25 ** 3
        self.plot_ref   = plot_ref
        self.n_array = []
        self.r_array = []
        self.shortest_path = None
        self.shortest_time = None
        if dependent:
            self.cov = np.array([[1, 0.5, 0.25],[0.5, 1.5, 0.5],[0.25, 0.5, 0.5]])
            self.add_dep = 'dep'
        else:
            self.cov = np.array([[1, 0.0, 0.0],[0.0, 1.5, 0.0],[0.0, 0.0, 0.5]])
            self.add_dep = 'ind'
        n = np.linalg.inv(self.cov) @ np.array([[1],[0],[0]])
        self.roter = n/np.linalg.norm(n)
        self.n_ref = 0
        self.r_ref = 0
        self.color = 'r'
        self.order = 4
        self.store = []
        self.count_link = count_link
        self.correlated = correlated
        if self.gaussian:
            self.add_bbm = 'bbm_'
            self.add_jump = ''
            self.dt         = 1.0/time_discrete
            self.change_jump = False
            if jump!=1:
                data = np.genfromtxt('CGMD_MSID.txt')
                self.n_msid, self.msid = data[:,0], data[:,1]
                self.change_jump = True if rho_max <2 else False
        else:
            self.add_bbm = ''
            self.dt         = 1.0
            if jump == 1:
                self.add_jump = ''
                self.change_jump = False
            else:
                data = np.genfromtxt('CGMD_MSID.txt')
                self.n_msid, self.msid = data[:,0], data[:,1]
                self.add_jump = 'scaled_'
                self.change_jump = True 
        if self.correlated:
            self.correlation = correlation
            self.add_corr = 'corr_'
        else:
            self.correlation = 0.0
            self.add_corr = ''
        if purge:
            self.add_purge = '_purge'
        else:
            self.add_purge = ''
        if terminate:
            self.add_terminate = '_terminate'
        else:
            self.add_terminate = ''
        if density_terminate:
            self.add_den_terminate = '_den_terminate'
        else:
            self.add_den_terminate = ''
        if stationary:
            self.add_stationary = '_stationary'
        else:
            self.add_stationary = ''
        if self.lattice:
            self.lat_jump = np.array([[self.jump,self.jump,self.jump],[self.jump,self.jump,-self.jump],[self.jump,-self.jump,self.jump],[self.jump,-self.jump,-self.jump],[-self.jump,self.jump,-self.jump],[-self.jump,self.jump,self.jump],[-self.jump,-self.jump,-self.jump],[-self.jump,-self.jump,self.jump]])/3**0.5

    def get_angles(self,r):
        """
        Generate random angles in radians.
        Parameters:
            r (int): The number of angles to generate.
        Returns:
            tuple: A tuple containing two arrays of angles in radians.
                   The first array represents the azimuthal angle (theta),
                   and the second array represents the polar angle (phi).
        """
        if not self.stationary:
            theta = 2*np.pi*np.random.rand(r,self.jump_rate)
            phi = np.arccos(1-2*np.random.rand(r,self.jump_rate))
        else:
            theta = 2 * np.pi * np.random.rand(r, 1)
            phi = np.arccos(1 - 2 * np.random.rand(r, 1))
        return theta, phi

    def branch_choice(self):
        """
        Generates a random branch choice based on the current settings.

        Returns:
            numpy.ndarray: A random branch choice generated based on the current settings.
        """
        if not self.stationary:
            if self.gaussian:
                brancher = np.random.multivariate_normal(np.zeros(3), self.jump**2 * np.identity(3),(3 * self.rows, self.jump_rate))
            elif self.lattice:
                brancher = np.array([[random.choice(self.lat_jump) for i in range(self.jump_rate)] for j in range(3*self.rows)])
            else:
                brancher = np.zeros((3*self.rows,self.jump_rate,3))
                theta, phi = self.get_angles(3*self.rows)
                brancher[:,:,0], brancher[:,:,1], brancher[:,:,2] = self.jump*np.sin(phi)*np.cos(theta), self.jump*np.sin(phi)*np.sin(theta), self.jump*np.cos(phi)
        else:
            if self.gaussian:
                brancher = np.random.multivariate_normal(np.zeros(3), self.cov * self.dt ,(self.rows + 2 *(1-int(self.count_link))* self.row_add,)) 
            elif self.lattice:
                brancher = np.array([[random.choice(self.lat_jump) for i in range(1)] for j in range(self.rows + 2 *(1-int(self.count_link))* self.row_add)])
            else:
                brancher = np.zeros((self.rows + 2 *(1-int(self.count_link))* self.row_add,3))
                theta, phi = self.get_angles(self.rows + 2 *(1-int(self.count_link))* self.row_add)
                theta = theta.reshape(-1,)
                phi = phi.reshape(-1, )
                brancher[:,0], brancher[:,1], brancher[:,2] = self.jump*np.sin(phi)*np.cos(theta), self.jump*np.sin(phi)*np.sin(theta), self.jump*np.cos(phi)
        return brancher

    def crossed_barrier(self):
        """
        Calculates if the termination criteria is met by the evolved trajectory.

        Parameters:
            self (object): The instance of the class.
        
        Returns:
            bool: True if the termination criteria is met, False otherwise.
        """
        self.evolved_traj = self.traj.cumsum(1)
        self.crossing = ((self.evolved_traj[:,:,0]-self.box)**2+self.evolved_traj[:,:,1]**2+self.evolved_traj[:,:,1]**2<4).astype(int)
        self.maxd = self.evolved_traj.max()
        return self.criteria in self.crossing

    def init_walk(self):
        """
        Initializes the random walk by generating a trajectory of positions.
        If the BRW not stationary (branching is deterministic), it generates a trajectory with either a Gaussian distribution or a discrete random choice. 
        - If the distribution is Gaussian, the trajectory is generated using `np.random.normal` function with a mean of 0 and a standard deviation of `self.jump`. The trajectory has shape `(1, self.jump_rate)`.
        - If the distribution is discrete, the trajectory is generated using `np.random.choice` function with choices of `[-self.jump, self.jump]`. The trajectory has shape `(1, self.jump_rate)`.
        If the BRW is stationary (branching at each timestep), it initializes `self.ctlk` as an array of zeros with length 1. Then, it generates a trajectory with either a Gaussian distribution or a discrete random choice. 
        - If the distribution is Gaussian, the trajectory is generated using `np.random.normal` function with a mean of 0 and a standard deviation of `self.jump`. The trajectory has shape `(1, 1)`.
        - If the distribution is discrete, the trajectory is generated using `np.random.choice` function with choices of `[-self.jump, self.jump]`. The trajectory has shape `(1, 1)`.
        """
        if not self.stationary:
            if self.gaussian:
                self.traj = np.random.multivariate_normal(np.zeros(3),self.jump**2*np.identity(3),(1,self.jump_rate))
            elif self.lattice:
                self.traj = np.array([[random.choice(self.lat_jump) for i in range(self.jump_rate)] for j in range(1)])
            else:
                self.traj = np.zeros((1,self.jump_rate,3))
                theta, phi = self.get_angles(1)
                self.traj[:,:,0], self.traj[:,:,1], self.traj[:,:,2] = self.jump*np.sin(phi)*np.cos(theta), self.jump*np.sin(phi)*np.sin(theta), self.jump*np.cos(phi)
        else:
            self.ctlk = np.zeros(1)
            if self.gaussian:
                self.traj = np.random.multivariate_normal(np.zeros(3),self.cov*self.dt,(1,1)) 
            elif self.lattice:
                self.traj = np.array([[random.choice(self.lat_jump) for i in range(1)] for j in range(1)])
            else:
                self.traj = np.zeros((1,1,3))
                theta, phi = self.get_angles(self.traj.shape[0])
                self.traj[:,:,0], self.traj[:,:,1], self.traj[:,:,2] = self.jump*np.sin(phi)*np.cos(theta), self.jump*np.sin(phi)*np.sin(theta), self.jump*np.cos(phi)

    def branch_traj(self):
        """
        Branches the trajectory based on initialized conditions (classical or adjusted/termination-based).
        Parameters:
            None
        Returns:
            None
        """
        self.rows,self.cols,_ = self.traj.shape
        if not self.stationary:
            new_traj = np.zeros((3*self.rows,self.cols+self.jump_rate,3))
            new_traj[:self.rows,:self.cols,:] = self.traj.copy()
            new_traj[self.rows:2*self.rows, :self.cols,:] = self.traj.copy()
            new_traj[2*self.rows:3 * self.rows, :self.cols, :] = self.traj.copy()
            new_traj[:,self.cols:,:] = self.branch_choice()
            self.traj = new_traj.copy()
        else:
            if not self.count_link:
                p_val = np.random.rand(self.rows)
            else:
                c_ind = np.where(self.ctlk==0)[0]
                p_val = np.random.rand(len(c_ind))
            b_ind = np.where(p_val < self.rate * self.dt)[0]
            self.row_add = len(b_ind)
            new_ct = np.zeros(self.rows + 2 * self.row_add)
            new_ct[self.rows:self.rows + self.row_add] += 1
            new_ct[self.rows + self.row_add:self.rows + 2 * self.row_add] += 1
            new_traj = np.zeros((self.rows + 2 * self.row_add, self.cols + 1, 3))
            if self.gaussian:
                jump_add = np.random.multivariate_normal(np.zeros(3), self.cov *self.dt ,(self.row_add,)) 
            else:
                theta, phi = self.get_angles(self.row_add)
                theta = theta.reshape(-1,)
                phi = phi.reshape(-1,)
                jump_add = np.zeros((self.row_add,3))
                jump_add[:,0], jump_add[:,1], jump_add[:,2] = int(self.count_link)*self.jump*np.sin(phi)*np.cos(theta), int(self.count_link)*self.jump*np.sin(phi)*np.sin(theta), int(self.count_link)*self.jump*np.cos(phi)
            new_traj[:self.rows, :self.cols, :] = self.traj.copy()
            if not self.count_link:
                new_traj[self.rows:self.rows + self.row_add, :self.cols, :] = self.traj[b_ind, :].copy()
                new_traj[self.rows + self.row_add:self.rows + 2 * self.row_add, :self.cols, :] = self.traj[b_ind, :].copy()
            else:
                new_traj[self.rows:self.rows + self.row_add, :self.cols, :] = self.traj[c_ind[b_ind], :].copy()
                new_traj[self.rows + self.row_add:self.rows + 2 * self.row_add, :self.cols, :] = self.traj[c_ind[b_ind],:].copy()
            if self.gaussian:
                new_traj[:self.rows+2*(1-int(self.count_link))*self.row_add, -1, :] = self.branch_choice()
            else:
                t_branch = self.branch_choice()
                to_jump = self.correlation*new_traj[:self.rows, -2, :] + (1-self.correlation**2)**0.5*t_branch[:self.rows,:] 
                new_traj[:self.rows, -1, :] = np.diag(self.jump / np.linalg.norm(to_jump,axis=1) ) @ to_jump 
            if self.count_link:
                new_traj[self.rows:self.rows + self.row_add, -1,:] = jump_add.copy()
                new_traj[self.rows + self.row_add:self.rows + 2 * self.row_add, -1,:] = jump_add.copy()
                self.ctlk = new_ct.copy()
            else:
                if not self.gaussian:
                    new_traj[self.rows:self.rows+self.row_add,-1,:] = t_branch[self.rows:self.rows+self.row_add,:]
                    no_jump = self.correlation*t_branch[self.rows:self.rows+self.row_add,:] + (1-self.correlation**2)**0.5*t_branch[self.rows+self.row_add:self.rows+2*self.row_add,:]
                    new_traj[self.rows+self.row_add:self.rows+2*self.row_add,-1,:] = np.diag(self.jump / np.linalg.norm(no_jump,axis=1) ) @ no_jump 
            self.traj = new_traj.copy()


    def count_nodes(self,r_in):
        """
        Count the number of nodes within a given radius.

        Parameters:
            r_in (float): The radius within which to count nodes.

        Returns:
            int: The number of nodes within the given radius.
        """
        evolved_pos = self.traj.cumsum(1)
        self.final_loc = evolved_pos[:,-1,:]
        ct_ind = np.where(evolved_pos[:,:,0]**2+evolved_pos[:,:,1]**2+evolved_pos[:,:,2]**2<r_in**2)[0]
        #ct_ind = np.where(self.final_loc[:, 0] ** 2 + self.final_loc[:, 1] ** 2 < r_in ** 2)[0]
        return len(ct_ind)

    def terminate_density(self):
        """
        Terminate the density calculation if the measured density exceeds the density threshold or if the maximum radius has been reached.
        Parameters:
            self (object): The instance of the class.
        
        Returns:
            None
        """
        n = []
        r = []
        r0 = self.r_ref + 1
        del_rows = False
        measured_density = 0
        while (measured_density < self.density_threshold) and (r0<self.box):
            new_counts = self.count_nodes(r0)
            measured_density = (self.n_ref+new_counts)*3/4/np.pi/r0**3
            n.append(self.n_ref+new_counts)
            r.append(r0)
            if measured_density > self.density_threshold:
                self.r_ref = r0
                self.n_ref += new_counts
                self.r_array.extend(r)
                self.n_array.extend(n)
                del_rows = True
            r0 += 1
        if del_rows:
            del_ind = np.where(self.final_loc[:,0]**2+self.final_loc[:,1]**2+self.final_loc[:,2]**2<self.r_ref**2)[0]
            self.traj = np.delete(self.traj,del_ind[:-1],axis=0)
        '''
        self.calc_front()
        del_ind = np.where(self.final_loc[:,0]**2+self.final_loc[:,1]**2<self.r_front**2)[0]
        if len(del_ind) > 0:
            self.traj = np.delete(self.traj, del_ind[:-1], axis=0)
        '''

    def purge_traj(self):
        """
        Purges the trajectory if the number of paths in the trajectory is greater than 9000.
        Retain the forwardmost 3^7 paths.
        Updates the trajectory array with the sorted and trimmed array.
        If the 'stationary' flag is True and the 'count_link' flag is True, also sorts and trims the 'ctlk' array.
        """
        lll = 1500 
        llp = 387 
        if self.traj.shape[0] > lll:
            traversal = self.traj[:,:,:].cumsum(1)
            traversal_p = traversal[:,-1,:] @ self.roter
            end_p = traversal_p[:,0]
            arranged_traj = self.traj[np.argsort(end_p),:,:].copy()
            self.traj = arranged_traj[-llp:,:,:].copy()
            if self.stationary:
                if self.count_link:
                    arranged_ct = self.ctlk[np.argsort(end_p)].copy()
                    self.ctlk = arranged_ct[-llp:].copy()

    def terminate_paths(self):
        """
        Terminates certain paths in the trajectory based on a random probability.
        
        Parameters:
            None.
        
        Returns:
            None.
        """
        if self.traj.shape[0]>1:
            cpath = self.traj.shape[0]
            p_val = np.random.rand(cpath)
            dind = np.where(p_val < 2.0 / 500.0 * self.dt)[0]
            if len(dind)==self.traj.shape[0]:
                ttind = dind.copy()
                dind = ttind[:-1]
            self.traj = np.delete(self.traj, dind, axis=0)
            if self.count_link:
                self.ctlk = np.delete(self.ctlk, dind)

    def SP_evolution(self):
        """
        Performs the evolution of the SP algorithm.
        Parameters:
            self (object): The current instance of the class.
        
        Returns:
            shortest_path (float): The shortest path of the evolved trajectory.
            shortest_time (float): The shortest time of the evolved trajectory.
        """
        should_restart = True
        reg_stuff = True
        while should_restart:
            should_restart = False
            self.init_walk()
            ct = 0
            while self.crossed_barrier() is False:
                ct+=1
                if self.purge:
                    self.purge_traj()
                self.branch_traj()
                if self.density_terminate:
                    self.terminate_density()
                if self.terminate:
                    if ct > 5:
                        self.terminate_paths()
                if not self.stationary:
                    if ct > (10000/self.jump_rate):
                        should_restart = True
                        break
                else:
                    if self.gaussian:
                        lim = 400+20/self.rate 
                    else:
                        lim = 4000
                    if ct > (lim):
                        if self.shortest_path is None:
                            should_restart = True
                            break
                        else:
                            should_restart = True
                            reg_stuff = True
                            break
        if reg_stuff:
            self.selected_choices = self.traj[np.where(self.crossing==self.criteria)[0][np.argmin(np.where(self.crossing==self.criteria)[1])],:]
            self.selected_path = self.evolved_traj[np.where(self.crossing == self.criteria)[0][np.argmin(np.where(self.crossing == self.criteria)[1])],:]
            schematic = True
            if schematic:
                final_time = np.min(np.where(self.crossing == self.criteria)[1])
                cc = np.arange(self.traj.shape[0],dtype=int)
                self.store.append(self.selected_path[:,:2])
                plt.figure(13)
                plt.plot(self.selected_path[:,0],self.selected_path[:,1],self.color,lw=3,alpha=0.3,zorder=self.order)
            if not self.stationary:
                self.shortest_path = np.where(self.crossing==self.criteria)[1].min() * self.jump**2 + (1-int(self.gaussian)-int(self.stationary))*ct * self.jump**2
                self.shortest_time = np.where(self.crossing == self.criteria)[1].min() + (1-int(self.gaussian)-int(self.stationary))*ct
            else:
                if self.gaussian:
                    self.shortest_path = np.where(self.crossing == self.criteria)[1].min() * self.jump**2 * self.dt
                    self.shortest_time = np.where(self.crossing == self.criteria)[1].min() * self.dt 
                else:
                    self.shortest_path = np.where(self.crossing == self.criteria)[1].min() * self.jump ** 2
                    self.shortest_time = np.where(self.crossing == self.criteria)[1].min()
        else:
            print('returning old val')
        return self.shortest_path, self.shortest_time

    def SP_distribution(self):
        """
        Calculate the SP distribution for the given number of paths.
        Parameters:
            self (object): The object itself.
        
        Returns:
            None
        """
        if self.change_jump:
            njump = np.interp(1.0/self.rate,self.n_msid, self.msid)
            self.jump = njump**0.5
        self.SPs = np.zeros(self.num_paths)
        self.STs = np.zeros(self.num_paths)
        for i in tqdm(range(self.num_paths)):
            self.SPs[i], self.STs[i] = self.SP_evolution()

        schematic = True
        if schematic:
            #plt.title(r'$q_x=%d$, rate = %.3f'%(self.box,self.rate))
            #plt.legend()
            array_dict = {f'array{i+1}': array for i, array in enumerate(self.store)}
            np.savez('AN_SP_%d_%s.npz'%(self.box,'small' if self.rate<0.5 else 'large'),**array_dict)
            #plt.axis('equal')
            #plt.xlabel('x')
            #plt.ylabel('y')
            #plt.show()

    def plot_and_fit_SPs(self):
        """
        Plot and fit the shortest paths (SPs) data.
        This function generates a plot of the SPs data and fits a histogram to it. 
        It also saves the SPs and the fitted data.
        Parameters:
            None
        Returns:
            None
        """
        plt.figure(1)
        if not self.stationary:
            data = np.genfromtxt('SP_data.dat', dtype=float)[:, :]
        else:
            data = np.genfromtxt('SP_data_poisson.dat', dtype=float)[:, :]
        SP = data[:, 1] - (1-int(self.count_link))*data[:,-2]
        if self.count_link:
            np.savez('cl/%s_%s%s%s3D_hist.npz'%(self.add_dep,self.add_jump,self.add_bbm,self.add_corr),sp_brw = self.STs, sp_cg = SP, sp_brw_dist = self.SPs)
        else:
            np.savez('no_cl/%s_%s%s%s3D_hist.npz'%(self.add_dep,self.add_jump,self.add_bbm,self.add_corr),sp_brw = self.STs, sp_cg = SP, sp_brw_dist = self.SPs)
        bin_range = np.linspace(np.min(np.array([SP.min(), self.STs.min()])), np.max(np.array([SP.max(), self.STs.max()])),
                                self.bins)
        plt.hist(self.STs, bin_range, density=True, alpha=0.3, label='Branched RW')
        if self.plot_ref:
            plt.hist(SP, bin_range, density=True, alpha=0.3, label='CGMD - 3D')
        data = np.genfromtxt('SP_2D_data.dat', dtype=float)[:, :]
        SP = data[:, 1]
        SP = SP[np.isfinite(SP)]
        #plt.hist(SP, self.bins, density=True, alpha=0.3, label='CGMD - 2D')
        plt.xlabel('Shortest Path (SP)')
        plt.ylabel('P(SP)')
        plt.legend()
        plt.tight_layout()
        if self.lattice:
            plt.savefig('CGMD_RW3D_lattice%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
        else:
            plt.savefig('CGMD_RW3D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
        plt.figure(2)
        plt.hist(self.STs, self.bins, density=True)
        plt.xlabel('First passage time (FPT)')
        plt.ylabel('P(FPT)')
        plt.tight_layout()
        schematic = False
        if schematic:
            final_time = np.min(np.where(self.crossing == self.criteria)[1])
            cc = np.arange(self.traj.shape[0],dtype=int)
            store = []
            #plt.figure(13)
            #plt.plot(self.selected_path[:,0],self.selected_path[:,1],'r--',lw=3,label='Shortest Path')
            #plt.plot(self.selected_path[0, 0], self.selected_path[0, 1], 'bo', lw=3, label=r'$p$')
            #plt.plot(self.selected_path[-1, 0], self.selected_path[-1, 1], 'ko', lw=3, label=r'$p^{\prime}$')
            store.append(self.selected_path[:,:2])
            for i in range(5):
                #plt.plot(self.evolved_traj[random.choice(cc),:final_time,0],self.evolved_traj[random.choice(cc),:final_time,1],'k--')
                store.append(self.evolved_traj[random.choice(cc),:final_time,:2])
            #print(self.evolved_traj.shape)
            #plt.legend()
            #plt.xlabel('x')
            #plt.ylabel('y')
            array_dict = {f'array{i+1}': array for i, array in enumerate(store)}
            np.savez('AN_SP_rest_%s.npz'%('small' if self.rate<0.5 else 'large'),**array_dict)
        print(self.STs.mean())	
        plt.show()

    def analyze_rate(self):
        """
        Analyze the rate by performing the following steps:
        
        1. Call the SP_distribution() method to calculate the distribution of SPs.
        2. Call the plot_and_fit_SPs() method to plot and fit the SPs.
        """
        self.SP_distribution()
        self.plot_and_fit_SPs()

    def tau_SP_analysis(self, plot_analysis = True):
        """
        Performs SP analysis for different box lengths and fit the mean of the SPs to the linear+log term.
        Parameters:
        - plot_analysis (bool): Whether to plot the analysis or not. Defaults to True.
        Returns:
        - None
        """
        self.boxes = np.exp(np.linspace(np.log(self.box_min), np.log(self.box_max), self.log_points))
        if self.box_max == 4*self.box_min:
            self.boxes = np.array([0.25,0.5,0.75,1])*65.5028
        test = True
        if test:
            self.boxes = np.array([0.1,0.15,0.2,0.25,0.5,0.75,1])*self.box
        schematic = True
        if schematic:
            self.boxes = np.array([15,30,45,60])
        self.tau = np.zeros(len(self.boxes))
        self.sps = np.zeros(len(self.boxes))
        self.spsig = np.zeros(len(self.boxes))
        self.stsig = np.zeros(len(self.boxes))
        sps = []
        for i, box in enumerate(self.boxes):
            if schematic:
                colors = ['r','b','k','m']
                self.color = colors[i]
                self.order = 4-i
                self.store = []
            self.box = box
            self.SP_distribution()
            sps.append(self.STs)
            self.tau[i], self.sps[i], self.stsig[i], self.spsig[i] = self.STs.mean(), self.SPs.mean(), self.STs.std(), self.SPs.std()
        #if self.count_link:
        #    np.savez('cl/%s%s%sSP_brw.npz'%(self.add_jump,self.add_bbm,self.add_corr),sp10 = sps[0], sp15 = sps[1], sp20 = sps[2], sp25 = sps[3],sp50 = sps[4], sp75=sps[5], sp100 = sps[6],box=self.boxes)
        #else:
        #    np.savez('no_cl/%s%s%sSP_brw.npz'%(self.add_jump,self.add_bbm,self.add_corr),sp10 = sps[0], sp15 = sps[1], sp20 = sps[2], sp25 = sps[3],sp50 = sps[4], sp75=sps[5], sp100 = sps[6],box=self.boxes)
        self.tau_by_x = self.tau / self.boxes
        self.sps_by_x = self.sps / self.boxes
        schematic = True
        if schematic:
            plt.title(r'rate = %.3f'%(self.rate))
            plt.legend()
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        if plot_analysis:
            popt,_=curve_fit(objective_log,self.boxes,self.tau)
            a,b,e=popt
            if self.count_link:
                np.savez('cl/%s_%s%s%stau_3D%s%s%s%s_rate_%.4f.npz'%(self.add_dep,self.add_jump,self.add_bbm,self.add_corr,self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary, self.rate),box =self.boxes, tau = self.tau, a=a,b=b,e=e)
            else:
                np.savez('no_cl/%s_%s%s%stau_3D%s%s%s%s_rate_%.4f.npz'%(self.add_dep,self.add_jump,self.add_bbm,self.add_corr,self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary, self.rate),box =self.boxes, tau = self.tau, a=a,b=b,e=e)
            popt, _ = curve_fit(objective_lin, np.log(self.boxes), np.log(self.stsig))
            c, d = popt
            fig, ax1 = plt.subplots()
            ax1.plot(self.boxes, self.tau,'b--*',lw=3)
            ax1.plot(np.array([0,self.boxes.max()]), objective_lin(np.array([0,self.boxes.max()]),a,b),'k--')
            ax1.set_title(r'$a(x)=%.3f x$ and intercept $=%.3f$'%(a,b))
            ax1.set_xlabel('Box size (x)')
            ax1.set_ylabel(r'$\tau(x)$')
            left, bottom, width, height = 0.5, 0.2, 0.25, 0.25
            ax2 = fig.add_axes([left, bottom, width, height])
            ax2.plot(self.boxes, self.tau, 'b--*', lw=3)
            ax2.plot(np.array([0,self.boxes.max()]), objective_lin(np.array([0,self.boxes.max()]),a,b),'k--')
            ax2.set_xlabel('Box size (x)')
            ax2.set_ylabel(r'$\tau(x)$')
            ax2.set_xlim([0, 15])
            ax2.set_ylim([b-5, b+45])
            fig.savefig('tau_x_3D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(2)
            plt.plot(self.boxes, self.tau_by_x,'b--*',lw=3)
            plt.xlabel('Box size (x)')
            plt.ylabel(r'$\tau(x)/x$')
            plt.savefig('tau_x_by_x_3D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(3)
            plt.plot(self.boxes, self.sps,'b--*',lw=3)
            plt.xlabel('Box size (x)')
            plt.ylabel(r'SP$(x)$')
            plt.savefig('SP_x_3D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(4)
            plt.plot(self.boxes, self.sps_by_x,'b--*')
            plt.xlabel('Box size (x)')
            plt.ylabel(r'SP$(x)/x$')
            plt.savefig('SP_x_by_x_3D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(5)
            plt.loglog(self.boxes, self.stsig, 'b--*', lw=3)
            plt.loglog(self.boxes, np.exp(d)*self.boxes**c, 'k--')
            plt.title(r'Scaling exponent$=%.3f$ ' %c)
            plt.xlabel('Box size (x)')
            plt.ylabel(r'$\sigma(\tau(x))$')
            plt.savefig('sig_tau_x_3D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.tight_layout()
            plt.show()

    def c_rho_analysis(self, plot_analysis=True):
        """
        Performs SP analysis for different box lengths and fit the mean of the SPs to the linear+log term at different branching rates/rho.
        Parameters:
        - plot_analysis (bool): Whether to plot the analysis or not. Defaults to True.
        Returns:
        - None
        """
        self.boxes = np.exp(np.linspace(np.log(self.box_min), np.log(self.box_max), self.log_points))
        if self.box_max == 4 * self.box_min:
            self.boxes = np.array([0.25, 0.5, 0.75, 1]) * 65.5028
        if self.rho_max > 1.5:
            self.rho_arr = np.linspace((self.rho_min), (self.rho_max), self.rho_points)
        else:
            self.rho_arr = np.array([1.0295233547701528, 1.0376824580907913, 1.0457569507079911, 1.0594585479958203, 1.075720859698337, 1.0860194380000792, 1.097926243894198, 1.1085438423022842, 1.1221982305721785, 1.1346392570776829, 1.1485329189890172, 1.158463320987782, 1.1738380193069964, 1.1867845421985102])
            self.boxes = np.array([0.25, 0.5, 0.75, 1]) * 65.5028
        self.boxes = np.array([0.25, 0.5, 0.75, 1]) * 65.5028
        self.c_arr = np.zeros(len(self.rho_arr))
        self.c2_arr = np.zeros(len(self.rho_arr))
        self.c_arr_bar = np.zeros(len(self.rho_arr))
        self.sig_arr = np.zeros(len(self.rho_arr))
        self.all_tau = np.zeros((len(self.rho_arr),len(self.boxes)))
        if not self.count_link:
            if not self.terminate:
                if not self.gaussian:
                    ref_rho = np.linspace(1.001,2.99,40)
                    ref_c1 = np.array([0.02581489562988281, 0.18289817810058595, 0.25312904357910154, 0.30454780578613283, 0.3459314727783203, 0.3807878875732422, 0.41095237731933604, 0.4375498199462891, 0.4613202667236328, 0.4827669525146485, 0.5022895050048829, 0.5201839447021485, 0.5366574859619141, 0.5519173431396484, 0.5660967254638674, 0.5793140411376954, 0.5916876983642578, 0.6033065032958984, 0.6142444610595703, 0.6245459747314452, 0.6342850494384766, 0.6435208892822265, 0.6522682952880859, 0.6605716705322265, 0.668490219116211, 0.676023941040039, 0.6832172393798829, 0.6900997161865234, 0.6966713714599609, 0.7029618072509767, 0.708985824584961, 0.7147730255126953, 0.7203382110595703, 0.725681381225586, 0.7308173370361327, 0.7357756805419923, 0.7405416107177736, 0.7451299285888673, 0.7495702362060548, 0.7538477325439453])
                else:
                    ref_rho = np.linspace(1.1,2.9,19)
                    ref_c1 = np.array([0.48436522724540426, 0.669919043180509, 0.8036290984776359, 0.9100765636379093, 0.9990335121606766, 1.075607740047942, 1.1428743801347632, 1.202854722248695, 1.2569598352776747, 1.3062191410537702, 1.351409379167942, 1.3931324839439145, 1.4318651616234765, 1.467991814987787, 1.5018271799576202, 1.5336323458949432, 1.563626375211189, 1.5919949108393012, 1.6188966704270606])
                    ref_c2 = np.array([13.115073359071328, 6.85601868468384, 4.764368358385502, 3.715016764985576, 3.0828793279705398, 2.6595539315430536, 2.355698170042486, 2.1266219100226706, 1.947483729442224, 1.8033688011112043, 1.6847783830802305, 1.585374254636288, 1.500763967616421, 1.4278065528394757, 1.3641958349216143, 1.3081999242448716, 1.2584925936870774, 1.214040818521264, 1.1740277960669137])
            else:
                ref_rho = np.linspace(1.1,2.9,19)
                ref_c1 = np.array([0.47501822093201074, 0.6637564994269697, 0.7988955421158086, 0.9061988293833917, 0.9957384637186124, 1.0727396977575236, 1.140334674080403, 1.2005762410084022, 1.2548945687448914, 1.3043314141280524, 1.349671883962955, 1.3915238121380318, 1.430368196563771, 1.466592643565939, 1.5005143327495765, 1.5323962523029837, 1.5624589638328703, 1.590889310559688, 1.6178469829886457])
                ref_c2 = np.array([13.636286004864852, 6.983916940495814, 4.820994582664085, 3.746878801172229, 3.1033165103523657, 2.673793937332584, 2.3662028812320934, 2.1347014736222496, 1.953899220864762, 1.8085925214980831, 1.6891189618798084, 1.5890419188297966, 1.503906888191967, 1.4305321891461393, 1.3665840349496698, 1.310311270749875, 1.2603738942079692, 1.2157288181333716, 1.1755517498548])
        else:
            if not self.terminate:
                ref_rho = np.linspace(1.001, 2.99, 100)
                ref_c1 = np.array([0.025800094604492185, 0.11678199768066404, 0.16063743591308594, 0.1930368804931641, 0.21929389953613285, 0.24156944274902345, 0.26098838806152347, 0.2782315826416015, 0.2937430572509766, 0.3078632354736328, 0.3208289337158203, 0.3327881622314453, 0.3439037322998047, 0.35427925109863284, 0.36398872375488284, 0.3731505584716797, 0.38177955627441407, 0.3899497222900391, 0.3977054595947266, 0.4050763702392579, 0.41210685729980473, 0.41882652282714855, 0.42525016784667974, 0.43140739440917975, 0.43732780456542975, 0.44301139831542974, 0.4484729766845703, 0.4537421417236328, 0.4588188934326172, 0.4637328338623047, 0.46846916198730476, 0.4730574798583985, 0.477497787475586, 0.48180488586425785, 0.48597877502441406, 0.49003425598144534, 0.49397132873535154, 0.49780479431152347, 0.501534652709961, 0.5051461029052735, 0.5086835479736328, 0.5121173858642578, 0.5154772186279297, 0.5187482452392578, 0.5219304656982422, 0.5250534820556642, 0.5280876922607423, 0.5310626983642578, 0.533978500366211, 0.5368202972412109, 0.5396028900146485, 0.5423262786865234, 0.5449904632568359, 0.547595443725586, 0.5501560211181641, 0.5526573944091796, 0.5551291656494141, 0.557541732788086, 0.5598950958251954, 0.5622188568115235, 0.5644982147216797, 0.5667331695556642, 0.5689385223388673, 0.5710846710205078, 0.5732160186767579, 0.5753029632568359, 0.5773455047607423, 0.5793584442138673, 0.5813417816162111, 0.5832955169677735, 0.5852048492431642, 0.5870993804931641, 0.5889495086669924, 0.5907848358154297, 0.5925757598876954, 0.5943518829345703, 0.5960984039306642, 0.5978153228759766, 0.5995174407958984, 0.601189956665039, 0.6028328704833984, 0.6044609832763673, 0.6060594940185547, 0.6076432037353516, 0.6091973114013672, 0.6107366180419923, 0.612246322631836, 0.613741226196289, 0.6152213287353516, 0.6166718292236327, 0.618122329711914, 0.6195432281494141, 0.6209345245361327, 0.6223258209228516, 0.6236875152587891, 0.6250344085693359, 0.6263813018798828, 0.6276985931396484, 0.6290010833740234, 0.6302887725830078])
            else:
                ref_rho = np.linspace(1.1,2.9,19)
                ref_c1 = np.array([0.46480816397970254, 0.6376839499722937, 0.7582763877573903, 0.8511755200233665, 0.9261910387617417, 0.9902923598818508, 1.0460507337167166, 1.0951884518658228, 1.1389243256791164, 1.1788092300318593, 1.2147919240405072, 1.2485734451654562, 1.2793162661468618, 1.307913051587142, 1.3345983115874998, 1.359569372250516, 1.3834432369976868, 1.4054469977702788, 1.4265884553075612])
                ref_c2 = np.array([14.241939746347155, 7.566685205207855, 5.351328035061747, 4.246962141051647, 3.586868784967414, 3.1375436902118463, 2.811972972806884, 2.5653045266354235, 2.3720667241783366, 2.2142649400293224, 2.085032586753077, 1.9737332204348157, 1.8800127530780222, 1.798700650937313, 1.7274898202777986, 1.664615346367086, 1.6076590504157586, 1.5577138981242844, 1.5118866377278726])
        for j, rho in enumerate(self.rho_arr):
            self.rate = (rho - 1.0) / 2.0
            self.jump_rate = int(1 / self.rate)
            #if self.rho_max > 1.5:
            self.tau = np.zeros(len(self.boxes))
            self.sps = np.zeros(len(self.boxes))
            self.spsig = np.zeros(len(self.boxes))
            self.stsig = np.zeros(len(self.boxes))
            for i, box in enumerate(self.boxes):
                self.box = box
                self.SP_distribution()
                self.tau[i], self.sps[i], self.stsig[i], self.spsig[
                    i] = self.STs.mean(), self.SPs.mean(), self.STs.std(), self.SPs.std()

            popt, _ = curve_fit(objective_log, self.boxes, self.tau)
            a, b, e = popt
            popt2, _ = curve_fit(objective_lin, self.boxes, self.tau)
            c,d = popt2
            self.c_arr[j] = 1.0 / a
            self.c2_arr[j] = b
            self.c_arr_bar[j] = 1.0 / c #self.boxes[-1]/self.tau[-1]
            self.sig_arr[j] = self.stsig[-1]
            self.all_tau[j,:] = self.tau
        if not self.count_link:
            if not self.terminate:
                np.savez('no_cl/asymm_%s_%s%s%s%s3D_C1_rho.npz'%(self.add_dep,self.add_jump,self.add_bbm,self.add_corr,'' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>3.5 else self.sig_arr, c1_bar = np.ones(4) if self.rho_max>3.5 else self.c_arr_bar, boxes = self.boxes, fpt = self.all_tau, c2_brw = self.c2_arr, c2_ref = ref_c2)
            else:
                np.savez('no_cl/asymm_%s_%s%s%s%sjust_term_3D_C1_rho.npz'%(self.add_dep,self.add_jump,self.add_bbm,self.add_corr,'' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>3.5 else self.sig_arr, c1_bar = np.ones(4) if self.rho_max>3.5 else self.c_arr_bar, boxes = self.boxes, fpt = self.all_tau, c2_brw = self.c2_arr, c2_ref = ref_c2)
        else:
            if not self.terminate:
                np.savez('cl/asymm_%s_%s%s%s%sadj_3D_C1_rho.npz'%(self.add_dep,self.add_jump,self.add_bbm,self.add_corr,'' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>3.5 else self.sig_arr, c1_bar = np.ones(4) if self.rho_max>3.5 else self.c_arr_bar, boxes = self.boxes, fpt = self.all_tau, c2_brw = self.c2_arr, c2_ref = ref_c2)
            else:
                np.savez('cl/asymm_%s_%s%s%s%sterm_3D_C1_rho.npz'%(self.add_dep,self.add_jump,self.add_bbm,self.add_corr,'' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>3.5 else self.sig_arr, c1_bar = np.ones(4) if self.rho_max>3.5 else self.c_arr_bar, boxes = self.boxes, fpt = self.all_tau, c2_brw = self.c2_arr, c2_ref = ref_c2)
        if plot_analysis:
            plt.figure(1)
            plt.plot(self.rho_arr, self.c_arr, 'b--*', lw=3,label='3D BRW')
            plt.plot(ref_rho, ref_c1, 'k--o', lw=3, label='Reference')
            plt.legend()
            plt.xlabel(r'$\rho$',fontsize=12)
            plt.ylabel(r'$C_1$',fontsize=12)
            plt.savefig('C1_vs_rho_3D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            #plt.show()


def main():
    parser = argparse.ArgumentParser(description="Computing branched random walk analysis")
    parser.add_argument("--single_analysis", default=False, action="store_true", help="to plot distribution for a single rate")
    parser.add_argument("--tau_analysis",      default=False, action="store_true", help="to plot Sp distributions as a function of box size")
    parser.add_argument("--crho_analysis", default=False, action="store_true",
                        help="to plot C1 vs rho")
    parser.add_argument("--lattice", default=False, action="store_true", help="RW along a cubic lattice")
    parser.add_argument("--purge", default=False, action="store_true", help="Whether to purge backwardmost paths")
    parser.add_argument("--gaussian", default=False, action="store_true", help="uncorrelated gaussian jumps")
    parser.add_argument("--correlated", default=False, action="store_true", help="Whether steps are correlated")
    parser.add_argument("--stationary", default=False, action="store_true", help="Deterministic time interval before jumps or every event jump")
    parser.add_argument("--parallel", default=False, action="store_true",
                        help="Deterministic time interval before jumps or every event jump")
    parser.add_argument("--plot_ref", default=False, action="store_true", help="If reference CGMD should be plotted")
    parser.add_argument("--count_link", default=False, action="store_true", help="Choose whether weight of croslsink is to be calculated")
    parser.add_argument("--terminate", default=False, action="store_true", help="Whether to arbitrarily terminate path to ensure non-infinite chains")
    parser.add_argument("--dependent", default=False, action="store_true", help="Whether the multivariate gaussian distribution for the jumps has dependency")
    parser.add_argument("--density_terminate", default=False, action="store_true",
                        help="terminate path in high density regions")
    parser.add_argument("--rate",   type=float, default=5.25/105, help="branching rate in units of inverse time")
    parser.add_argument("--jump", type=float, default=1, help="jump length")
    parser.add_argument("--correlation", type=float, default=0.4, help="extent of correlation")
    parser.add_argument("--box", type=float, default=65.5028, help="box length or periodic repeat length for completion")
    parser.add_argument("--num_paths", type=int,   default=1000,   help="number of paths for computing distribution")
    parser.add_argument("--bins", type=int, default=25, help="number of bins for SP distribution")
    parser.add_argument("--dim", type=int, default=1, help="Branching random-walk dimensionality")
    parser.add_argument("--box_min", type=int, default=10, help="minimum box size to use in tau analysis")
    parser.add_argument("--box_max", type=int, default=60, help="maximum box size to use in tau analysis")
    parser.add_argument("--rho_min", type=float, default=1.1, help="minimum rate to use in tau analysis")
    parser.add_argument("--rho_max", type=float, default=1.9, help="maximum rate to use in tau analysis")
    parser.add_argument("--time_discrete", type=float, default=1.0, help="discretization steps per unit time")
    parser.add_argument("--log_points", type=int, default=4, help="number of points on the log scale for deciding box sizes")
    parser.add_argument("--rho_points", type=int, default=9,
                        help="number of points on the log scale for deciding branching rates")
    args = parser.parse_args()
    if args.dim == 1:
        raise ValueError('asymmetric jump code has not been written for --dim 1. This script currently only suppoerts --dim 3.')
    elif args.dim == 2:
        raise ValueError('asymmetric jump code has not been written for --dim 1. This script currently only suppoerts --dim 3.')
    elif args.dim == 3:
        util = Branched_3D_Walkers(rate=args.rate, jump=args.jump, box=args.box, num_paths=args.num_paths,
                           bins=args.bins, box_min=args.box_min, box_max=args.box_max, time_discrete=args.time_discrete,
                           log_points=args.log_points, lattice=args.lattice, purge=args.purge, terminate=args.terminate,
                                   plot_ref=args.plot_ref, rho_min=args.rho_min, rho_max=args.rho_max, rho_points=args.rho_points,
                                   stationary=args.stationary, density_terminate=args.density_terminate, parallel=args.parallel,
                                   count_link=args.count_link, correlated=args.correlated, correlation=args.correlation,
                                   gaussian=args.gaussian,dependent=args.dependent)
    else:
        raise ValueError("Incorrect dimensionality specified by --dim. Try default --dim or --dim [2,3]")

    if args.single_analysis:
        util.analyze_rate()
    if args.tau_analysis:
        util.tau_SP_analysis()
    if args.crho_analysis:
        util.c_rho_analysis()


if __name__ == '__main__':
    main()

