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

class Branched_1D_Walkers():
    def __init__(self, rate = 0.01, jump = 1.0, box = 100.0, num_paths = 2000, bins = 100,
                 box_min = 60, box_max = 2000, log_points = 10, pooling = False,
                 purge = True, terminate = True, gaussian = False, plot_ref = False,
                 density_terminate = False, rho_min = 0.1, rho_max = 0.9, rho_points = 6,
                 stationary = False, parallel = False, count_link = False):
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
        self.criteria   = 1
        self.gaussian   = gaussian
        self.purge      = purge
        self.terminate  = terminate
        self.rho_min    = rho_min
        self.rho_max    = rho_max
        self.parallel   = parallel
        self.rho_points = rho_points
        self.density_terminate = density_terminate
        self.plot_ref   = plot_ref
        self.density_threshold = 1/np.pi/0.5**2
        self.n_array = []
        self.r_array = []
        self.n_ref = 0
        self.r_ref = 0
        self.count_link = count_link
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

    def branch_choice(self):
        """
        Generates a random branch choice based on the current settings.

        Returns:
            numpy.ndarray: A random branch choice generated based on the current settings.
        """
        if not self.stationary:
            if self.gaussian:
                return np.random.normal(0,self.jump,(3*self.rows,self.jump_rate))
            else:
                return np.random.choice(np.array([-self.jump,self.jump]),(3*self.rows,self.jump_rate))
        else:
            if self.gaussian:
                return np.random.normal(0,self.jump,(self.rows+2*(1-int(self.count_link))*self.row_add,))
            else:
                return np.random.choice(np.array([-self.jump,self.jump]),(self.rows+2*(1-int(self.count_link))*self.row_add,))

    def count_nodes(self,r_in):
        """
        Count the number of nodes within a given radius.

        Parameters:
            r_in (float): The radius within which to count nodes.

        Returns:
            int: The number of nodes within the given radius.
        """
        evolved_pos = self.traj.cumsum(1)
        self.final_loc = evolved_pos[:,-1]
        #ct_ind = np.where(abs(evolved_pos)<r_in)[0]
        ct_ind = np.where(abs(self.final_loc) < r_in)[0]
        return len(ct_ind)

    def calc_front(self):
        """
        Calculate the front of the trajectory.

        This function takes no parameters.

        Returns:
            None
        """
        evolved_pos = self.traj.cumsum(1)
        self.final_loc = evolved_pos[:,-1]
        self.r_front = 0.1*self.traj.shape[1]**0.5

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
            measured_density = (self.n_ref+new_counts)/2/r0
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
            del_ind = np.where(abs(self.final_loc)<self.r_ref)[0]
            self.traj = np.delete(self.traj,del_ind[:-1],axis=0)
        '''
        self.calc_front()
        del_ind = np.where(abs(self.final_loc) < self.r_front)[0]
        if len(del_ind)>0:
            self.traj = np.delete(self.traj, del_ind[:-1], axis=0)
        '''

    def crossed_barrier(self):
        """
        Calculates if the termination criteria is met by the evolved trajectory.

        Parameters:
            self (object): The instance of the class.
        
        Returns:
            bool: True if the termination criteria is met, False otherwise.
        """
        self.evolved_traj = self.traj.cumsum(1)
        self.crossing = (abs(self.evolved_traj-self.box)<1).astype(int)
        self.maxd = self.evolved_traj.max()
        return self.criteria in self.crossing

    def init_walk(self):
        """
        Initializes the random walk by generating a trajectory of positions.
        If the BRW not stationary (branching is deterministic), it generates a trajectory with either a Gaussian distribution or a discrete random choice. 
        - If the distribution is Gaussian, the trajectory is generated using `np.random.normal` function with a mean of 0 and a standard deviation of `self.jump`. The trajectory has shape `(1, self.jump_rate)`.
        - If the distribution is discrete, the trajectory is generated using `np.random.choice` function with choices of `[-self.jump, self.jump]`. The trajectory has shape `(1, self.jump_rate)`.
        If the BRW is stationary (branching at each timestep), it initializes `self.ctlk` as an array of zeros with length 2000. Then, it generates a trajectory with either a Gaussian distribution or a discrete random choice. 
        - If the distribution is Gaussian, the trajectory is generated using `np.random.normal` function with a mean of 0 and a standard deviation of `self.jump`. The trajectory has shape `(2000, 1)`.
        - If the distribution is discrete, the trajectory is generated using `np.random.choice` function with choices of `[-self.jump, self.jump]`. The trajectory has shape `(2000, 1)`.
        """
        if not self.stationary:
            if self.gaussian:
                self.traj = np.random.normal(0,self.jump,(1,self.jump_rate))
            else:
                self.traj = np.random.choice(np.array([-self.jump,self.jump]),(1,self.jump_rate))
        else:
            self.ctlk = np.zeros(1)
            if self.gaussian:
                self.traj = np.random.normal(0,self.jump,(1,1))
            else:
                self.traj = np.random.choice(np.array([-self.jump,self.jump]),(1,1))

    def branch_traj(self):
        """
        Branches the trajectory based on initialized conditions (classical or adjusted/termination-based).
        Parameters:
            None
        Returns:
            None
        """
        self.rows,self.cols = self.traj.shape
        if not self.stationary:
            new_traj = np.zeros((3*self.rows,self.cols+self.jump_rate))
            new_traj[:self.rows,:self.cols] = self.traj.copy()
            new_traj[self.rows:2*self.rows, :self.cols] = self.traj.copy()
            new_traj[2*self.rows:3 * self.rows, :self.cols] = self.traj.copy()
            new_traj[:,self.cols:] = self.branch_choice()
            self.traj = new_traj.copy()
        else:
            if not self.count_link:
                p_val = np.random.rand(self.rows)
            else:
                c_ind = np.where(self.ctlk == 0)[0]
                p_val = np.random.rand(len(c_ind))
            b_ind = np.where(p_val<self.rate)[0]
            self.row_add = len(b_ind)
            new_traj = np.zeros((self.rows+2*self.row_add, self.cols + 1))
            new_ct = np.zeros(self.rows + 2 * self.row_add)
            new_traj[:self.rows, :self.cols] = self.traj.copy()
            new_ct[self.rows:self.rows+self.row_add]+= 1
            new_ct[self.rows+self.row_add:self.rows+2*self.row_add] += 1
            jump_add = np.zeros(self.row_add)
            jump_add = int(self.count_link)*np.random.choice(np.array([-self.jump,self.jump]),(self.row_add,))
            if not self.count_link:
                new_traj[self.rows:self.rows+self.row_add, :self.cols] = self.traj[b_ind,:].copy()
                new_traj[self.rows+self.row_add:self.rows+2*self.row_add, :self.cols] = self.traj[b_ind,:].copy()
            else:
                new_traj[self.rows:self.rows + self.row_add, :self.cols] = self.traj[c_ind[b_ind], :].copy()
                new_traj[self.rows + self.row_add:self.rows + 2 * self.row_add, :self.cols] = self.traj[c_ind[b_ind], :].copy()
            new_traj[:self.rows+2*(1-int(self.count_link))*self.row_add, -1] = self.branch_choice()
            if self.count_link:
                new_traj[self.rows:self.rows + self.row_add, -1] = jump_add.copy()
                new_traj[self.rows + self.row_add:self.rows + 2 * self.row_add, -1] = jump_add.copy()
                self.ctlk = new_ct.copy()
            self.traj = new_traj.copy()


    def purge_traj(self):
        """
        Purges the trajectory if the number of paths in the trajectory is greater than 9000.
        Retain the forwardmost 3^7 paths.
        Updates the trajectory array with the sorted and trimmed array.
        If the 'stationary' flag is True and the 'count_link' flag is True, also sorts and trims the 'ctlk' array.
        """
        if self.traj.shape[0] > 9000:
            traversal = self.traj.cumsum(1)
            end_p = traversal[:,-1]
            arranged_traj = self.traj[np.argsort(end_p),:].copy()
            self.traj = arranged_traj[-3**7:,:].copy()
            if self.stationary:
                if self.count_link:
                    arranged_ct = self.ctlk[np.argsort(end_p)].copy()
                    self.ctlk = arranged_ct[-3 ** 7:].copy()

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
            dind = np.where(p_val < 2.0/500.0 )[0]
            if len(dind)==self.traj.shape[0]:
                ttind = dind.copy()
                dind = ttind[:-1]
            self.traj = np.delete(self.traj,dind,axis=0)
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

        self.selected_choices = self.traj[np.where(self.crossing==self.criteria)[0][np.argmin(np.where(self.crossing==self.criteria)[1])],:]
        self.selected_path = self.evolved_traj[np.where(self.crossing == self.criteria)[0][np.argmin(np.where(self.crossing == self.criteria)[1])],:]
        if not self.stationary:
            self.shortest_path = np.where(self.crossing==self.criteria)[1].min() * self.jump**2 + (1-int(self.gaussian)-int(self.stationary))*ct * self.jump**2
            self.shortest_time = np.where(self.crossing == self.criteria)[1].min() + (1-int(self.gaussian)-int(self.stationary))*ct
        else:
            self.shortest_path = np.where(self.crossing == self.criteria)[1].min() * self.jump ** 2
            self.shortest_time = np.where(self.crossing == self.criteria)[1].min()
        return self.shortest_path, self.shortest_time

    def SP_distribution(self):
        """
        Calculate the SP distribution for the given number of paths.
        Parameters:
            self (object): The object itself.
        
        Returns:
            None
        """
        self.SPs = np.zeros(self.num_paths)
        self.STs = np.zeros(self.num_paths)
        for i in tqdm(range(self.num_paths)):
            self.SPs[i], self.STs[i] = self.SP_evolution()

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
            data = np.genfromtxt('data/SP_data.dat',dtype=float)[:,:]
        else:
            data = np.genfromtxt('data/SP_data_poisson.dat', dtype=float)[:, :]
        SP = data[:,1] - (1-int(self.count_link))*data[:,-2]
        if self.count_link:
            np.savez('cl/1D_hist.npz',sp_brw = self.SPs, sp_cg = SP)
        else:
            np.savez('no_cl/1D_hist.npz',sp_brw = self.SPs, sp_cg = SP)
        bin_range = np.linspace(np.min(np.array([SP.min(),self.SPs.min()])),np.max(np.array([SP.max(),self.SPs.max()])),self.bins)
        plt.hist(self.SPs, bin_range, density=True, alpha = 0.3, label='Branched RW')
        if self.plot_ref:
            plt.hist(SP, bin_range, density=True, alpha=0.3, label='CGMD - 3D')
        
        plt.xlabel('Shortest Path (SP)')
        plt.ylabel('P(SP)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('CGMD_RW1D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
        plt.figure(2)
        plt.hist(self.STs, self.bins, density=True)
        plt.xlabel('First passage time (FPT)')
        plt.ylabel('P(FPT)')
        plt.tight_layout()
        if self.density_terminate:
            self.r_array = np.array(self.r_array)
            self.n_array = np.array(self.n_array)
            plt.figure(3)
            #plt.plot(self.r_array, self.n_array/4/np.pi/self.r_array**3)
            rs = np.arange(1,self.box)
            ns = np.array([self.count_nodes(i) for i in rs])
            plt.plot(rs,ns/2/rs)
            plt.xlabel('Distance')
            plt.ylabel('Particle count')
            plt.tight_layout()

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
        self.tau = np.zeros(len(self.boxes))
        self.sps = np.zeros(len(self.boxes))
        self.spsig = np.zeros(len(self.boxes))
        self.stsig = np.zeros(len(self.boxes))
        for i, box in enumerate(self.boxes):
            self.box = box
            self.SP_distribution()
            self.tau[i], self.sps[i], self.stsig[i], self.spsig[i] = self.STs.mean(), self.SPs.mean(), self.STs.std(), self.SPs.std()

        self.tau_by_x = self.tau / self.boxes
        self.sps_by_x = self.sps / self.boxes
        if plot_analysis:
            print(self.spsig, self.stsig)
            popt,_=curve_fit(objective_log,self.boxes,self.tau)
            a,b,e=popt
            if self.count_link:
                np.savez('cl/tau_1D%s%s%s%s_rate_%.4f.npz' % (
            self.add_purge, self.add_terminate, self.add_den_terminate, self.add_stationary, self.rate), box=self.boxes,
                     tau=self.tau, a=a, b=b, e=e)
            else:
                np.savez('no_cl/tau_1D%s%s%s%s_rate_%.4f.npz' % (
            self.add_purge, self.add_terminate, self.add_den_terminate, self.add_stationary, self.rate), box=self.boxes,
                     tau=self.tau, a=a, b=b, e=e)
            popt, _ = curve_fit(objective_lin, np.log(self.boxes), np.log(self.stsig))
            c, d = popt
            fig, ax1 = plt.subplots()
            ax1.plot(self.boxes, self.tau,'b--*',lw=3)
            ax1.plot(np.array([0,self.boxes.max()]), objective_log(np.array([0,self.boxes.max()]),a,b,e),'k--')
            ax1.set_title(r'$a=%.3f$, $b=%.3f$, and $c=%.3f$'%(a,b,e))
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
            fig.savefig('tau_x%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(2)
            plt.plot(self.boxes, self.tau_by_x,'b--*',lw=3)
            plt.xlabel('Box size (x)')
            plt.ylabel(r'$\tau(x)/x$')
            plt.savefig('tau_x_by_x%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(3)
            plt.plot(self.boxes, self.sps,'b--*',lw=3)
            plt.xlabel('Box size (x)')
            plt.ylabel(r'SP$(x)$')
            plt.savefig('SP_x%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(4)
            plt.plot(self.boxes, self.sps_by_x,'b--*')
            plt.xlabel('Box size (x)')
            plt.ylabel(r'SP$(x)/x$')
            plt.savefig('SP_x_by_x%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(5)
            plt.loglog(self.boxes, self.stsig, 'b--*', lw=3)
            plt.loglog(self.boxes, np.exp(d)*self.boxes**c, 'k--')
            plt.title(r'Scaling exponent$=%.3f$ ' %c)
            plt.xlabel('Box size (x)')
            plt.ylabel(r'$\sigma(\tau(x))$')
            plt.savefig('sig_tau_x%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.tight_layout()
            plt.show()

    def c_rho_analysis(self, plot_analysis = True):
        """
        Performs SP analysis for different box lengths and fit the mean of the SPs to the linear+log term at different branching rates/rho.
        Parameters:
        - plot_analysis (bool): Whether to plot the analysis or not. Defaults to True.
        Returns:
        - None
        """
        self.boxes = np.exp(np.linspace(np.log(self.box_min), np.log(self.box_max), self.log_points))
        if self.box_max == 4*self.box_min:
            self.boxes = np.array([0.25,0.5,0.75,1])*65.5028

        self.rho_arr = np.linspace((self.rho_min),(self.rho_max), self.rho_points)
        self.c_arr = np.zeros(len(self.rho_arr))
        if not self.terminate:
            ref_rho = np.linspace(1.01,1.99,40)
            ref_c1 = np.array([0.14083058929443362, 0.26125785064697266, 0.33870771026611324, 0.39917511749267576, 0.4496835098266601, 0.4933605117797851, 0.5320031356811523, 0.5666813583374023, 0.5981496505737306, 0.6269567184448243, 0.6534729385375977, 0.6780412521362306, 0.7008811416625977, 0.722198371887207, 0.7421438369750978, 0.7608684310913086, 0.7784818954467773, 0.795066535949707, 0.8107320938110352, 0.8255334396362306, 0.8395254440307617, 0.852790412902832, 0.8653557815551757, 0.8772764205932617, 0.8885797653198243, 0.8992932510375977, 0.9094717483520508, 0.9191289749145508, 0.9282786483764649, 0.9369482040405274, 0.9451650772094727, 0.9529292678833008, 0.9602544937133789, 0.9671407546997071, 0.9736017684936524, 0.9796375350952149, 0.9852206192016603, 0.9903373031616213, 0.9949189987182618, 0.9988148117065431])
        else:
            ref_rho = np.linspace(1.01,1.8,50)
            ref_c1 = np.array([0.0886304473876953, 0.1981876373291016, 0.2643482208251953, 0.3157373809814453, 0.35880836486816403, 0.39629936218261724, 0.4297052764892579, 0.4599437713623047, 0.48762168884277346, 0.5131830596923829, 0.5369387054443359, 0.5591550445556642, 0.5799948883056641, 0.5996506500244141, 0.6182111358642578, 0.635809555053711, 0.6525347137451172, 0.668445816040039, 0.6836316680908204, 0.6981366729736328, 0.7120200347900391, 0.7253113555908204, 0.7380550384521485, 0.7502806854248047, 0.7620475006103516, 0.7733554840087891, 0.7842342376708984, 0.7947281646728516, 0.8048224639892579, 0.8145615386962891, 0.823960189819336, 0.8330332183837891, 0.8417954254150392, 0.8502616119384767, 0.8584317779541017, 0.8663355255126953, 0.8739728546142578, 0.8813585662841797, 0.8885074615478517, 0.8954047393798829, 0.9020948028564453, 0.9085480499267578, 0.9147940826416017, 0.9208329010009767, 0.926664505004883, 0.9323036956787111, 0.9377652740478517, 0.9430196380615234, 0.9480963897705078, 0.9529955291748048])
        for j,rho in enumerate(self.rho_arr):
            self.rate = (rho-1.0)/2.0
            self.jump_rate = int(1 / self.rate)
            self.tau = np.zeros(len(self.boxes))
            self.sps = np.zeros(len(self.boxes))
            self.spsig = np.zeros(len(self.boxes))
            self.stsig = np.zeros(len(self.boxes))
            for i, box in enumerate(self.boxes):
                self.box = box
                self.SP_distribution()
                self.tau[i], self.sps[i], self.stsig[i], self.spsig[i] = self.STs.mean(), self.SPs.mean(), self.STs.std(), self.SPs.std()

            popt, _ = curve_fit(objective_log, self.boxes, self.tau)
            a, b, e = popt
            self.c_arr[j] = 1.0/a
        if not self.count_link:
            if not self.terminate:
                np.savez('no_cl/%s1D_C1_rho.npz'%('' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr)
            else:
                np.savez('no_cl/%sjust_term_1D_C1_rho.npz'%('' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr)
        else:
            if not self.terminate:
                np.savez('cl/%sadj_1D_C1_rho.npz'%('' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr)
            else:
                np.savez('cl/%sterm_1D_C1_rho.npz'%('' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr)
        if plot_analysis:
            plt.figure(1)
            plt.plot(self.rho_arr, self.c_arr,'b--*',lw=3,label='1D BRW')
            plt.plot(ref_rho,ref_c1,'k--o',lw=3,label='Reference')
            plt.legend()
            plt.xlabel(r'$\rho$', fontsize=12)
            plt.ylabel(r'$C_1$', fontsize=12)
            plt.savefig('C1_vs_rho_1D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.show()

class Branched_2D_Walkers():
    def __init__(self, rate = 0.01, jump = 1.0, box = 100.0, num_paths = 2000, bins = 100,
                 box_min = 60, box_max = 200, log_points = 10, pooling = False,
                 lattice = False, purge = True, terminate = True, gaussian = False, plot_ref = False,
                 density_terminate = False, rho_min = 0.1, rho_max = 0.9, rho_points = 6,
                 stationary = False, parallel = False, count_link = False):
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
        self.terminate  = terminate
        self.rho_min    = rho_min
        self.rho_max    = rho_max
        self.rho_points = rho_points
        self.parallel   = parallel
        self.plot_ref   = plot_ref
        self.density_terminate = density_terminate
        self.density_threshold = 1 / np.pi / 0.25 ** 2
        self.n_array = []
        self.r_array = []
        self.n_ref = 0
        self.r_ref = 0
        self.count_link = count_link
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
            self.lat_jump = np.array([[-self.jump,self.jump],[self.jump,self.jump],[self.jump,-self.jump],[-self.jump,-self.jump]])/2**0.5

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
            return 2*np.pi*np.random.rand(r,self.jump_rate)
        else:
            return 2 * np.pi * np.random.rand(r,1)

    def branch_choice(self):
        """
        Generates a random branch choice based on the current settings.

        Returns:
            numpy.ndarray: A random branch choice generated based on the current settings.
        """
        if not self.stationary:
            if self.gaussian:
                brancher = np.random.multivariate_normal(np.zeros(2),self.jump*np.identity(2),(3*self.rows,self.jump_rate))
            elif self.lattice:
                brancher = np.array([[random.choice(self.lat_jump) for i in range(self.jump_rate)] for j in range(3*self.rows)])
            else:
                brancher = np.zeros((3*self.rows,self.jump_rate,2))
                ang = self.get_angles(3*self.rows)
                brancher[:,:,0], brancher[:,:,1] = self.jump*np.cos(ang), self.jump*np.sin(ang)
        else:
            if self.gaussian:
                brancher = np.random.multivariate_normal(np.zeros(2),self.jump*np.identity(2),(self.rows + 2 * self.row_add,))
            elif self.lattice:
                brancher = np.array([[random.choice(self.lat_jump) for i in range(1)] for j in range(self.rows + 2 * self.row_add)])
            else:
                brancher = np.zeros((self.rows + 2 *(1-int(self.count_link))* self.row_add,2))
                ang = self.get_angles(self.rows + 2 *(1-int(self.count_link))* self.row_add).reshape(-1,)
                brancher[:,0], brancher[:,1] = self.jump*np.cos(ang), self.jump*np.sin(ang)
        return brancher

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
        ct_ind = np.where(evolved_pos[:,:,0]**2+evolved_pos[:,:,1]**2<r_in**2)[0]
        #ct_ind = np.where(self.final_loc[:, 0] ** 2 + self.final_loc[:, 1] ** 2 < r_in ** 2)[0]
        return len(ct_ind)

    def calc_front(self):
        """
        Calculate the front of the trajectory.

        This function takes no parameters.

        Returns:
            None
        """
        evolved_pos = self.traj.cumsum(1)
        self.final_loc = evolved_pos[:,-1,:]
        self.r_front = 3 * self.traj.shape[1] ** 0.5

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
            measured_density = (self.n_ref+new_counts)/np.pi/r0**2
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
            del_ind = np.where(self.final_loc[:,0]**2+self.final_loc[:,1]**2<self.r_ref**2)[0]
            self.traj = np.delete(self.traj,del_ind[:-1],axis=0)
        '''
        self.calc_front()
        del_ind = np.where(self.final_loc[:,0]**2+self.final_loc[:,1]**2<self.r_front**2)[0]
        if len(del_ind) > 0:
            self.traj = np.delete(self.traj, del_ind[:-1], axis=0)
        '''

    def crossed_barrier(self):
        """
        Calculates if the termination criteria is met by the evolved trajectory.

        Parameters:
            self (object): The instance of the class.
        
        Returns:
            bool: True if the termination criteria is met, False otherwise.
        """
        self.evolved_traj = self.traj.cumsum(1)
        self.crossing = ((self.evolved_traj[:,:,0]-self.box)**2+self.evolved_traj[:,:,1]**2<1).astype(int)
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
                self.traj = np.random.multivariate_normal(np.zeros(2),self.jump*np.identity(2),(1,self.jump_rate))
            elif self.lattice:
                self.traj = np.array([[random.choice(self.lat_jump) for i in range(self.jump_rate)] for j in range(1)])
            else:
                self.traj = np.zeros((1,self.jump_rate,2))
                ang = self.get_angles(1)
                self.traj[:,:,0], self.traj[:,:,1] = self.jump*np.cos(ang), self.jump*np.sin(ang)
        else:
            self.ctlk = np.zeros(1)
            if self.gaussian:
                self.traj = np.random.multivariate_normal(np.zeros(2),self.jump*np.identity(2),(1,1))
            elif self.lattice:
                self.traj = np.array([[random.choice(self.lat_jump) for i in range(1)] for j in range(1)])
            else:
                self.traj = np.zeros((1,1,2))
                ang = self.get_angles(self.traj.shape[0])
                self.traj[:,:,0], self.traj[:,:,1] = self.jump*np.cos(ang), self.jump*np.sin(ang)

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
            new_traj = np.zeros((3*self.rows,self.cols+self.jump_rate,2))
            new_traj[:self.rows,:self.cols,:] = self.traj.copy()
            new_traj[self.rows:2*self.rows, :self.cols,:] = self.traj.copy()
            new_traj[2*self.rows:3 * self.rows, :self.cols, :] = self.traj.copy()
            new_traj[:,self.cols:,:] = self.branch_choice()
            self.traj = new_traj.copy()
        else:
            if not self.count_link:
                p_val = np.random.rand(self.rows)
            else:
                c_ind = np.where(self.ctlk == 0)[0]
                p_val = np.random.rand(len(c_ind))
            b_ind = np.where(p_val < self.rate)[0]
            self.row_add = len(b_ind)
            new_ct = np.zeros(self.rows + 2 * self.row_add)
            new_ct[self.rows:self.rows + self.row_add] += 1
            new_ct[self.rows + self.row_add:self.rows + 2 * self.row_add] += 1
            new_traj = np.zeros((self.rows + 2 * self.row_add, self.cols + 1,2))
            jump_ang = self.get_angles(self.row_add).reshape(-1,)
            jump_add = np.zeros((self.row_add,2))
            jump_add[:,0], jump_add[:,1] = int(self.count_link)*self.jump*np.cos(jump_ang), int(self.count_link)*self.jump*np.sin(jump_ang)
            new_traj[:self.rows, :self.cols, :] = self.traj.copy()
            if not self.count_link:
                new_traj[self.rows:self.rows + self.row_add, :self.cols, :] = self.traj[b_ind, :].copy()
                new_traj[self.rows + self.row_add:self.rows + 2 * self.row_add, :self.cols, :] = self.traj[b_ind, :].copy()
            else:
                new_traj[self.rows:self.rows + self.row_add, :self.cols, :] = self.traj[c_ind[b_ind], :].copy()
                new_traj[self.rows + self.row_add:self.rows + 2 * self.row_add, :self.cols, :] = self.traj[c_ind[b_ind],:].copy()
            new_traj[:self.rows+2*(1-int(self.count_link))*self.row_add, -1,:] = self.branch_choice()
            if self.count_link:
                new_traj[self.rows:self.rows + self.row_add, -1, :] = jump_add.copy()
                new_traj[self.rows + self.row_add:self.rows + 2 * self.row_add, -1, :] = jump_add.copy()
                self.ctlk = new_ct.copy()
            self.traj = new_traj.copy()


    def purge_traj(self):
        """
        Purges the trajectory if the number of paths in the trajectory is greater than 9000.
        Retain the forwardmost 3^7 paths.
        Updates the trajectory array with the sorted and trimmed array.
        If the 'stationary' flag is True and the 'count_link' flag is True, also sorts and trims the 'ctlk' array.
        """
        if self.traj.shape[0] > 9000:
            traversal = self.traj[:,:,0].cumsum(1)
            end_p = traversal[:,-1]
            arranged_traj = self.traj[np.argsort(end_p),:,:].copy()
            self.traj = arranged_traj[-2187:,:,:].copy()
            if self.stationary:
                if self.count_link:
                    arranged_ct = self.ctlk[np.argsort(end_p)].copy()
                    self.ctlk = arranged_ct[-2187:].copy()

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
            dind = np.where(p_val < 2.0 / 500.0)[0]
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
        while should_restart:
            should_restart = False
            self.init_walk()
            ct = 0
            while self.crossed_barrier() is False:
                ct += 1
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
                    if ct > 15000:
                        should_restart = True
                        break


        self.selected_choices = self.traj[np.where(self.crossing==self.criteria)[0][np.argmin(np.where(self.crossing==self.criteria)[1])],:]
        self.selected_path = self.evolved_traj[np.where(self.crossing == self.criteria)[0][np.argmin(np.where(self.crossing == self.criteria)[1])],:]
        if not self.stationary:
            self.shortest_path = np.where(self.crossing==self.criteria)[1].min() * self.jump**2 + (1-int(self.gaussian)-int(self.stationary))*ct * self.jump**2
            self.shortest_time = np.where(self.crossing == self.criteria)[1].min() + (1-int(self.gaussian)-int(self.stationary))*ct
        else:
            self.shortest_path = np.where(self.crossing == self.criteria)[1].min() * self.jump ** 2
            self.shortest_time = np.where(self.crossing == self.criteria)[1].min()
        return self.shortest_path, self.shortest_time

    def SP_distribution(self):
        """
        Calculate the SP distribution for the given number of paths.
        Parameters:
            self (object): The object itself.
        
        Returns:
            None
        """
        self.SPs = np.zeros(self.num_paths)
        self.STs = np.zeros(self.num_paths)
        for i in tqdm(range(self.num_paths)):
            self.SPs[i], self.STs[i] = self.SP_evolution()

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
            data = np.genfromtxt('data/SP_data.dat', dtype=float)[:, :]
        else:
            data = np.genfromtxt('data/SP_data_poisson.dat', dtype=float)[:, :]
        SP = data[:, 1] - (1-int(self.count_link))*data[:,-2]
        if self.count_link:
            np.savez('cl/2D_hist.npz',sp_brw = self.SPs, sp_cg = SP)
        else:
            np.savez('no_cl/2D_hist.npz',sp_brw = self.SPs, sp_cg = SP)
        bin_range = np.linspace(np.min(np.array([SP.min(), self.SPs.min()])), np.max(np.array([SP.max(), self.SPs.max()])),
                                self.bins)
        plt.hist(self.SPs, bin_range, density=True, alpha=0.3, label='Branched RW')
        if self.plot_ref:
            plt.hist(SP, bin_range, density=True, alpha=0.3, label='CGMD - 3D')
        
        plt.xlabel('Shortest Path (SP)')
        plt.ylabel('P(SP)')
        plt.legend()
        plt.tight_layout()
        if self.lattice:
            plt.savefig('CGMD_RW2D_lattice%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
        else:
            plt.savefig('CGMD_RW2D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
        plt.figure(2)
        plt.hist(self.STs, self.bins, density=True)
        plt.xlabel('First passage time (FPT)')
        plt.ylabel('P(FPT)')
        plt.tight_layout()
        schematic = False
        if schematic:
            final_time = np.min(np.where(self.crossing == self.criteria)[1])
            cc = np.arange(self.traj.shape[0],dtype=int)
            plt.figure(3)
            plt.plot(self.selected_path[:,0],self.selected_path[:,1],'r--',lw=3,label='Shortest Path')
            plt.plot(self.selected_path[0, 0], self.selected_path[0, 1], 'bo', lw=3, label=r'$p$')
            plt.plot(self.selected_path[-1, 0], self.selected_path[-1, 1], 'ko', lw=3, label=r'$p^{\prime}$')
            for i in range(20):
                plt.plot(self.evolved_traj[random.choice(cc),:final_time,0],self.evolved_traj[random.choice(cc),:final_time,1],'k--')

            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')

        if self.density_terminate:
            self.r_array = np.array(self.r_array)
            self.n_array = np.array(self.n_array)
            plt.figure(3)
            #plt.plot(self.r_array, self.n_array/4/np.pi/self.r_array**3)
            rs = np.arange(1, self.box)
            ns = np.array([self.count_nodes(i) for i in rs])
            plt.plot(rs, ns / np.pi / rs**2)
            plt.xlabel('Distance')
            plt.ylabel('Particle count')
            plt.tight_layout()
        plt.show()

    def analyze_rate(self):
        """
        Analyze the rate by performing the following steps:
        
        1. Call the SP_distribution() method to calculate the distribution of SPs.
        2. Call the plot_and_fit_SPs() method to plot and fit the SPs.
        """
        self.SP_distribution()
        self.plot_and_fit_SPs()

    def tau_SP_analysis(self, plot_analysis=True):
        """
        Performs SP analysis for different box lengths and fit the mean of the SPs to the linear+log term.
        Parameters:
        - plot_analysis (bool): Whether to plot the analysis or not. Defaults to True.
        Returns:
        - None
        """
        self.boxes = np.exp(np.linspace(np.log(self.box_min), np.log(self.box_max), self.log_points))
        if self.box_max == 4 * self.box_min:
            self.boxes = np.array([0.25, 0.5, 0.75, 1]) * 65.5028
        self.tau = np.zeros(len(self.boxes))
        self.sps = np.zeros(len(self.boxes))
        self.spsig = np.zeros(len(self.boxes))
        self.stsig = np.zeros(len(self.boxes))
        for i, box in enumerate(self.boxes):
            self.box = box
            self.SP_distribution()
            self.tau[i], self.sps[i], self.stsig[i], self.spsig[
                i] = self.STs.mean(), self.SPs.mean(), self.STs.std(), self.SPs.std()

        self.tau_by_x = self.tau / self.boxes
        self.sps_by_x = self.sps / self.boxes
        if plot_analysis:
            popt, _ = curve_fit(objective_log, self.boxes, self.tau)
            a, b, e = popt
            if self.count_link:
                np.savez('cl/tau_2D%s%s%s%s_rate_%.4f.npz' % (
            self.add_purge, self.add_terminate, self.add_den_terminate, self.add_stationary, self.rate), box=self.boxes,
                     tau=self.tau, a=a, b=b, e=e)
            else:
                np.savez('no_cl/tau_2D%s%s%s%s_rate_%.4f.npz' % (
            self.add_purge, self.add_terminate, self.add_den_terminate, self.add_stationary, self.rate), box=self.boxes,
                     tau=self.tau, a=a, b=b, e=e)
            popt, _ = curve_fit(objective_lin, np.log(self.boxes), np.log(self.stsig))
            c, d = popt
            fig, ax1 = plt.subplots()
            ax1.plot(self.boxes, self.tau, 'b--*', lw=3)
            ax1.plot(np.array([0, self.boxes.max()]), objective_log(np.array([0, self.boxes.max()]), a, b, e), 'k--')
            ax1.set_title(r'$a=%.3f$, $b=%.3f$, and $c=%.3f$' % (a, b, e))
            ax1.set_xlabel('Box size (x)')
            ax1.set_ylabel(r'$\tau(x)$')
            left, bottom, width, height = 0.5, 0.2, 0.25, 0.25
            ax2 = fig.add_axes([left, bottom, width, height])
            ax2.plot(self.boxes, self.tau, 'b--*', lw=3)
            ax2.plot(np.array([0, self.boxes.max()]), objective_lin(np.array([0, self.boxes.max()]), a, b), 'k--')
            ax2.set_xlabel('Box size (x)')
            ax2.set_ylabel(r'$\tau(x)$')
            ax2.set_xlim([0, 15])
            ax2.set_ylim([b - 5, b + 45])
            fig.savefig('tau_x_2D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(2)
            plt.plot(self.boxes, self.tau_by_x, 'b--*', lw=3)
            plt.xlabel('Box size (x)')
            plt.ylabel(r'$\tau(x)/x$')
            plt.savefig('tau_x_by_x_2D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(3)
            plt.plot(self.boxes, self.sps, 'b--*', lw=3)
            plt.xlabel('Box size (x)')
            plt.ylabel(r'SP$(x)$')
            plt.savefig('SP_x_2D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(4)
            plt.plot(self.boxes, self.sps_by_x, 'b--*')
            plt.xlabel('Box size (x)')
            plt.ylabel(r'SP$(x)/x$')
            plt.savefig('SP_x_by_x_2D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.figure(5)
            plt.loglog(self.boxes, self.stsig, 'b--*', lw=3)
            plt.loglog(self.boxes, np.exp(d) * self.boxes ** c, 'k--')
            plt.title(r'Scaling exponent$=%.3f$ ' % c)
            plt.xlabel('Box size (x)')
            plt.ylabel(r'$\sigma(\tau(x))$')
            plt.savefig('sig_tau_x_2D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
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

        self.rho_arr = np.linspace((self.rho_min), (self.rho_max), self.rho_points)
        self.c_arr = np.zeros(len(self.rho_arr))
        if not self.terminate:
            ref_rho = np.linspace(1.1,2.9,18)
            ref_c1 = np.array(
            [0.304599609375, 0.42206054687500005, 0.503525390625, 0.564150390625, 0.613408203125, 0.653193359375,
             0.6872949218750001, 0.715712890625, 0.7422363281250001, 0.764970703125, 0.783916015625,
             0.8009667968750002,
             0.8161230468750001, 0.8293847656250001, 0.8426464843750001, 0.8540136718750001, 0.863486328125,
             0.8729589843750001])
        else:
            ref_rho = np.linspace(1.01,2.99,20)
            ref_c1 = np.array([0.06315788269042968, 0.3163738250732422, 0.42806236267089853, 0.5058121490478517, 0.5654602813720704, 0.6134600067138672, 0.6532451629638673, 0.6868878936767578, 0.7157498931884766, 0.7407784271240235, 0.7626987457275392, 0.7820288848876954, 0.7991536712646485, 0.8144283294677735, 0.8281044769287111, 0.8403745269775391, 0.8514456939697266, 0.8614511871337891, 0.8705094146728516, 0.8787387847900391])
        for j, rho in enumerate(self.rho_arr):
            self.rate = (rho - 1.0) / 2.0
            self.jump_rate = int(1 / self.rate)
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
            self.c_arr[j] = 1.0 / a
        if not self.count_link:
            if not self.terminate:
                np.savez('no_cl/%s2D_C1_rho.npz'%('' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr)
            else:
                np.savez('no_cl/%sjust_term_2D_C1_rho.npz'%('' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr)
        else:
            if not self.terminate:
                np.savez('cl/%sadj_2D_C1_rho.npz'%('' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr)
            else:
                np.savez('cl/%sterm_2D_C1_rho.npz'%('' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr)
        if plot_analysis:
            plt.figure(1)
            plt.plot(self.rho_arr, self.c_arr, 'b--*', lw=3,label='2D BRW')
            plt.plot(ref_rho, ref_c1, 'k--o', lw=3, label='Reference')
            plt.legend()
            plt.xlabel(r'$\rho$', fontsize=12)
            plt.ylabel(r'$C_1$', fontsize=12)
            plt.savefig('C1_vs_rho_2D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.show()

class Branched_3D_Walkers():
    def __init__(self, rate = 0.01, jump = 1.0, box = 100.0, num_paths = 2000, bins = 100,
                 box_min = 60, box_max = 200, log_points = 10, pooling = False,
                 lattice = False, purge = True, terminate = True, gaussian = False, plot_ref = False,
                  rho_min = 0.1, rho_max = 0.9, rho_points = 6, stationary = False, density_terminate = False,
                 parallel = False, count_link = False, correlated = False, correlation = 8.0/28.0, time_discrete = 1.0):
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
        self.n_ref = 0
        self.r_ref = 0
        self.count_link = count_link
        self.correlated = correlated
        if self.gaussian:
            self.add_bbm = 'bbm_'
            self.add_jump = ''
            self.dt         = 1.0/time_discrete
            self.change_jump = False
            if jump!=1:
                data = np.genfromtxt('data/CGMD_MSID.txt')
                self.n_msid, self.msid = data[:,0], data[:,1]
                self.change_jump = True if rho_max <2 else False
        else:
            self.add_bbm = ''
            self.dt         = 1.0
            if jump == 1:
                self.add_jump = ''
                self.change_jump = False
            else:
                data = np.genfromtxt('data/CGMD_MSID.txt')
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
                brancher = np.random.multivariate_normal(np.zeros(3), self.jump**2 * self.dt * np.identity(3)/3.0,(self.rows + 2 *(1-int(self.count_link))* self.row_add,)) 
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
                self.traj = np.random.multivariate_normal(np.zeros(3),self.jump**2*self.dt*np.identity(3)/3.0,(1,1)) 
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
                jump_add = np.random.multivariate_normal(np.zeros(3), self.jump**2 *self.dt* np.identity(3)/3.0,(self.row_add,)) 
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
        if self.traj.shape[0] > 9000:
            traversal = self.traj[:,:,0].cumsum(1)
            end_p = traversal[:,-1]
            arranged_traj = self.traj[np.argsort(end_p),:,:].copy()
            self.traj = arranged_traj[-2187:,:,:].copy()
            if self.stationary:
                if self.count_link:
                    arranged_ct = self.ctlk[np.argsort(end_p)].copy()
                    self.ctlk = arranged_ct[-2187:].copy()

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
                        lim = 100000
                    else:
                        lim = 25000
                    if ct > (lim):
                        should_restart = True
                        break

        self.selected_choices = self.traj[np.where(self.crossing==self.criteria)[0][np.argmin(np.where(self.crossing==self.criteria)[1])],:]
        self.selected_path = self.evolved_traj[np.where(self.crossing == self.criteria)[0][np.argmin(np.where(self.crossing == self.criteria)[1])],:]
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
            data = np.genfromtxt('data/SP_data.dat', dtype=float)[:, :]
        else:
            data = np.genfromtxt('data/SP_data_poisson.dat', dtype=float)[:, :]
        SP = data[:, 1] - (1-int(self.count_link))*data[:,-2]
        if self.count_link:
            np.savez('cl/%s%s%s3D_hist.npz'%(self.add_jump,self.add_bbm,self.add_corr),sp_brw = self.STs, sp_cg = SP, sp_brw_dist = self.SPs)
        else:
            np.savez('no_cl/%s%s%s3D_hist.npz'%(self.add_jump,self.add_bbm,self.add_corr),sp_brw = self.STs, sp_cg = SP, sp_brw_dist = self.SPs)
        bin_range = np.linspace(np.min(np.array([SP.min(), self.STs.min()])), np.max(np.array([SP.max(), self.STs.max()])),
                                self.bins)
        plt.hist(self.STs, bin_range, density=True, alpha=0.3, label='Branched RW')
        if self.plot_ref:
            plt.hist(SP, bin_range, density=True, alpha=0.3, label='CGMD - 3D')
        
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
            plt.figure(3)
            plt.plot(self.selected_path[:,0],self.selected_path[:,1],'r--',lw=3,label='Shortest Path')
            plt.plot(self.selected_path[0, 0], self.selected_path[0, 1], 'bo', lw=3, label=r'$p$')
            plt.plot(self.selected_path[-1, 0], self.selected_path[-1, 1], 'ko', lw=3, label=r'$p^{\prime}$')
            for i in range(2):
                plt.plot(self.evolved_traj[random.choice(cc),:final_time,0],self.evolved_traj[random.choice(cc),:final_time,1],'k--')
            print(self.evolved_traj.shape)
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
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
        self.tau = np.zeros(len(self.boxes))
        self.sps = np.zeros(len(self.boxes))
        self.spsig = np.zeros(len(self.boxes))
        self.stsig = np.zeros(len(self.boxes))
        sps = []
        for i, box in enumerate(self.boxes):
            self.box = box
            self.SP_distribution()
            sps.append(self.STs)
            self.tau[i], self.sps[i], self.stsig[i], self.spsig[i] = self.STs.mean(), self.SPs.mean(), self.STs.std(), self.SPs.std()
        if self.count_link:
            np.savez('cl/%s%s%sSP_brw.npz'%(self.add_jump,self.add_bbm,self.add_corr),sp10 = sps[0], sp15 = sps[1], sp20 = sps[2], sp25 = sps[3],sp50 = sps[4], sp75=sps[5], sp100 = sps[6],box=self.boxes)
        else:
            np.savez('no_cl/%s%s%sSP_brw.npz'%(self.add_jump,self.add_bbm,self.add_corr),sp10 = sps[0], sp15 = sps[1], sp20 = sps[2], sp25 = sps[3],sp50 = sps[4], sp75=sps[5], sp100 = sps[6],box=self.boxes)
        self.tau_by_x = self.tau / self.boxes
        self.sps_by_x = self.sps / self.boxes
        if plot_analysis:
            popt,_=curve_fit(objective_log,self.boxes,self.tau)
            a,b,e=popt
            if self.count_link:
                np.savez('cl/%s%s%stau_3D%s%s%s%s_rate_%.4f.npz'%(self.add_jump,self.add_bbm,self.add_corr,self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary, self.rate),box =self.boxes, tau = self.tau, a=a,b=b,e=e)
            else:
                np.savez('no_cl/%s%s%stau_3D%s%s%s%s_rate_%.4f.npz'%(self.add_jump,self.add_bbm,self.add_corr,self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary, self.rate),box =self.boxes, tau = self.tau, a=a,b=b,e=e)
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
        self.c_arr = np.zeros(len(self.rho_arr))
        self.c_arr_bar = np.zeros(len(self.rho_arr))
        self.sig_arr = np.zeros(len(self.rho_arr))
        if not self.count_link:
            if not self.terminate:
                if not self.gaussian:
                    ref_rho = np.linspace(1.001,2.99,40)
                    ref_c1 = np.array(
                    [0.02581489562988281, 0.18289817810058595, 0.25312904357910154, 0.30454780578613283, 0.3459314727783203, 0.3807878875732422, 0.41095237731933604, 0.4375498199462891, 0.4613202667236328, 0.4827669525146485, 0.5022895050048829, 0.5201839447021485, 0.5366574859619141, 0.5519173431396484, 0.5660967254638674, 0.5793140411376954, 0.5916876983642578, 0.6033065032958984, 0.6142444610595703, 0.6245459747314452, 0.6342850494384766, 0.6435208892822265, 0.6522682952880859, 0.6605716705322265, 0.668490219116211, 0.676023941040039, 0.6832172393798829, 0.6900997161865234, 0.6966713714599609, 0.7029618072509767, 0.708985824584961, 0.7147730255126953, 0.7203382110595703, 0.725681381225586, 0.7308173370361327, 0.7357756805419923, 0.7405416107177736, 0.7451299285888673, 0.7495702362060548, 0.7538477325439453])
                else:
                    ref_rho = np.linspace(1.1,2.9,19)
                    ref_c1 = np.array([0.31622776601683794, 0.4472135954999579, 0.5477225575051661, 0.6324555320336759, 0.7071067811865476, 0.7745966692414834, 0.8366600265340756, 0.8944271909999159, 0.9486832980505138, 0.9999999999999999, 1.0488088481701514, 1.0954451150103321, 1.140175425099138, 1.1832159566199232, 1.224744871391589, 1.2649110640673518, 1.3038404810405297, 1.3416407864998738, 1.378404875209022])
            else:
                ref_rho = np.linspace(1.01,2.99,50)
                ref_c1 = np.array([0.051183853149414066, 0.1688964080810547, 0.23036506652832034, 0.27632225036621094, 0.31381324768066404, 0.3457538604736328, 0.37366859436035155, 0.3984899139404297, 0.4208542633056641, 0.4411760711669922, 0.45981056213378907, 0.4769945526123047, 0.49292045593261724, 0.5077510833740234, 0.5216048431396484, 0.5346001434326172, 0.5468257904052735, 0.558370590209961, 0.569264144897461, 0.5795952606201172, 0.5893935394287111, 0.5987033843994141, 0.6075839996337891, 0.6160501861572266, 0.6241463470458983, 0.6318724822998046, 0.6392877960205077, 0.6463774871826172, 0.6532007598876953, 0.6597428131103515, 0.6660332489013672, 0.6720868682861327, 0.6779332733154297, 0.6835576629638673, 0.6889748382568359, 0.6942144012451172, 0.6992763519287111, 0.7041754913330077, 0.7088970184326172, 0.713485336303711, 0.7179256439208984, 0.7222179412841797, 0.7263918304443359, 0.7304325103759766, 0.7343547821044922, 0.7381734466552735, 0.7418737030029298, 0.7454703521728516, 0.7489633941650391, 0.7523676300048829])
        else:
            if not self.terminate:
                ref_rho = np.linspace(1.001, 2.99, 100)
                ref_c1 = np.array([0.025800094604492185, 0.11678199768066404, 0.16063743591308594, 0.1930368804931641, 0.21929389953613285, 0.24156944274902345, 0.26098838806152347, 0.2782315826416015, 0.2937430572509766, 0.3078632354736328, 0.3208289337158203, 0.3327881622314453, 0.3439037322998047, 0.35427925109863284, 0.36398872375488284, 0.3731505584716797, 0.38177955627441407, 0.3899497222900391, 0.3977054595947266, 0.4050763702392579, 0.41210685729980473, 0.41882652282714855, 0.42525016784667974, 0.43140739440917975, 0.43732780456542975, 0.44301139831542974, 0.4484729766845703, 0.4537421417236328, 0.4588188934326172, 0.4637328338623047, 0.46846916198730476, 0.4730574798583985, 0.477497787475586, 0.48180488586425785, 0.48597877502441406, 0.49003425598144534, 0.49397132873535154, 0.49780479431152347, 0.501534652709961, 0.5051461029052735, 0.5086835479736328, 0.5121173858642578, 0.5154772186279297, 0.5187482452392578, 0.5219304656982422, 0.5250534820556642, 0.5280876922607423, 0.5310626983642578, 0.533978500366211, 0.5368202972412109, 0.5396028900146485, 0.5423262786865234, 0.5449904632568359, 0.547595443725586, 0.5501560211181641, 0.5526573944091796, 0.5551291656494141, 0.557541732788086, 0.5598950958251954, 0.5622188568115235, 0.5644982147216797, 0.5667331695556642, 0.5689385223388673, 0.5710846710205078, 0.5732160186767579, 0.5753029632568359, 0.5773455047607423, 0.5793584442138673, 0.5813417816162111, 0.5832955169677735, 0.5852048492431642, 0.5870993804931641, 0.5889495086669924, 0.5907848358154297, 0.5925757598876954, 0.5943518829345703, 0.5960984039306642, 0.5978153228759766, 0.5995174407958984, 0.601189956665039, 0.6028328704833984, 0.6044609832763673, 0.6060594940185547, 0.6076432037353516, 0.6091973114013672, 0.6107366180419923, 0.612246322631836, 0.613741226196289, 0.6152213287353516, 0.6166718292236327, 0.618122329711914, 0.6195432281494141, 0.6209345245361327, 0.6223258209228516, 0.6236875152587891, 0.6250344085693359, 0.6263813018798828, 0.6276985931396484, 0.6290010833740234, 0.6302887725830078])
            else:
                ref_rho = np.linspace(1.01, 2.99, 50)
                ref_c1 = np.array([0.051124649047851564, 0.16535896301269531, 0.2220616912841797, 0.2628533172607422, 0.2950603485107422, 0.3217761993408203, 0.3445845794677735, 0.36447715759277344, 0.3821199798583985, 0.39791267395019536, 0.41222526550292976, 0.42529457092285156, 0.4372982025146485, 0.44838417053222657, 0.45868568420410155, 0.4683063507080079, 0.47729057312011725, 0.4857567596435547, 0.49371971130371095, 0.5012534332275391, 0.5083875274658204, 0.5151515960693359, 0.5216048431396484, 0.5277472686767578, 0.5336232757568359, 0.5392328643798828, 0.5446056365966796, 0.5497711944580077, 0.5547295379638673, 0.5595102691650391, 0.5640985870361328, 0.5685240936279298, 0.5728015899658204, 0.5769310760498048, 0.5809273529052735, 0.5847904205322266, 0.5885350799560548, 0.5921613311767578, 0.595669174194336, 0.5990882110595703, 0.6024036407470703, 0.6056302642822265, 0.608768081665039, 0.6118170928955078, 0.6147920989990234, 0.6176782989501953, 0.6205052947998047, 0.623258285522461, 0.6259372711181641, 0.6285570526123047])
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

            if self.rho_max > 1.5:
                popt, _ = curve_fit(objective_log, self.boxes, self.tau)
                a, b, e = popt
            else:
                popt, _ = curve_fit(objective_lin, self.boxes, self.tau)
                a,b = popt
            self.c_arr[j] = 1.0 / a
            self.c_arr_bar[j] = self.boxes[-1]/self.tau[-1]
            self.sig_arr[j] = self.spsig[-1]
            #else:
            #self.SP_distribution()
            #self.tau, self.sps, self.stsig, self.spsig = self.STs.mean(), self.SPs.mean(), self.STs.std(), self.SPs.std()
            #self.c_arr[j] = self.box/self.tau
            #self.c_arr[j] = 1.0 / a
            #self.sig_arr[j] = self.stsig
        if not self.count_link:
            if not self.terminate:
                np.savez('no_cl/%s%s%s%s3D_C1_rho.npz'%(self.add_jump,self.add_bbm,self.add_corr,'' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr, c1_bar = np.ones(4) if self.rho_max>1.5 else self.c_arr_bar)
            else:
                np.savez('no_cl/%s%s%s%sjust_term_3D_C1_rho.npz'%(self.add_jump,self.add_bbm,self.add_corr,'' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr, c1_bar = np.ones(4) if self.rho_max>1.5 else self.c_arr_bar)
        else:
            if not self.terminate:
                np.savez('cl/%s%s%s%sadj_3D_C1_rho.npz'%(self.add_jump,self.add_bbm,self.add_corr,'' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr, c1_bar = np.ones(4) if self.rho_max>1.5 else self.c_arr_bar)
            else:
                np.savez('cl/%s%s%s%sterm_3D_C1_rho.npz'%(self.add_jump,self.add_bbm,self.add_corr,'' if self.rho_max > 1.5 else 'small_'), rho_ref=ref_rho, c1_ref=ref_c1, rho_brw=self.rho_arr, c1_brw=self.c_arr, sig_brw = np.ones(4) if self.rho_max>1.5 else self.sig_arr, c1_bar = np.ones(4) if self.rho_max>1.5 else self.c_arr_bar)
        if plot_analysis:
            plt.figure(1)
            plt.plot(self.rho_arr, self.c_arr, 'b--*', lw=3,label='3D BRW')
            plt.plot(ref_rho, ref_c1, 'k--o', lw=3, label='Reference')
            plt.legend()
            plt.xlabel(r'$\rho$',fontsize=12)
            plt.ylabel(r'$C_1$',fontsize=12)
            plt.savefig('C1_vs_rho_3D%s%s%s%s.png'%(self.add_purge,self.add_terminate,self.add_den_terminate,self.add_stationary))
            plt.show()


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
    parser.add_argument("--time_discrete", type=float, default=3.0, help="discretization steps per unit time")
    parser.add_argument("--log_points", type=int, default=4, help="number of points on the log scale for deciding box sizes")
    parser.add_argument("--rho_points", type=int, default=9,
                        help="number of points on the log scale for deciding branching rates")
    args = parser.parse_args()
    if args.dim == 1:
        util = Branched_1D_Walkers(rate=args.rate, jump=args.jump, box=args.box, num_paths=args.num_paths,
                                   bins=args.bins, box_min=args.box_min, box_max=args.box_max,
                                   log_points=args.log_points, purge=args.purge, terminate=args.terminate,
                                   plot_ref=args.plot_ref, density_terminate=args.density_terminate,
                                   rho_min=args.rho_min, rho_max=args.rho_max, rho_points=args.rho_points,
                                   stationary=args.stationary, parallel=args.parallel,
                                   count_link=args.count_link,gaussian=args.gaussian)
    elif args.dim == 2:
        util = Branched_2D_Walkers(rate=args.rate, jump=args.jump, box=args.box, num_paths=args.num_paths,
                                   bins=args.bins, box_min=args.box_min, box_max=args.box_max,
                                   log_points=args.log_points, lattice=args.lattice, purge=args.purge, terminate=args.terminate,
                                   plot_ref=args.plot_ref, density_terminate=args.density_terminate,
                                   rho_min=args.rho_min, rho_max=args.rho_max, rho_points=args.rho_points,
                                   stationary=args.stationary, parallel=args.parallel,
                                   count_link=args.count_link,gaussian=args.gaussian)
    elif args.dim == 3:
        util = Branched_3D_Walkers(rate=args.rate, jump=args.jump, box=args.box, num_paths=args.num_paths,
                           bins=args.bins, box_min=args.box_min, box_max=args.box_max, time_discrete=args.time_discrete,
                           log_points=args.log_points, lattice=args.lattice, purge=args.purge, terminate=args.terminate,
                                   plot_ref=args.plot_ref, rho_min=args.rho_min, rho_max=args.rho_max, rho_points=args.rho_points,
                                   stationary=args.stationary, density_terminate=args.density_terminate, parallel=args.parallel,
                                   count_link=args.count_link, correlated=args.correlated, correlation=args.correlation,gaussian=args.gaussian)
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

