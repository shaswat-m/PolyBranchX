import sys, os
import numpy as np
import re, random
from tqdm import tqdm
import matplotlib.pyplot as plt

class Random_Walker():
    def __init__(self, nchain = 500, nbead = 1000, corr = False):
        self.nchain = nchain
        self.nbead = nbead
        if corr:
            self.correlation = 0.4
        else:
            self.correlation = 0
        self.corr_angle = np.arccos(self.correlation)
        self.corr = corr
        self.jump = 1
    
    def get_angles(self,r,first=False):
        """
        Generate random angles in radians.
        Parameters:
            r (int): The number of angles to generate.
        Returns:
            tuple: A tuple containing two arrays of angles in radians.
                   The first array represents the azimuthal angle (theta),
                   and the second array represents the polar angle (phi).
        """
        theta = 2 * np.pi * np.random.rand(r, 1)
        phi = np.arccos(1 - 2 * np.random.rand(r, 1))
        return theta, phi

    def branch(self):
        """
        Generates a random branch choice based on the current settings.

        Returns:
            numpy.ndarray: A random branch choice generated based on the current settings.
        """
        brancher = np.zeros((self.nchain,3))
        theta, phi = self.get_angles(self.nchain)
        theta = theta.reshape(-1,)
        phi = phi.reshape(-1, )
        brancher[:,0], brancher[:,1], brancher[:,2] = self.jump*np.sin(phi)*np.cos(theta), self.jump*np.sin(phi)*np.sin(theta), self.jump*np.cos(phi)
        return brancher
  
    def generate_traj(self):
        self.traj = np.zeros((self.nchain,self.nbead,3))
        theta, phi = self.get_angles(self.traj.shape[0], first=True)
        self.traj[:,0:1,0], self.traj[:,0:1,1], self.traj[:,0:1,2] = self.jump*np.sin(phi)*np.cos(theta), self.jump*np.sin(phi)*np.sin(theta), self.jump*np.cos(phi)
        for i in range(1,self.nbead):
            to_jump = self.correlation*self.traj[:,i-1,:]+(1-self.correlation**2)**0.5*self.branch() 
            #print(np.mean(np.linalg.norm(self.branch() ,axis=1)))
            #print(np.mean(np.linalg.norm(np.diag(self.jump / np.linalg.norm(to_jump,axis=1) ) @ to_jump ,axis=1)))
            self.traj[:,i,:] = np.diag(self.jump / np.linalg.norm(to_jump,axis=1) ) @ to_jump 
            #self.traj[:,i,:] = to_jump 
        self.evolved_traj = self.traj.cumsum(1)

    def compute_msid(self):
        self.n = np.arange(1,self.nbead)
        self.msid = np.zeros(self.nbead-1)
        self.r2 = np.zeros(self.nchain)
        for i in tqdm(range(1,self.nbead)):
            dist = []
            for j in range(self.nchain):
                dr = self.evolved_traj[j,:-i,:]-self.evolved_traj[j,i:,:]
                dist.extend(np.linalg.norm(dr,axis=1))
                if i == self.nbead-1:
                    self.r2[j] = self.evolved_traj[j,:-i,0]-self.evolved_traj[j,i:,0]
            dist = np.array(dist)**2
            self.msid[i-1] = dist.mean()/i

    def calculator(self):
        self.generate_traj()
        self.compute_msid()



Uncorr = Random_Walker(corr=False)
Uncorr.calculator()
Corr = Random_Walker(corr=True)
Corr.calculator()
data = np.genfromtxt('data/target_MSID.txt')[:,:]
ref_n, ref_msid = data[:,0], data[:,1]

plt.figure(1)
plt.semilogx(Uncorr.n,Uncorr.msid,lw=3,label = 'Uncorrelated random walk')
plt.semilogx(Corr.n,Corr.msid,lw=3,label = 'Correlated random walk')
plt.semilogx(ref_n,ref_msid,lw=3,label='CGMD')
plt.xlim([1,500])
plt.legend()
plt.xlabel('n')
plt.ylabel('MSID')
plt.savefig('MSID.png')

np.savez('msid.npz',ref_n = ref_n, ref_msid = ref_msid, Corr_n = Corr.n, Corr_msid = Corr.msid)

data = np.genfromtxt('data/CGMD_MSID.txt')
n, msid = data[:,0], data[:,1]
interpn = np.linspace(n[0],n[-1],10000)
interpmsid = np.interp(interpn,n,msid)
plt.semilogx(n,msid,label = 'Ref')
plt.semilogx(interpn,interpmsid,'r--',label='Interp')
plt.legend()
plt.show()
