# --------------------------------------------------------
# theory_util.py
# by Zhenyuan Zhang, zzy@stanford.edu
# Shaswat Mohanty, shaswatm@stanford.edu
# Objectives
# Utility functions for theoretical predictions branching spatial processes (BRW, BMRW, BBM and GBRW)
#
# Cite: (to be added)
# --------------------------------------------------------
import numpy as np
import sys, os
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import fsolve,root,bisect, curve_fit
import scipy.special as special
from tqdm import tqdm
import argparse

def objective(x,a,b):
    """
    Compute the objective value based on the given parameters.

    Parameters:
        x (float): The input value.
        a (float): The coefficient for x.
        b (float): The constant term.

    Returns:
        float: The computed objective value.
    """
    return a*x + b

class Theoretical_BRW():
    def __init__(self, dim = 3, method = 'classical', rate = None, rho_range = 'small'):
        self.dim = dim
        self.method = method
        self.rate = rate
        self.rho_range = rho_range
        if rho_range == 'small':
            self.rho_arr = np.array([1.0295233547701528, 1.0376824580907913, 1.0457569507079911, 1.0594585479958203, 1.075720859698337, 1.0860194380000792, 1.097926243894198, 1.1085438423022842, 1.1221982305721785, 1.1346392570776829, 1.1485329189890172, 1.158463320987782, 1.1738380193069964, 1.1867845421985102])
            self.MSID_values = np.array([1.80679726, 1.78554887, 1.76561357 ,1.74226128 ,1.72231856, 1.71120625, 1.69914301, 1.68896488 ,1.6771406 , 1.66745474 ,1.65691882, 1.64941093, 1.63779955,1.62840362])
        elif rho_range == 'large':
            self.rho_arr = np.linspace(1.1,2.9,19)

        self.Lxs = np.exp(np.linspace(np.log(20),np.log(60),4))

    def f(self,l,y):
        """
        Calculate the value of f based on the given parameters.
        Parameters:
            l (float): The value of l.
            y (float): The value of y.
            dim (int, optional): The dimensionality of the calculation. Defaults to 2.
        Returns:
            float: The calculated value of f.
        Raises:
            TypeError: If the dimensionality is neither 2 nor 3.
        """
        if self.dim==2:
            return (y*math.e**(l*y)/(1-y**2)**(1/2))/math.pi
        elif self.dim ==3:
            return y*math.e**(l*y)
        else:
            TypeError('Wrong dimensionality')

    def g(self,l,y):
        """
        Calculate the value of the function based on the given parameters.
        Parameters:
            l (float): The value of l.
            y (float): The value of y.
            dim (int, optional): The dimensionality of the function. Defaults to 2.
        Returns:
            float: The calculated value of the function.
        
        Raises:
            TypeError: If the dimensionality is neither 2 nor 3.
        """
        if self.dim==2:
            return (math.e**(l*y)/(1-y**2)**(1/2))/math.pi
        elif self.dim ==3:
            return math.e**(l*y)
        else:
            TypeError('Wrong dimensionality') 

    def d(self,l):
        """
        Calculate the ratio of the integral of f and g.
        Parameters:
        l (float): The input value for the function.
        dim (int, optional): The dimension of the function. Default is 2.
        Returns:
        float: The result of the integral.
        """
        if self.dim==1:
            return (math.e**l-math.e**(-l))/(math.e**l+math.e**(-l))
        else:
            A=integrate.quad(lambda y: self.f(l,y), -1, 1)
            B=integrate.quad(lambda y: self.g(l,y), -1, 1)
            C=A[0]/B[0]
            return C

    def m(self,x):
        """
        A function that finds where d(y) = x for a given dimensionality. m is used to denote the main term in the asymptotic.
        Parameters:
            x (float): The input value.
            dim (int, optional): The dimension of the calculation. Defaults to 2.
        Returns:
            float: The calculated root.
        """
        def p(y):
            return  self.d(y)-x
        return bisect(p,1e-5 if self.dim<2 else 1e-2 if self.dim>2 else 1e-4,100,xtol=0.001)

    def I(self,x):
        """
        Calculate the value of I based on the input x and dimension dim. I is the large deviation rate function.
        Parameters:
            x (float): The input value.
            dim (int, optional): The dimension of the calculation. Default is 2.
        Returns:
            float: The calculated value of I.
        """
        if self.dim==1:
            return self.m(x)*x-math.log((math.e**(self.m(x))+math.e**(-self.m(x)))/2) 
        else:
            return self.m(x)*x-math.log(integrate.quad(lambda y: self.g(self.m(x),y), -1, 1)[0]/(self.dim-1))  
            
    def c1(self,rho):
        """
        Calculate the value of c1 given the value of rho and the dimension obtained by finding where the large deviation rate is equal to log(rho)
        Args:
            rho (float): The value of rho.
            dim (int, optional): The dimension. Defaults to 2.
        Returns:
            float: The calculated value of c1.
        """
        if self.method == 'classical':
            def q(y):
                """
                Calculates the value of q.

                Parameters:
                - y: The input value for the function.

                Returns:
                - The calculated value of q.
                """
                return self.I(y)-math.log(rho)
        else:
            if self.method == 'termination':
                r = self.rate
            elif self.method == 'delayed_branching':
                r = 0
            else:
                raise TypeError("The method of branching random walk is unrecognized. Assign either classical, termination or delayed_branching.")
            def q(y):
                """
                Calculates the value of q for a given input y.

                Parameters:
                    y (float): The input value for the function.

                Returns:
                    float: The calculated value of q.
                """
                return self.I(y)-math.log(0.5*(1-r)+((rho-1)*(1-0.5*r)+0.25*(1-r)**2)**(1/2))
                
        return bisect(q,1e-5 if self.dim<2 else 1e-2 if self.dim>2 else 1e-4,0.999 if self.dim==1 else 0.98,xtol=0.00001)

    def c2(self,rho):
        """
        Calculate the value of c2 given the value of rho and the dimension obtained by finding where the large deviation rate is equal to log(rho)
        Args:
            rho (float): The value of rho.
            dim (int, optional): The dimension. Defaults to 2.
        Returns:
            float: The calculated value of c2.
        """
        return self.m(self.c1(rho))

    def FPT(self,rho):
        """
        Calculate the value of FPT given the value of rho.
        Args:
            rho (float): The value of rho.
            dim (int, optional): The dimension. Defaults to 2.
        Returns:
            float: The calculated value of FPT.
        """
        return self.Lx/self.c1(rho)+(self.dim+2)/2/self.c1(rho)/self.c2(rho)*math.log(self.Lx)

    def calc_c1(self):
        """
        Calculate c1 values for each rho in rho_arr. FOr the smaller range corresponding to the CGMD rho values, we calculation c1_bar
        Parameters:
            - rho_arr (list): A list of rho values.
            - dim (int, optional): The dimension of the calculation. Defaults to 1.
        Returns:
            - np.array: An array of calculated c1 values.
        """
        c1_arr=[]
        for i, rho in tqdm(enumerate(self.rho_arr)):
            if self.rho_range == 'large':
                c1_arr.append(self.c1(rho))
            else:
                ls = np.zeros(4)
                for j, Lx in enumerate(self.Lxs):
                    self.Lx = Lx/math.sqrt(self.MSID_values[i])
                    ls[j] = self.FPT(rho)
                popt, _ = curve_fit(objective, self.Lxs, ls)
                a,b=popt
                c1_arr.append(1.0/a)

        plt.figure(1)
        plt.plot((self.rho_arr-1.0)/2.0,c1_arr)
        plt.xlabel(r'$\tilde{\lambda}$')
        if self.rho_range == 'large':
            plt.ylabel(r'$c_1$')
        else:
            plt.ylabel(r'$\overline{c}_1$')
        plt.show()
            
        return np.array(c1_arr)

class Theoretical_BBM():
    def __init__(self, dim = 3, method = 'classical', rate = None, rho_range = 'small'):
        self.dim = dim
        self.method = method
        self.rate = rate
        self.rho_range = rho_range
        if rho_range == 'small':
            self.rho_arr = np.array([1.0295233547701528, 1.0376824580907913, 1.0457569507079911, 1.0594585479958203, 1.075720859698337, 1.0860194380000792, 1.097926243894198, 1.1085438423022842, 1.1221982305721785, 1.1346392570776829, 1.1485329189890172, 1.158463320987782, 1.1738380193069964, 1.1867845421985102])
            self.MSID_values = np.array([1.80679726, 1.78554887, 1.76561357 ,1.74226128 ,1.72231856, 1.71120625, 1.69914301, 1.68896488 ,1.6771406 , 1.66745474 ,1.65691882, 1.64941093, 1.63779955,1.62840362])
        elif rho_range == 'large':
            self.rho_arr = np.linspace(1.1,2.9,19)
            self.MSID_values = np.ones(len(self.rho_arr))*3.0
            self.rate = 0.0

        self.Lxs = np.exp(np.linspace(np.log(20),np.log(60),4))

    def c1(self,rho):
        """
        Calculate the value of c1.

        Parameters:
            rho (float): The value of rho.

        Returns:
            float: The calculated value of c1.
        """
        def q(y):
            """
            Calculates the value of q based on the input parameter y.

            Parameters:
            - y: The input value for the calculation.

            Returns:
            - float: The calculated value of q.
            """
            return self.I(y)-math.log(0.5*(1-self.rate)+((rho-1)*(1-0.5*self.rate)+0.25*(1-self.rate)**2)**(1/2))
    
        if self.rate != 0:
            return bisect(q,0.01,0.98,xtol=0.00001)
        else:
            return math.sqrt(2*(rho-1))

    def FPT(self,v,r1):
        """
        Calculates the FPT (First Passage Time) based on the given parameters.

        Parameters:
            v (float): The velocity of the object.
            r1 (float): The radius of the first circular boundary.

        Returns:
            float: The calculated FPT value.

        Note:
            - The FPT is calculated using the formula: self.Lx / (v * math.sqrt(2 * math.log(r1))) + 5 * math.log(self.Lx / v) / (4 * math.log(r1))
            - The formula assumes that self.Lx is already defined.
        """
        return self.Lx/(v*math.sqrt(2*math.log(r1)))+5*math.log(self.Lx/v)/(4*math.log(r1))

    def rr(self,rho):
        """
        Calculate the effective branching rate for BBM with delayed branching.

        Parameters:
            rho (float): A constant value used in the function.

        Returns:
            float: The root of the function.
        """
        def p(y):
            """
            Calculates the value of p based on the given parameter y.

            Parameters:
                y (float): The input parameter y.

            Returns:
                float: The calculated value of p.
            """
            return rho-1-y*(self.rate+math.log(y))
    
        return bisect(p,1,10,xtol=0.001)

    def calc_c1(self):
        """
        Calculate c1 values for each rho in rho_arr. FOr the smaller range corresponding to the CGMD rho values, we calculation c1_bar
        Parameters:
            - rho_arr (list): A list of rho values.
            - dim (int, optional): The dimension of the calculation. Defaults to 1.
        Returns:
            - np.array: An array of calculated c1 values.
        """
        c1_arr = []
        for i, rho in tqdm(enumerate(self.rho_arr)):
            if self.rho_range == 'large':
                c1_arr.append(self.c1(rho))
            else:
                v=math.sqrt(self.MSID_values[i]/3)
                ri = self.rr(rho)
                ls = np.zeros(4)
                for j, self.Lx in enumerate(self.Lxs):
                    ls[j] = self.FPT(v,ri)
                popt, _ = curve_fit(objective, self.Lxs, ls)
                a,b=popt
                c1_arr.append(1.0/a)

        plt.figure(1)
        plt.plot((self.rho_arr-1.0)/2.0,c1_arr)
        plt.xlabel(r'$\tilde{\lambda}$')
        if self.rho_range == 'large':
            plt.ylabel(r'$c_1$')
        else:
            plt.ylabel(r'$\overline{c}_1$')
        plt.show()

        return np.array(c1_arr)

class Theoretical_GBRW():
    def __init__(self, dim = 3, method = 'classical', rate = None, rho_range = 'small'):
        self.dim = dim
        self.method = method
        self.rate = rate
        self.rho_range = rho_range
        if rho_range == 'small':
            self.rho_arr = np.array([1.0295233547701528, 1.0376824580907913, 1.0457569507079911, 1.0594585479958203, 1.075720859698337, 1.0860194380000792, 1.097926243894198, 1.1085438423022842, 1.1221982305721785, 1.1346392570776829, 1.1485329189890172, 1.158463320987782, 1.1738380193069964, 1.1867845421985102])
            self.MSID_values = np.array([1.80679726, 1.78554887, 1.76561357 ,1.74226128 ,1.72231856, 1.71120625, 1.69914301, 1.68896488 ,1.6771406 , 1.66745474 ,1.65691882, 1.64941093, 1.63779955,1.62840362])
        elif rho_range == 'large':
            self.rho_arr = np.linspace(1.1,2.9,19)
            self.MSID_values = np.ones(len(self.rho_arr))*3.0
            self.rate = 0

        self.Lxs = np.exp(np.linspace(np.log(20),np.log(60),4))

    def I(self,x):
        """
        Calculates the square of the given number and returns half of it.

        Parameters:
            x (float): The number to be squared.

        Returns:
            float: The result of squaring the number and dividing it by 2.
        """
        return x**2/2.0

    def c1(self,rho):
        """
        Calculates the value of c1 based on the given parameter rho.
        
        Args:
            rho (float): The value of rho to be used in the calculation.
        
        Returns:
            float: The calculated value of c1.
        """
        def q(y):
            """
            Calculates the value of q based on the given parameter y.

            Parameters:
            y (int): The value to calculate q for.

            Returns:
            float: The calculated value of q.
            """
            return self.I(y)-math.log(0.5*(1-self.rate)+((rho-1)*(1-0.5*self.rate)+0.25*(1-self.rate)**2)**(1/2))
    
        if self.rate != 0:
            return bisect(q,0.01,0.98,xtol=0.00001)
        else:
            return math.sqrt(2*math.log(rho))

    def FPT(self,v,rho):
        """
        Calculate the FPT (First Passage Time) for a given velocity and density.

        Args:
            v (float): The velocity of the object.
            rho (float): The density of the object.

        Returns:
            float: The calculated FPT value.
        """
        return ((self.Lx/math.sqrt(v/3))/self.c1(rho)+5*(math.log(self.Lx/math.sqrt(v/3)))/(2*self.c1(rho)*(self.c1(rho))))

    def calc_c1(self):
        """
        Calculate c1 values for each rho in rho_arr. FOr the smaller range corresponding to the CGMD rho values, we calculation c1_bar
        Parameters:
            - rho_arr (list): A list of rho values.
            - dim (int, optional): The dimension of the calculation. Defaults to 1.
        Returns:
            - np.array: An array of calculated c1 values.
        """
        c1_arr = []
        for i, rho in tqdm(enumerate(self.rho_arr)):
            if self.rho_range == 'large':
                c1_arr.append(self.c1(rho))
            else:
                v=self.MSID_values[i]
                ls = np.zeros(4)
                for j, self.Lx in enumerate(self.Lxs):
                    ls[j] = self.FPT(v,rho)
                popt, _ = curve_fit(objective, self.Lxs, ls)
                a,b=popt
                c1_arr.append(1.0/a)

        plt.figure(1)
        plt.plot((self.rho_arr-1.0)/2.0,c1_arr)
        plt.xlabel(r'$\tilde{\lambda}$')
        if self.rho_range == 'large':
            plt.ylabel(r'$c_1$')
        else:
            plt.ylabel(r'$\overline{c}_1$')
        plt.show()
        return np.array(c1_arr)

def main():
    parser = argparse.ArgumentParser(description="Computing branched random walk analysis")
    parser.add_argument("--BRW", default=False, action="store_true", help="to plot distribution for a single rate")
    parser.add_argument("--BBM",      default=False, action="store_true", help="to plot Sp distributions as a function of box size")
    parser.add_argument("--GBRW", default=False, action="store_true",
                        help="to plot C1 vs rho")
    parser.add_argument("--rate",   type=float, default=3.0/500.0, help="termination rate in units of inverse time")
    parser.add_argument("--dim", type=int, default=3, help="Branching random-walk dimensionality")
    parser.add_argument("--rho_range", type=str, default='small', help="minimum rate to use in tau analysis")
    parser.add_argument("--method", type=str, default='termination', help="maximum rate to use in tau analysis")

    args = parser.parse_args()
    if args.BRW:
        util = Theoretical_BRW(rate=args.rate, dim=args.dim, rho_range=args.rho_range, method=args.method)
    elif args.BBM:
        util = Theoretical_BBM(rate=args.rate, dim=args.dim, rho_range=args.rho_range, method=args.method)
    elif args.GBRW:
        util = Theoretical_GBRW(rate=args.rate, dim=args.dim, rho_range=args.rho_range, method=args.method)
    else:
        raise ValueError("Incorrect dimensionality specified by --dim. Try default --dim or --dim [2,3]")

    print(util.calc_c1())


if __name__ == '__main__':
    main()

