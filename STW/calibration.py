
import numpy as np
from scipy import stats
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy import interpolate
from .model import STW

# lognormal parameter "loc" is directly calibrated in definition of density, given sigma

class Calibration(): 
    def __init__(self, Ucalib, distributionname, target_sep  = 0.12, target_sep_eff = 0.07, target_inactive_share = 0.538): 
        # use STW class (to be consistent) and its solution for the outside option
        # constant paramters
        self.U = Ucalib
        self.mean = 1
        self.qup = 5
        self.distributionname = distributionname
        
        # moments
        self.target_sep_eff = target_sep_eff
        self.target_sep = target_sep
        self.target_inactive_share = target_inactive_share
        
        
        # initial guess 
        if self.target_sep_eff > 0:
            self.guess = [4.6, 0.4, 0.4, self.U/2] # [phi, A, sigma, SEARCH]
        else:
            self.guess = [4.6, 0.4, 0.4, 0] # [phi, A, sigma, SEARCH]
        # self.guess = [4.6, 0.4, 0.4] # [phi, A, sigma]
        
        
    def lowerupper(self):
        return self.dist.ppf([0.0005, 0.9995])
    
    def n_foc(self, theta, params):
        ''' make sure that this is still the same with a changed model. relies on linear production now'''
        phi, A, sigma, _ = params
        
        exponent = 1/(phi-1)
        fraction = theta/(phi*A)
        
        return fraction**exponent
    
    def theta_cutoffs(self, params):
        ''' make sure that this is still the same with a changed model. relies on linear production now'''
        phi, A, sigma, SEARCH = params

        fraction = 1/(phi*A)
        exponent1 = 1/(phi-1)
        exponent2 = phi/(phi-1)
        exponent3 = (1-phi)/phi
        exponent4 = (phi-1)/phi
        theta_0 = ((fraction**exponent1 - A*(fraction**exponent2))**exponent3) * self.U**exponent4 # cutoff for outside options
        if self.target_sep_eff == 0:
            theta_eff = 0
        else:
            theta_eff = ((fraction**exponent1 - A*(fraction**exponent2))**exponent3) * SEARCH**exponent4 # cutoff for planner (CHECK)

        return [theta_0, theta_eff]

    def n_hat(self, theta, params):
        ''' make sure that this is still the same with a changed model. relies on linear production now'''
        return self.n_foc(theta, params) * (theta>self.theta0)  
    
    def n_eff(self, theta, params):
        ''' make sure that this is still the same with a changed model. relies on linear production now'''
        return self.n_foc(theta, params) * (theta>self.thetaeff)  


    def add_distribution(self, params):
        phi, A, sigma, SEARCH = params
        if self.distributionname == "lognormal":
            self.loc = self.mean - np.exp(sigma**2/2)
            self.dist = stats.lognorm(s = sigma, loc = self.loc)
            self.lower, self.upper = self.lowerupper()
        elif self.distributionname == "normal":
            self.loc = self.mean
            self.dist = stats.norm(loc = self.loc, scale = sigma)
            self.lower, self.upper = self.lowerupper()
        elif self.distributionname == "uniform":
            self.loc = self.mean
            self.lower = 0
            self.upper = 2
            self.dist = stats.uniform(loc = self.lower, scale = self.upper)
        else: raise NameError("distribution not defined")

    
    
    def evaluate(self, params): 
        ''' Evaluate the fit of each model equation.
        Uses an explicit solution for n_hat. CAREFUL! MAY NEED TO CHANGE IF MODEL OBJECT CHAGNES!
        '''
         
        phi, A, sigma, SEARCH = params
        if self.target_sep_eff == 0:
            SEARCH = 0
        else: 
            pass
        self.add_distribution(params)
        
        
        # phi, A, sigma, loc = params
        # self.loc = loc
        
        theta0, thetaeff = self.theta_cutoffs(params)
        self.theta0 = theta0
        self.thetaeff = thetaeff
        
        #equation 1: inaction share (among active firms!)
        e1 = self.dist.expect(lambda x: (np.abs(self.n_hat(x, params) - self.mean) < 0.05), lb = self.theta0, ub = self.upper, conditional=True) 

        # equation 2: normal hours (among active firms)
        e2 = self.dist.expect(lambda x: self.n_hat(x, params), lb = self.theta0, ub = self.upper, conditional=True)

        # equation 3: separation rate 
        e3 = self.dist.expect(lambda x: (x < self.theta0), lb = self.lower, ub = self.upper, conditional=False) 
        
        # equation 4: efficient separation rate
        e4 = self.dist.expect(lambda x: (x < self.thetaeff), lb = self.lower, ub = self.upper, conditional=False) 

        
        return np.array([e1 - self.target_inactive_share, e2  - 1, e3  - self.target_sep,  10*(e4 - self.target_sep_eff)])
        
        
    def evaluate_STW(self, params): 
        ''' Evaluate the fit of each model equation using the approximation of q_hat from STW 
            Use to check if the calibration above also works with the STW object. If not, maybe different models are used. 
        '''
        # initialize STW object to get consistent outside-option qhat
        phi, A, sigma, SEARCH = params

        # i, A, sigma, loc = params
        # lf.loc = loc
        
        theta0, thetaeff = self.theta_cutoffs(params)
        
        self.add_distribution(params)


        self.stw_ = STW(display = False, distribution = self.distributionname, sigma = sigma, lower = self.lower, mean = self.mean,  
                        upper = self.upper, A = A, U = self.U, phi = phi, qup = self.qup, nt = 1001,
                       participation = "flex", search = SEARCH, information = "imperfect")
        self.stw_perf_ = STW(display = False, distribution = self.distributionname, sigma = sigma, lower = self.lower, mean = self.mean,  
                        upper = self.upper, A = A, U = self.U, phi = phi, qup = self.qup, nt = 401, 
                        participation = "flex", search = SEARCH, information = "perfect")
        
        # targeted moments are hard-coded in here for now

        #equation 1: inaction share
        e1 = self.dist.expect(lambda x: (np.abs(self.stw_.outside_qhat(x) - self.mean) < 0.05), lb = self.theta0, ub = self.upper, conditional=True)

        # equation 2: normal hours 
        e2 = self.dist.expect(self.stw_.outside_qhat, lb = theta0, ub = self.upper, conditional=True) 

        # equation 3: separation rate 
        e3 = self.dist.expect(lambda x: (np.abs(self.stw_.outside_qhat(x)) < 0.0001), lb = self.lower, ub = theta0, conditional=False)

        
        ### Need to solve the perfect-information problem to check for the cutoff of efficient separations in the STW object
        self.stw_perf_.init_gekko()
        self.stw_perf_.solve_gekko()
        
        grid = np.array(self.stw_perf_.theta)
        qperfect = interpolate.interp1d(grid, np.array(self.stw_perf_.q))
        
        # equation 4: efficient separations
        e4 = self.dist.expect(lambda x: (qperfect(x)<0.0001), lb = self.lower, ub = self.upper, conditional = False)
        

        return np.array([e1 - self.target_inactive_share, e2  - 1, e3  - self.target_sep, e4 - self.target_sep_eff])

    
    def loss(self, params):
         return np.sum(self.evaluate(params)**2)
        
        
    def optimize(self): 
        # need a method without gradient
        args = dict(method = "Nelder-Mead")
        opts = dict(disp = True, return_all = True)
        o = optimize.minimize(self.loss, self.guess, options = opts, **args)
        return o 
        
