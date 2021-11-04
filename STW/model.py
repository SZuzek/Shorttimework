
from gekko import GEKKO
import numpy as np
from scipy import stats
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy import interpolate


class STW():
    #### PARAMETERS ####
    def __init__(self, nt = 101, lower = 0, upper = 1, qlow = 0, qup = 1, tau = 1, U = 0.3, gamma = 1, search = 0,
                 cont = 0.0, A = 0.2, phi = 2, information = "imperfect", beta = 0.0, distribution = "uniform",
                 participation = "forced", mean = 1, sigma = 0.1, b = 0.2, display = True): 
    
        
        ## INITIALIZE GEKKO
        self.m = GEKKO(remote = False)
    
        ## PARAMETERS 
        self.nt = nt # number of grid points

        # support of theta
        self.lower = lower
        self.upper = upper
        
        # bargaining parameter of the worker (doesnt matter yet since no continuation value in next period)
        self.beta = beta
        
        # distribution
        self.distribution = distribution
        self.mean = mean
        self.sigma = sigma
        
        # bounds on q
        self.qup = qup
        self.qlow = qlow

        # production
        self.gamma = gamma # gamma = 1: linear

        
        # Marginal cost of public funds
        self.tau = tau # MCPF = (1+tau)
        
        # Outside option values
        
        self.search = search # value of searching for a job
        self.b = b # currently not used
        self.U = U # promised utiltiy
        self.cont = cont # continuation value, currently not used

        # Switch between different models
        self.information = information # perfect vs imperfect information (turns off IC)
        self.participation = participation # whether exclusion/inclusion is fixed or a choice 
        
        # cost function
        self.A = A # parameter for cost function
        self.phi = phi # exponent of cost function
        
        ### Solver options
        self.display = display # display details
        
        ### Distribution 
        if self.distribution == "uniform": 
            self.dist =  stats.uniform(self.lower, self.upper)
        elif self.distribution == "lognormal": 
            self.loc = self.mean - np.exp(self.sigma**2/2)
            self.dist =  stats.lognorm(s = self.sigma, loc = self.loc)
        elif self.distribution == "normal": 
            self.dist =  stats.norm(loc = self.mean, scale = self.sigma)
        else: raise NameError("distribution not defined")

        
        
    #### FUNCTIONS ####
    # disutility of labor
    def cost(self,x): return self.A*(x**self.phi)
    
   # density 
    def dens(self, x): 
        if self.distribution == "uniform": 
            return np.ones(len(x))/(self.upper-self.lower)  # uniform
        elif self.distribution == "lognormal": 
            loc = self.mean - np.exp(self.sigma**2/2)
            return stats.lognorm.pdf(x, s = self.sigma, loc = loc)
        elif self.distribution == "normal": 
            return stats.norm.pdf(x, loc = self.mean, scale = self.sigma)
        else: raise NameError("distribution not defined")

    
    # production
    def prod(self,theta, q): return theta*(q**self.gamma)
    
    # surplus (S_a)
    def surplus_function(self, theta, x, iota):  return iota*self.prod(theta, x) - self.cost(x)  + (1-iota)*self.search - (1-self.beta)*self.U + self.beta*self.cont
    
    # surplus in outside option (S_oo)
    def surplus_hat(self, theta, cutoff = 0): # 
        qgrid = np.arange(self.qlow,self.qup, 0.0001).reshape(-1,1) # transpose
        P = self.surplus_function(theta, qgrid, iota = 1) # you dont take into account where the utility of job-loss comes from. (transfers vs reallocation) 
        P = np.maximum(P, cutoff) # max between cutoff and surplus
        # to get efficient hours/separations etc, set cutoff = -search
        
        qhat = qgrid[np.argmax(P, axis = 0)]
        phat = np.max(P, axis = 0)
        return phat, qhat

    # dsa/dtheta 
    def dsa(self, theta, q): return q**self.gamma 

    # outside 
    def outside_option(self, theta): 
        phat, _ = self.surplus_hat(theta)
        return phat
    
    def outside_qhat(self, theta): 
        _, qhat = self.surplus_hat(theta)
        return qhat
    
    #### Summarize solution #### 
    # derive moments of transfers and hours for the optimal solution qhat
    def moments(self, cutoff = 0.05): 
        # approximate hours (linear interpolation) for integration
        grid = np.array(self.theta)
        hours = interpolate.interp1d(grid, np.array(self.q))
        outside_hours = interpolate.interp1d(grid, np.array(self.qhat))
        transfers = interpolate.interp1d(grid, np.array(self.t))
        hourdiff = interpolate.interp1d(grid, np.array(self.q) - np.array(self.qhat))
        statediff = interpolate.interp1d(grid, np.array(self.sa) - np.array(self.oo))
        
#        prod = interpolate.interp1d(grid, np.array)
        
        
        mom = dict()
        mom["separation_rate"] = self.dist.expect(lambda x: hours(x) < 0.0001, self.lower, self.upper, conditional = True)
        mom["separation_rate_oo"] = self.dist.expect(lambda x: outside_hours(x) < 0.0001, self.lower, self.upper, conditional = True)
        mom["total_transfers"] = self.dist.expect(transfers, self.lower, self.upper, conditional = True)
        mom["total_transfers_oo"] = self.dist.expect(lambda x: ((1-self.beta)*self.U - self.beta*self.cont - self.search) * (outside_hours(x) < 0.0001), self.lower, self.upper, conditional = True)
        mom["total_output"] = self.dist.expect(lambda x: self.prod(x, hours(x)), self.lower, self.upper, conditional = True)
        mom["total_output_oo"] = self.dist.expect(lambda x: self.prod(x, outside_hours(x)), self.lower, self.upper, conditional = True)
        # CHECK: mom["av_hour_reduction_STW"] = self.dist.expect(lambda x: (hourdiff(x) / outside_hours(x)) * ( np.abs(hourdiff(x)) < 0.001 ) , self.lower, self.upper, conditional = True)
        mom["transfer_STW"] = self.dist.expect(lambda x: transfers(x) * ( np.abs(hourdiff(x)) >= cutoff ) , self.lower, self.upper, conditional = True)
        mom["transfer_UI"] = self.dist.expect(lambda x: transfers(x) * ( np.abs(hourdiff(x)) < cutoff ) , self.lower, self.upper, conditional = True)
        mom["share_in_STW"] = self.dist.expect(lambda x: ( np.abs(hourdiff(x)) >= cutoff ) , self.lower, self.upper, conditional = True) 
        
        return mom
        
    
    
    #### MODEL ####
    # returns a Gekko object with all model equations with the parameters of the underlying STW object
    # variables that should be accesible have to be declared with self. 
    def init_gekko(self): 
        
        m = self.m
        # variables
        
        m.time = np.linspace(self.lower,self.upper,self.nt) # grid

        # distribution over the grid
        f = m.Param(value = self.dens(m.time))
        self.f = f

        # outside option. "parameters", since they do not depend on choice variables
        oo = m.Param(self.outside_option(self.m.time))
        qhat = m.Param(self.outside_qhat(self.m.time))
        self.oo = oo
        self.qhat = qhat
        
        # efficient hours (for later reference)
        _ , self.qeff = self.surplus_hat(self.m.time, cutoff = -self.U + self.search)
        
        # inclusion: (integer, 0 or 1)
        if (self.participation == "forced"):
            iota = m.Param(value = 1)
        elif (self.participation == "flex") & (self.information == "imperfect"):
            iota = m.Var(value = 0.0, lb = 0, ub = 1, integer = True)
        elif (self.participation == "flex") & (self.information == "perfect"): # otherwise it doesnt find the eq
            iota = m.Var(value = 1.0, lb = 0, ub = 1, integer = True)
        else: raise NameError("participation variable not defined")
        self.iota = iota
        
        # Variables
        # social planner transfer
        self.sa = m.Var(value=0) # lowest firm must be binding (?)
        self.saf = m.Var(value = 0)
        self.q = m.Var(value=0.0,lb=self.qlow,ub=self.qup)
        sa = self.sa # shortcuts
        saf = self.saf
        q = self.q 
        
        
        # To add theta, we need to add it as an auxilliary equation,  i.e, theta = time
        self.theta = m.Var(self.lower) 
        theta = self.theta
        m.Equation(theta.dt()==1) 

        # costs, surplus, transfers and ic are defined as intermediate variables (like a function).
        self.p = m.Intermediate(self.prod(theta, q))
        p = self.p    
        
        self.ic = m.Intermediate(self.dsa(theta, q))
        ic = self.ic

        self.c = m.Intermediate(self.cost(q))
        c = self.c
        
        # surplus generated by the match (paroduction, search etc)
        self.sfun = m.Intermediate(self.surplus_function(theta, q, iota))
                
        # transfer: residual between promised state sa and generated surplus sfun
        self.t = m.Intermediate(sa - self.sfun) # transfer
        t = self.t
        
        
        # Final value
        x = np.zeros(self.nt) # mark final time point
        x[-1] = 1.0
        final = self.m.Param(value=x)

        # Equations
        if self.information == "imperfect": 
            m.Equation(sa.dt()== ic)
        elif self.information == "perfect": 
            pass
        else: raise
        
        # constraint through outside option
        m.Equation(sa>= oo)
            
        # payoff function
        m.Equation(saf.dt() == (sa - (1+self.tau)*t )*f) # 

        
        # Objective function
        m.Obj( - saf*final) 
        
        # get out the objective value
        self.value = m.Intermediate(saf*final)
    
    # Solution settings
    def solve_gekko(self):
        # setup solver and solve
        m = self.m
        m.options.IMODE = 6 # optimal control mode
        m.options.MAX_ITER = 1200 
        m.options.DIAGLEVEL = 1 # diagnostics level
#         m.options.SOLVER = 0 # compare all available solvers
        m.options.SOLVER = 3 # default. 
        m.solve(disp=self.display) # solve

    
    