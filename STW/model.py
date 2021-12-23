
from gekko import GEKKO
import numpy as np
from scipy import stats
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy import interpolate


#### Summarize solution #### 
# derive moments of transfers and hours for the optimal solution qhat
def moments(stw, epsilon = 0.05): 
    # approximate hours (linear interpolation) for integration
    grid = np.array(stw.theta)
    hours = interpolate.interp1d(grid, np.array(stw.q))
    outside_hours = interpolate.interp1d(grid, np.array(stw.qhat))
    transfers = interpolate.interp1d(grid, np.array(stw.t))
    hourdiff = interpolate.interp1d(grid, np.array(stw.q) - np.array(stw.qhat))
    statediff = interpolate.interp1d(grid, np.array(stw.sa) - np.array(stw.oo))

#        prod = interpolate.interp1d(grid, np.array)


    mom = dict()
    mom["separation_rate"] = stw.dist.expect(lambda x: hours(x) < 0.0001, stw.lower, stw.upper, conditional = True)
    mom["separation_rate_oo"] = stw.dist.expect(lambda x: outside_hours(x) < 0.0001, stw.lower, stw.upper, conditional = True)
    mom["total_transfers"] = stw.dist.expect(transfers, stw.lower, stw.upper, conditional = True)
    mom["total_transfers_oo"] = stw.dist.expect(lambda x: (stw.U - stw.search) * (outside_hours(x) < 0.0001), stw.lower, stw.upper, conditional = True)
    mom["total_output"] = stw.dist.expect(lambda x: stw.prod(x, hours(x)), stw.lower, stw.upper, conditional = True)
    mom["total_output_oo"] = stw.dist.expect(lambda x: stw.prod(x, outside_hours(x)), stw.lower, stw.upper, conditional = True)
    # CHECK: mom["av_hour_reduction_STW"] = stw.dist.expect(lambda x: (hourdiff(x) / outside_hours(x)) * ( np.abs(hourdiff(x)) < 0.001 ) , stw.lower, stw.upper, conditional = True)
    mom["transfer_STW"] = stw.dist.expect(lambda x: transfers(x) * ( np.abs(hourdiff(x)) >= epsilon ) , 
                                          stw.lower, stw.upper, conditional = True)
    mom["transfer_UI"] = stw.dist.expect(lambda x: transfers(x) * ( np.abs(hourdiff(x)) < epsilon ) , 
                                         stw.lower, stw.upper, conditional = True)
    mom["share_in_STW"] = stw.dist.expect(lambda x: ( np.abs(hourdiff(x)) >= epsilon ) , stw.lower, stw.upper, conditional = True) 

    # Hours cutoff (relative to average full time hours n), below which matches do not get supported 
    mom["cutoff_n"] = min(np.array(stw.q[1:])[np.array(stw.iota[1:]) > 0.9]) # lowest n with iota = 1 (excluding the first)

    # share with zero transfers
    no_transfers_cutoff = min(np.array(stw.theta)[np.array(stw.t < 0.01)])
    mom["share_no_transfers"] = 1 - stw.dist.cdf(no_transfers_cutoff)


    return mom




class STW():
    #### PARAMETERS ####
    def __init__(self, nt = 101, lower = 0, upper = 1, qlow = 0, qup = 1, tau = 1, U = 0.3, gamma = 1, search = 0,
                 cont = 0.0, A = 0.2, phi = 2, information = "imperfect", beta = 0.0, distribution = "uniform",
                 participation = "forced", mean = 1, sigma = 0.1, display = True): 
    
        
    
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
    def surplus_function(self, theta, x, iota):  return iota*self.prod(theta, x) - iota*self.cost(x)  + (1-iota)*self.search - self.U 
    
    # surplus in outside option (S_oo)
    def surplus_hat(self, theta, cutoff = 0): # 
        qgrid = np.arange(self.qlow,self.qup, 0.00005).reshape(-1,1) # transpose
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
    
    
    #### MODEL ####
    # returns a Gekko object with all model equations with the parameters of the underlying STW object
    # variables that should be accesible have to be declared with self. 
    def init_gekko(self): 
        
        ## INITIALIZE GEKKO
        self.m = GEKKO(remote = True)
        
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
            iota = m.Var(value = 0.5, lb = 0, ub = 1, integer = True)
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
    def solve_gekko(self, rtol = 1.0e-6, otol = 1.0e-6, max_iter = 1000):
        # setup solver and solve
        m = self.m
        m.options.IMODE = 6 # optimal control mode
        m.options.MAX_ITER = 1200 
        m.options.DIAGLEVEL = 1 # diagnostics level
#         m.options.SOLVER = 0 # compare all available solvers
#         m.options.SOLVER = 1 
        m.options.SOLVER = 3 # default. 

        m.options.RTOL = rtol
        m.options.OTOL = otol
        
        m.options.MAX_ITER = max_iter # default: 250
        m.options.MAX_MEMORY = 4 # default: 4. sufficient, unless memory error is thrown 
        
        # m.options.SOLVER = 1 # integer problems. not performing well
        m.solve(disp=self.display) # solve

    
    