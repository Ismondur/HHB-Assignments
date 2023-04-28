import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborFertModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        # set parameters - I have used the parameter of the lecture 5 notebook

        par.T = 10 # time periods
        
        # preferences
        par.rho = 1/(1.02) # discount factor

        par.beta_0 = 0.1 # weight on labor dis-utility (constant)
        par.beta_1 = 0.05 # additional weight on labor dis-utility (children)
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.3 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.1 # labor income tax
        par.p_spouse = 0.0 # probability of spouse

        # children
        par.p_birth = 0.1
        par.theta = 0.0 # child care cost

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 50 #70 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in human capital grid
        par.Nk = 20 # number of grid points in human capital grid    

        par.Nn = 2 # number of children +1
        par.Ns = 2 # possibility of spouse

        # simulation
        par.simT = par.T # number of periods
        par.simN = 10_000 # number of individuals

        # event time grid
        par.min_time = -8
        par.max_time = 8
        par.event_grid = np.arange(par.min_time,par.max_time+1)


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1.1)

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # c. number of children grid and spouse grid
        par.n_grid = np.arange(par.Nn)
        par.s_grid = np.arange(par.Ns)

        # d. solution arrays
        shape = (par.T,par.Nn,par.Ns,par.Na,par.Nk)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)
        sim.s = np.zeros(shape,dtype=np.int_)

        # f. draws used to simulate child arrival and spouse
        np.random.seed(9210)
        #np.random.seed(2109)
        sim.draws_uniform = np.random.uniform(size=shape)

        np.random.seed(1573)
        sim.draw_2_uniform = np.random.uniform(size=shape)

        # g. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)
        sim.s_init = np.ones(par.simN,dtype=np.int_)

        # h. vector of incomes
        par.w_vec = par.w * np.ones(par.T) # vector of wages
        par.y_vec = 0.1 + 0.01*np.arange(10) # vector of spouse income


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):
            #print(f'time: {t}')

            # i. loop over state variables: spuse present, number of children, human capital and wealth in beginning of period
            for i_n,kids in enumerate(par.n_grid):
                for i_s,spouse in enumerate(par.s_grid):
                    if ((par.p_spouse == 0.0) & (i_s == 1)): # no need to solve for spouse is present, if that never happens
                        sol.c[:,:,1,:,:] = sol.c[:,:,0,:,:] 
                        sol.h[:,:,1,:,:] = sol.h[:,:,0,:,:] 
                        sol.V[:,:,1,:,:] = sol.V[:,:,0,:,:] 
                        continue  # skip the current iteration

                    for i_a,assets in enumerate(par.a_grid):
                        for i_k,capital in enumerate(par.k_grid):
                            idx = (t,i_n,i_s,i_a,i_k)

                            # ii. find optimal consumption and hours at this level of wealth in this period t.

                            if t==par.T-1: # last period
                                obj = lambda x: self.obj_last(x[0],assets,capital,kids,spouse)

                                constr = lambda x: self.cons_last(x[0],assets,capital,kids,spouse)
                                nlc = NonlinearConstraint(constr, lb=0.0, ub=np.inf,keep_feasible=True)

                                # call optimizer
                                hours_min = (par.theta * kids - assets - par.y_vec[par.T-1] * spouse) / self.wage_func(capital,t) + 1.0e-5 # minimum amout of hours that ensures positive consumption
                                hours_min = np.maximum(hours_min,1.5)
                                init_h = np.array([hours_min]) if i_k==0 else np.array([sol.h[t,i_n,i_s,i_a,i_k-1]]) # initial guess on optimal hours

                                #print(f'Minimizing for {idx} with kids = {kids}') if i_k==0 & i_a == 0 else None
                                
                                res = minimize(obj,init_h,bounds=((0.0,np.inf),),constraints=nlc,method='trust-constr')

                                # store results
                                sol.c[idx] = self.cons_last(res.x[0],assets,capital,kids,spouse)
                                sol.h[idx] = res.x[0]
                                sol.V[idx] = -res.fun

                            else:
                                
                                # objective function: negative since we minimize
                                obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,kids,spouse,t)  

                                # bounds on consumption 
                                lb_c = 0.000001 # avoid dividing with zero
                                ub_c = np.inf

                                # bounds on hours
                                lb_h = 0.0
                                ub_h = np.inf 

                                bounds = ((lb_c,ub_c),(lb_h,ub_h))
                    
                                # call optimizer

                                init = np.array([lb_c,1.0]) if (i_n == 0 & i_a==0 & i_k==0) else res.x  # initial guess on optimal consumption and hours
                                #init = np.array([sol.c[t+1,i_n,i_s,i_a,i_k],sol.h[t+1,i_n,i_s,i_a,i_k]]) #if (i_n == 0 & i_a==0 & i_k==0) else res.x  # initial guess on optimal consumption and hours
                                res = minimize(obj,init,bounds=bounds,method='L-BFGS-B') 
                                #print(f'Minimized for {idx}') if i_k ==0 else None
                            
                                # store results
                                sol.c[idx] = res.x[0]
                                sol.h[idx] = res.x[1]
                                sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital,kids,spouse):
        par = self.par

        income = self.wage_func(capital,par.T-1) * hours + par.y_vec[par.T-1] * spouse
        cons = assets + income - par.theta*kids

        #print(f'child care with {kids} children costs {assets+income-cons} and taxes are {par.tau*100}%') if capital==0 else None

        return cons

    def obj_last(self,hours,assets,capital,kids,spouse):
        cons = self.cons_last(hours,assets,capital,kids,spouse)
        return - self.util(cons,hours,kids)    

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,kids,spouse,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. penalty for violating bounds. 
        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0
            cons = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. utility from consumption
        util = self.util(cons,hours,kids)
        
        # d. *expected* continuation value from savings
        income = self.wage_func(capital,t) * hours + par.y_vec[t] * spouse
        a_next = (1.0+par.r)*(assets + income - cons - par.theta * kids)
        k_next = capital + hours

        # no birth
        kids_next = kids
        #V_next = par.p_spouse * sol.V[t+1,kids_next,0] 
        V_next = par.p_spouse * sol.V[t+1,kids_next,1] + (1- par.p_spouse) * sol.V[t+1,kids_next,0]
        #print(f'V_next is {V_next}')
        V_next_no_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        # birth
        if (kids>=(par.Nn-1)):
            # cannot have more children
            V_next_birth = V_next_no_birth
        
        else:
            kids_next = kids + 1
            #V_next = sol.V[t+1,kids_next,0]
            V_next = sol.V[t+1,kids_next,1] # if there is a birth, there must be a spouse present. In baseline, this equals sol.V[t+1,kids_next,0]
            V_next_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        #next period expected value - use par.p_spouse**np.ceil(par.p_spouse) to avoid using an if-statement
        EV_next = par.p_spouse**np.ceil(par.p_spouse)*par.p_birth * V_next_birth + (1-par.p_spouse**np.ceil(par.p_spouse)*par.p_birth)*V_next_no_birth

        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours,kids):
        par = self.par

        beta = par.beta_0 + par.beta_1*kids

        return (c)**(1.0+par.eta) / (1.0+par.eta) - beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func(self,capital,t):
        # after tax wage rate
        par = self.par

        return (1.0 - par.tau)* par.w_vec[t] * (1.0 + par.alpha * capital)

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.n[i,0] = sim.n_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]
            sim.s[i,0] = 0*sim.s_init[i] # no spouse in first period
            if (sim.draw_2_uniform[i,0]<= par.p_spouse): # ... except if probability demands it
                sim.s[i,0] = sim.s_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.n[i,t],sim.s[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.wage_func(sim.k[i,t],t)*sim.h[i,t] + par.y_vec[t]*sim.s[i,t]
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t] - par.theta * sim.n[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    sim.s[i,t+1] = 0 # no spouse in next period
                    if (sim.draw_2_uniform[i,t+1]<= par.p_spouse): # ... except if probability demands it
                        sim.s[i,t+1] = 1

                    birth = 0 
                    if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1)) & ((sim.s[i,t+1]==1) | (par.p_spouse==0))):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth

    def ChildPenalty(self):

        # a. unpack
        par = self.par
        sim = self.sim

        # b. setup for births
        periods = np.tile([t for t in range(par.simT)],(par.simN,1))
        birth = np.zeros(sim.n.shape,dtype=np.int_)

        # c. calculate time since birth
        birth[:,1:] = (sim.n[:,1:] - sim.n[:,:-1]) > 0
        time_of_birth = np.max(periods * birth, axis=1)

        I = time_of_birth>0
        time_of_birth[~I] = -1000 # never as a child
        time_of_birth = np.transpose(np.tile(time_of_birth , (par.simT,1)))

        time_since_birth = periods - time_of_birth

        # d. calculate average outcome across time since birth

        event_hours = np.nan + np.zeros(par.event_grid.size)
        for t,time in enumerate(par.event_grid):
            event_hours[t] = np.mean(sim.h[time_since_birth==time])

        return (event_hours - event_hours[par.event_grid==-1])/event_hours[par.event_grid==-1]
                            
                            
                            


