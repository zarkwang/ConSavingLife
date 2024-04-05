import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import CubicSpline
from tqdm import tqdm



def utility(c,type='CRRA',**kwargs):

    if type == 'CRRA':
        u = c**(1-kwargs['risk_coef']) / (1-kwargs['risk_coef'])
    elif type == 'CARA':
        u = 1 - np.exp(-kwargs['risk_coef']*c)

    return u 


def utility_1st_d(c,type='CRRA',**kwargs):

    if type == 'CRRA':
        u_prime = c**(-kwargs['risk_coef'])
    elif type == 'CARA':
        u_prime = kwargs['risk_coef']*np.exp(-kwargs['risk_coef']*c)

    return u_prime


def inverse_utility_1st_d(u_prime,type='CRRA',**kwargs):

    if type == 'CRRA':
        c = u_prime**(-1/kwargs['risk_coef'])
    elif type == 'CARA':
        c = -1/kwargs['risk_coef']*np.log(u_prime/kwargs['risk_coef'])

    return c




class life_path:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    
    def get_next_wealth(self,wealth_now,income_now,consume_now):
           
        w = self.interest_rate*(wealth_now + income_now - consume_now)
        
        return w


    def get_wealth_process(self,consume_process,income_process,init_wealth):

        life_end = len(income_process)-1

        wealth_process = [init_wealth]

        for t in range(life_end + 1):
            wealth_process += [self.get_next_wealth(wealth_process[t],income_process[t],consume_process[t])]
        
        return np.array(wealth_process)
    
    def get_bequest(self,consume_process,income_process,init_wealth):

        life_end = len(income_process) - 1

        interest_process = np.array([self.interest_rate**(life_end + 1 - t) for t in range(life_end + 1)])

        cum_saving = (income_process - consume_process) @ interest_process
        cum_init_wealth = self.interest_rate**(life_end + 1) * init_wealth

        return cum_saving + cum_init_wealth
    
    def life_utility(self,consume_process,**kwargs):

        income_process,init_wealth = kwargs['income_process'],kwargs['init_wealth']

        c = np.array(consume_process).astype(float)

        u = utility(c,risk_coef=self.risk_coef,type=self.utility_type)
        d = [self.discount_factor**t for t in range(len(c))]
        
        u_cum_consumption = u @ d

        bequest = self.get_bequest(consume_process,income_process,init_wealth)
        u_bequest = utility(bequest,risk_coef=self.risk_coef,type=self.utility_type)

        return u_cum_consumption + self.bequest_motive * u_bequest


class forwardSolver(life_path):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_constraint_consume_coef(self,t,life_end):

        # The linear constraint are Ac <= b
        # This function creates A (the coefficients of c) 
    
        coefs = [self.interest_rate**(t+1-i) for i in range(0,t+1)]
        zero_coefs = [0 for i in range(t+1,life_end+1)]
        
        return coefs + zero_coefs
    
    def set_constraint_bound(self,t,income_process,init_wealth):

        # The linear constraints are Ac <= b
        # This function creates b (the bounds) 
    
        cum_wealth = self.interest_rate**(t+1)*init_wealth 
        cum_income = np.sum([self.interest_rate**(t+1-i)*income_process[i] for i in range(t+1)])
        
        return cum_wealth + cum_income
    
    def opt_consumption(self,income_process,init_wealth):

        # Solve optimal planned consumption for the remaining life at period t
        # income_process: income from t to the end
        # init_wealth: initial wealth at period t

        life_end = len(income_process) - 1
        x0 = np.repeat(1,len(income_process))

        A = np.array([self.set_constraint_consume_coef(t,life_end) for t in range(life_end+1)])
        b = np.array([self.set_constraint_bound(t,income_process,init_wealth) for t in range(life_end+1)])

        cons = optimize.LinearConstraint(A,ub=b)

        bounds = [(self.consumption_floor, None)] * len(x0)

        obj = lambda x:-self.life_utility(x,income_process=income_process,init_wealth=init_wealth)

        solver = optimize.basinhopping(obj,x0,
                                    minimizer_kwargs={'method':'COBYLA','constraints':cons,'bounds':bounds},
                                    stepwise_factor = 0.5, T = 0.5, niter = 300)

        # solver = optimize.minimize(obj,x0,args=(obj_args,),constraints=cons,bounds=bounds)

        return solver.x
    
    def opt_life_path(self,income_process,init_wealth):

        # Generate ages, planned consumption, wealth for the remaining life at period t
        # Solve optimal planned consumption for the remaining life at period t
        # income_process: income from t to the end
        # init_wealth: initial wealth at period t

        ages = np.arange(0,self.death_age+1)
        consumption = self.opt_consumption(income_process,init_wealth)
        wealth = self.get_wealth_process(consumption,income_process,init_wealth)[1:]

        return {'age':ages,'consumption':consumption,'wealth':wealth}
    

    def plan_over_life(self,income_process,init_wealth):

        # Generate ages, planned consumption, wealth over the lifecycle

        life_plans = {}

        plan_path = self.opt_life_path(income_process,init_wealth)

        life_plans['age_0'] = {'age':np.arange(0,len(income_process)),
                            'consumption':plan_path['consumption'],
                            'wealth':plan_path['wealth']}

        for t in tqdm(range(1,len(income_process))):
            new_income = income_process[t:]
            new_init_wealth = life_plans[f'age_{t-1}']['wealth'][0]
            plan_path = self.opt_life_path(new_income,new_init_wealth)

            life_plans[f'age_{t}'] = {'age':np.arange(t,len(income_process)),
                                    'consumption':plan_path['consumption'],
                                    'wealth':plan_path['wealth']}
        
        self.life_plans = life_plans
    


class backwardSolver(life_path):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not hasattr(self, 'approx_consume_func'):
            self.approx_consume_func = {f'age_{t}':{} for t in range(self.death_age+1)}

    
    def set_w_grid(self,init_wealth):

        # set grid points for wealth

        _step = self.maxGrid / self.nGrid
        _stop = _step*self.nGrid

        self.w_grid = np.arange(_step,_stop,_step)*init_wealth
        self.init_wealth = init_wealth
        
    
    def last_period_obj(self,c,w,income_last):

        # FOC condition for the final period

        bequest = (w + income_last - c)*self.interest_rate

        u_prime_w = utility_1st_d(bequest,risk_coef=self.risk_coef,type=self.utility_type)

        u_prime_c = utility_1st_d(c,risk_coef=self.risk_coef,type=self.utility_type) 
        
        diff = u_prime_c - self.interest_rate*self.bequest_motive*u_prime_w

        return diff**2
    
    def solve_last_period(self,income_last):

        # Solve optimal consumption at each wealth level for the final period

        self.approx_consume_func[f'age_{self.death_age}'] = {'wealth':[],
                                                        'consumption':[]}
        
        w_grid = self.w_grid + max(0,self.consumption_floor - income_last)

        for w in w_grid:
            obj = lambda c: self.last_period_obj(c,w,income_last)
            x0 = 1.0
            bounds = [(self.consumption_floor, w+income_last)]

            # solver = optimize.basinhopping(obj,x0,
            #             minimizer_kwargs={'method':'COBYLA','bounds':bounds},
            #             stepwise_factor = 0.5, T = 0.5)
            
            solver = optimize.minimize(obj,x0,method='COBYLA',bounds=bounds)
            
            if isinstance(solver.x, np.ndarray):
                c = solver.x[0]
            else:
                c = solver.x

            self.approx_consume_func[f'age_{self.death_age}']['wealth'] += [w]
            self.approx_consume_func[f'age_{self.death_age}']['consumption'] += [c]

    def interpolate_consume(self,t,w):

        # Interpolate consumption to create an approx consumption function

        approx_func = self.approx_consume_func[f'age_{t}']
        
        cs = CubicSpline(x=approx_func['wealth'], y=approx_func['consumption'])

        return cs(w)
        
    def approx_bequest(self,t,w,new_income_process):

        # calculate the approx bequest at period t
        # w: wealth at period t
        # new_income_process: income from t to the end

        wealth_now = w
        income_now = new_income_process[0]
        consume_now = self.interpolate_consume(t,w)

        for age in range(t+1,self.death_age+1):
            wealth_now = self.get_next_wealth(wealth_now,income_now,consume_now)
            income_now = new_income_process[age-t]
            consume_now = self.interpolate_consume(age,w)
        
        bequest = self.get_next_wealth(wealth_now,income_now,consume_now)

        return bequest

    def next_period_1st_cond(self,t_next,w_next,new_income_process):

        # Calculate the RHS containing next-period state in FOC condition
        # t_next: pperiod t+1
        # w_next: wealth at period t+1
        # new_income_process: income from t to the end 

        consume_now = self.interpolate_consume(t_next,w_next)
        bequest = self.approx_bequest(t_next,w_next,new_income_process[1:])

        u_prime_c = utility_1st_d(consume_now,risk_coef=self.risk_coef,type=self.utility_type)
        u_prime_w = utility_1st_d(bequest,risk_coef=self.risk_coef,type=self.utility_type)

        tmp_c = u_prime_c*self.discount_factor
        tmp_w = u_prime_w*(1-self.discount_factor)*self.bequest_motive*self.interest_rate**(self.death_age+1-t_next)

        return self.interest_rate*(tmp_c + tmp_w)
    

    def each_period_obj(self,c,t,w,new_income_process):

        '''
        Calculate FOC condition for each period t (except the final period)
        The FOC condition: 
            u'(c_t) = next_period_1st_cond(t_next,w_next,new_income_process)
            w_next = R (w_t + y_t - c_t)

        c: consumption at period t
        w: wealth at period t
        new_income_process: income from t to the end 
        '''

        t_next = t+1
        w_next = self.get_next_wealth(wealth_now=w,income_now=new_income_process[0],consume_now=c)

        tmp_now = utility_1st_d(c,risk_coef=self.risk_coef,type=self.utility_type)
        tmp_next = self.next_period_1st_cond(t_next,w_next,new_income_process)

        diff = tmp_now - tmp_next

        return diff**2

    
    def consume_func_now(self,t,new_income_process):

        # Solve optimal consumption for each wealth level at period t

        self.approx_consume_func[f'age_{t}'] = {'wealth':[],
                                                'consumption':[]}
        
        w_grid = self.w_grid + max(0,self.consumption_floor - new_income_process[0])

        for w in w_grid:
            obj = lambda c:self.each_period_obj(c,t,w,new_income_process)
            x0 = 1.0
            bounds = [(self.consumption_floor, w+new_income_process[0])]

            # solver = optimize.basinhopping(obj,x0,
            #             minimizer_kwargs={'method':'COBYLA','bounds':bounds},
            #             stepwise_factor = 0.5, T = 0.5)
            
            solver = optimize.minimize(obj,x0,method='COBYLA',bounds=bounds)

            if isinstance(solver.x, np.ndarray):
                c = solver.x[0]
            else:
                c = solver.x
           
            self.approx_consume_func[f'age_{t}']['wealth'] += [w]
            self.approx_consume_func[f'age_{t}']['consumption'] += [c]


    def solve_consume_func(self,income_process,t_stop=0):

        # Iteratively solve consumption function from the final period to t_stop

        self.income_process = income_process

        # solve the last period
        income_last = income_process[-1]
        self.solve_last_period(income_last)

        print('Final period is solved')

        # backward induction
        for i in tqdm(range(1,self.death_age-t_stop+1)):
            t = self.death_age - i
            new_income_process = income_process[t:]
            self.consume_func_now(t,new_income_process)

    
    def gen_life_path(self):

        # Generate wealth, consumption, income over the lifecycle

        init_c = self.interpolate_consume(0,self.init_wealth)

        self.life_path = {'wealth':[self.init_wealth],
                          'consumption':[float(init_c)],
                          'income':self.income_process}
        
        for t in range(1,self.death_age+1):
            w = self.interest_rate*(self.life_path['wealth'][-1] - self.life_path['consumption'][-1] + self.income_process[t-1])
            c = self.interpolate_consume(t,w)
            self.life_path['wealth'] += [w]
            self.life_path['consumption'] += [float(c)]
        

            
        

        




    












        


    



            




