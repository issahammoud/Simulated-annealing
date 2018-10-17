#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-info">
#     <h2>Imports</h2>
# </div>

# In[131]:


import numpy as np


# <div class="alert alert-info">
#     <h2>Generate random values - Alternative A</h2>
# </div>

# In[132]:


np.random.seed(int(np.random.uniform(0,10000)))

def generate_rand_vector(S):
    """
    generate a random vector x belonging to set S 
    with a uniform distribution.
    """

    # generate a random value following a uniform 
    # distribution over each dimension of S.
    x = [np.random.uniform(s[0],s[1]) for s in S]
    
    return x


# <div class="alert alert-info">
#     <h2>Initialize</h2>
# </div>

# In[133]:


def init_c(f, S, m, ksi = 0.9):
    """
    This function initializes the value of the control parameter c.
    It returns the initial control paramter c0 and a realizable value x.
    f is the objective function.
    S is a 2D array representing the realizable values.
    m is the number of trials.
    ksi is the initial acceptance ratio.
    """
    
    # a list that should contain the positive 
    # deltas (f(y) - f(x))
    delta = []
    
    # generate a value belonging to S
    x = generate_rand_vector(S)
    
    # m is the number of trials 
    for i in range(m):
        
        # generate a value belonging to S
        y = generate_rand_vector(S)
        
        deltaf = f(y) - f(x)
        
        if(deltaf > 0):
            delta.append(deltaf)
        
        x = y
      
    delta = np.array(delta)
    
    # m2 is the number of trials with deltaf > 0
    m2 = len(delta)
    
    # m1 is the number of trials with deltaf <= 0
    m1 = m - m2
    
    delta_mean = np.mean(delta)
    
    # if m1 == m2, c0 will be infinity, and we are
    # interested to have m1 and m2 very near so c is high
    
    if(m1 == m2):
        m2+=1
        m1-=1
     
    # if m1 greater than m2, we have c negative, so we inverse them
    
    if(m1 > m2):
        tmp = m1
        m1 = m2
        m2 = tmp
    
    
    c0 = delta_mean/np.log(m2/(m2*ksi + m1*(1 - ksi)))

    return c0


# <div class="alert alert-info">
#     <h2>Simulated Annealing algorithm</h2>
# </div>

# In[134]:


def SimulatedAnnealing(f,n,x):
    """
    This function calculate the global optimun of
    a function f defined over a set S, using the 
    simulated annealing algorithm.
    """
    
    S = np.ones((n,2))
    
    S[:,0] = -100
    S[:,1] = 100
    
    # get c0 for m = 50
    c0 = init_c(f,S, 50)
    
    stopcriterion = False
    
    L0 = 20
    
    eps_s = 0.0001
    
    delta = 0.1
    
    c = c0
    
    # a list to stock the initial evaluations f(y's) for
    # the first Markov chain. It is used for the stop criterion.
    inital_Markov_chain_values = []
    
    inital_Markov_chain_values.append(f(x))
    
    # a boolean to indicate if the algorithm is in the first 
    # Markov chain or not.
    first_chain = True
    
    previous_accepted_values = []
    
    
    while(stopcriterion == False):
        
        # list of accepted values for each round.
        accepted_values = []
        
#         accepted_values.append(f(x))
        
        for i in range(n*L0):
            
            y = generate_rand_vector(S)
            
            # This condition is used when the value of c = 0, 
            # because we can't divide by zero in the exponential term.
            if(c == 0):
                acc = 0
            else :
                acc = np.exp(-(f(y) - f(x))/c)
        
              
            # acceptance criterion
            if(f(y) - f(x) <= 0 or acc > np.random.rand()):
                
                x = y
                                
                accepted_values.append(f(y))
                
                # Save the first Markov chain
                if(first_chain == True):
                    
                    inital_Markov_chain_values.append(f(y))
        
        # Test for the stop criterion if we are not in the first Markov chain
        if(first_chain == False):
            
            # the term f(c0)(bar) in the paper
            f_0= np.mean(inital_Markov_chain_values)
            
            # check if there are accpeted values because
            # the np.mean will return nan in this case 
            if(len(accepted_values) == 0):
                f_mean = 0
            else :
                f_mean = np.mean(accepted_values)
                
            # f_ is the term delta fs(c)/dc in the paper
            f_ = (f_mean - np.mean(previous_accepted_values))/(c - previous_c)
            
            # check stop criterion
            if(np.abs(f_ * c / f_0) < eps_s ):
                stopcriterion = True
            
        if(len(accepted_values) != 0): 
            
            previous_accepted_values = accepted_values
            
            previous_c = c
        
        # check if there are accpeted values because
        # the np.std will return nan in this case 
        if(len(accepted_values) <= 1 or np.std(accepted_values) == 0):
            c = 0
        else:
            c = c/(1 + (c*np.log(1+delta))/(3*np.std(accepted_values)))
            
        
        # a boolean to indicate that the algo terminated the first Markov chain
        first_chain = False
    
    
        
    return np.mean(previous_accepted_values)


# <div class="alert alert-info">
#     <h2>Test the algorithm on a simple parabola function</h2>
# </div>

# In[135]:


def parabola(X):
    """
    A function to test the algorithm.
    """
    r = 0
    for x in X:
        r+=np.abs(x)
        
    return r


# In[136]:


SimulatedAnnealing(parabola,3,[95.0])

