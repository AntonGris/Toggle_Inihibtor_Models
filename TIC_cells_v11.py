# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:32:25 2022
@author: Anton Grisheshckin
"""
#%%Importing packages#
import numpy as np
import matplotlib.pyplot as plt
#%% Model variables#
class modelvar :
    # Hill parameters    
    n = 3  # Hill function power in g eqn
    k1 = 2 #[AU]
    k2 = 2 #[AU]
    k3 = 0.1 #[AU]
    ################
    ψ = 0.1 # Max amplification for h [AU]hr^-1
    γ1 = 0.1 #Dilution rate hr^-1
    γ2 = 0.1
    I = 0 # dsRNA trigger [AU]hr^-1
    σ1 = 0.02 # Noise amplitudes hr^-1/2
    σ2 = 0.02 
    #################
    divcellrate= 4 # the "rate" of the cell division reaction
    cellnum = 1 #initial number of cells
    Vtot = 200 # total synthesis "chemical" available 
    ʎ = 0.3 # number of excitation events per hour
    L=0.4 # 'boost' from the trigger
    pr= 0.005 # Probability of differentiation from the excited state
    ##########################
#%%Deterministic part of the code#
def Hill(top, bottom, power):
    "Creates a Hill function"
    return (top**power)/(top**power+bottom**power)
#%%Stochastic part of the code #

def mu1(g: float,h : float,  _t: float, MaxAmpl) -> float:
    """
    Implement the drift part of the dg equation
    """
    return (modelvar.I+MaxAmpl*Hill(g,modelvar.k1,modelvar.n)*Hill(modelvar.k2,h,1)-modelvar.γ1*g)

def mu2(g: float, h : float,  _t: float) -> float:   
    """
    Implement the drift part of the dh equation
    """
    return (modelvar.ψ*(g/(modelvar.k3+g))-modelvar.γ2*h)

def sigma1(g: float, h:float,  _t: float):
    """
    Implement the BM part of dg equation
    """
    return modelvar.σ1 * g

def sigma2(g: float, h:float,  _t: float):
    """
    Implement the BM part of dh equation
    """
    return modelvar.σ2 * h

def dW1(delta_t: float, N : int):
    """
    Sample a random number at each call.
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t),size=N)

def dW2(delta_t: float, N : int):
    
    """
    Sample a random number at each call independent of dW1.
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t),size=N)
#%% Cell spliting algorithm #
def Next_Proliferation_Time(Reaction_Rate,Cell_Num):
    #Calculating time until next cell division
    r=np.random.uniform(0,1)
    τ= (np.log(1/r))/(Cell_Num*Reaction_Rate)
    return τ

def splitting_algorithm(Sim_time,DT,cell_division_rate,initial_cell_num,g_init,h_init,crit_value_g,crit_value_h):
   """
   *Sim time*: Simulation run time in hrs 
   *DT* : Timestep of the algorithm in hours
   *cell_division_rate* the rate constant in the "reaction" of proliferation A-->2A
   *crit_value_g* The critical value after which a cell differentiates
   
   """   
   
   #Here we shall at first as a simplicstic model, implement a reaction type proliferation i.e. A->2A
   
   Division_Time=Next_Proliferation_Time(cell_division_rate,initial_cell_num) #First division time

   differentcellnum=0 # Numnber of differentiated cells
   #Creating arrays to record different concentrations of chemicals in each cell#
   cell_array_g=g_init*np.ones(initial_cell_num)
   cell_array_h=h_init*np.ones(initial_cell_num)
   Event_site = np.zeros(initial_cell_num)
   #Creating arrays to record data#
   timearray=[0]
   cellnumarray=[1]
   diffcellarray=[0]
   ##############################################################################
   # best to use for loop rather than update T inside while loop
   for T in np.arange(0,Sim_time,DT):
       #Checking whether cells should divide#
       # OK - best that cells do not divide in synchrony - lets simply use a probabilty here?
       if cell_array_g.shape[0] ==0:
           break
       #ends the programme if there are no more proliferating cells
       while T > Division_Time :
             
           pos=np.random.randint(0,cell_array_g.shape[0]) #Selecting which cell to divide
           cell_array_g= np.concatenate( (cell_array_g[:pos],np.array([cell_array_g[pos]]),cell_array_g[pos:]))
           cell_array_h= np.concatenate( (cell_array_h[:pos],np.array([cell_array_h[pos]]),cell_array_h[pos:]))
           Event_site= np.concatenate((Event_site[:pos],np.array([0]),Event_site[pos:]))
           #Note in this variation the daughter cell doesn't get the same boost as the mother cell 
           #(Think about a realistic approach later)
           Division_Time=Division_Time+Next_Proliferation_Time(cell_division_rate,initial_cell_num)
       #####################################
       #Poisson#
       Events_num = np.random.poisson(DT*modelvar.ʎ*cell_array_g.shape[0])

       if Events_num != 0 :
           if Events_num >= cell_array_g.shape[0] :
               Event_site=modelvar.L*np.ones(cell_array_g.shape[0])
        
           else:
               for j in range(0,Events_num):
                   New_event_index=np.random.randint(0,cell_array_g.shape[0])
                   Event_site= np.concatenate((Event_site[:New_event_index],np.array([modelvar.L]),Event_site[New_event_index+1:]))
                   
       #######################################
       cost=np.sum(cell_array_g)
       # Calculating new values
       cell_array_g, cell_array_h = cell_array_g + Event_site + mu1(cell_array_g,cell_array_h,T,(modelvar.Vtot/cost)) * DT + sigma1(cell_array_g,cell_array_h,T) * dW1(DT,cell_array_g.shape[0]), \
                                    cell_array_h + mu2(cell_array_g,cell_array_h,T)*DT + sigma2(cell_array_g,cell_array_h,T)*dW2(DT,cell_array_h.shape[0])
       ##########################                             
       # Differentiation part#
       to_differentiate_g = cell_array_g>crit_value_g 
       to_differentiate_h =  cell_array_h>crit_value_h
       to_differentiate =np.logical_and(to_differentiate_g,to_differentiate_h)
       get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
       index_list=np.array(get_indexes(True,to_differentiate))
       if index_list !=[]:
           for k in index_list:
               new_differentiation=np.random.choice([True, False],p=[modelvar.pr , 1-modelvar.pr])
               to_differentiate[k]=new_differentiation
           differentcellnum=differentcellnum+to_differentiate.sum()
           cell_array_g, cell_array_h = cell_array_g[~to_differentiate],cell_array_h[~to_differentiate]
           Event_site = Event_site[~to_differentiate]
       #Recording the relevant values#    
       timearray.append(T)
       cellnumarray.append(cell_array_g.shape[0])
       diffcellarray.append(differentcellnum)
       ###################
       
   return(cell_array_g.shape[0],cell_array_g,cell_array_h,differentcellnum,timearray,cellnumarray,diffcellarray)
#%%Main body #
a,b,c,d,timearr,cellnumarr,diffnumarr=splitting_algorithm(1000,0.1,modelvar.divcellrate,modelvar.cellnum,0.1,0.1,6,0.6)

plt.plot(timearr,cellnumarr)
plt.xlabel("Time (hours)")
plt.ylabel("Number of cells")
plt.show()
