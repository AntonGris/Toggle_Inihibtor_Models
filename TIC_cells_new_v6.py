#%% Documentation and package importing
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:25:23 2022

@author: Anton Grisheshkin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
x_data=[]
y_data=[]
#%% Variables used in the model
class Modelvar :
    # Hill parameters    
    n = 3  # Hill function power in g eqn
    k1 = 2 #[AU]
    k2 = 2 #[AU]
    k3 = 0.1 #[AU]
    ################
    ψ = 1 # Max amplification for h [AU]hr^-1
    γ1 = 1 #Dilution rate hr^-1
    γ2 = 1
    σ1 = 1 # Noise amplitudes hr^-1/2
    σ2 = 1
    #################
    divcellprob= 0.01 # probability of a cell dividing in 1 hr
    cellnum = 12 #initial number of cells
    Vtot =0.005 # total synthesis "chemical" available 
    excitationprob=0.4 # probability of an excitation event occuring in 1 hr
    L=1.6 # 'boost' from the trigger
    diffprob= 0.4 # Probability of differentiation from the excited state
    ##########################
#%% Functions used in the model
def Hill(top, bottom, power):
    "Creates a Hill function"
    return (top**power)/(top**power+bottom**power)

def mu1(g: float,h : float,  _t: float, MaxAmpl) -> float:
    """
    Implement the drift part of the dg equation
    """
    return (MaxAmpl*Hill(g,Modelvar.k1,Modelvar.n)*Hill(Modelvar.k2,h,1)-Modelvar.γ1*g)

def mu2(g: float, h : float,  _t: float) -> float:   
    """
    Implement the drift part of the dh equation
    """
    return (Modelvar.ψ*Hill(g,Modelvar.k3,1)-Modelvar.γ2*h)

def sigma1(g: float, h:float,  _t: float):
    """
    Implement the BM part of dg equation
    """
    return Modelvar.σ1 * g

def sigma2(g: float, h:float,  _t: float):
    """
    Implement the BM part of dh equation
    """
    return Modelvar.σ2 * h

def dW1(delta_t: float, N : int):
    """``
    Sample a random number at each call.
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t),size=N)

def dW2(delta_t: float, N : int):
    
    """
    Sample a random number at each call independent of dW1.
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t),size=N)
#%% Functions preforming the model operations

def tic_splitting(Sim_Time,DT,initial_cell_num,g_init,h_init,crit_value_g,crit_value_h):
    """
    *Sim time*: Simulation run time in hrs 
    *DT* : Timestep of the algorithm in hrs (has to be <1hr for probabilities to work)
    *initial_cell_num* Number of cells at T=0
    *cell_div_prob* probability that a single cell divides in 1hr
    *g_init,h_init* Initial concentrations of g & h
    *crit_value_g , h* The critical values after which a cell begins to differentiate
    """  
    #Cell array to capture all of the information of the system
    #Columns are cells, rows are chemicals
    #Currently Row1=g conc, Row2= h conc Row3= Excitability of the cell (0=excited, L=excited)
    cell_array=np.array((g_init*np.ones(initial_cell_num),h_init*np.ones(initial_cell_num)))
    # Arrays for recording the relevant data
    timearray=[0]
    cellnumarray=[initial_cell_num]
    lowcellarray=[initial_cell_num]
    # Cost array just for seeing its behviour, Temporary
    costarray=[np.sum(cell_array[0,:])]
    ####################################
    #Main for loop for the algorithm
    for T in np.arange(0,Sim_Time,DT):
        # Stops alg if no cells are present or too many cells are present
        if cell_array.shape[1]==0 or cell_array.shape[1]>50000:
            break
        #%%%Division part#
        #Implemented using a simple coin toss, all cells assumed independent and uniform probabiliy of division over time
        #As such using a Bin dist to find out number of divisions and then randomly selecting cells to divide
        Num_of_divisions=np.random.binomial(cell_array.shape[1], Modelvar.divcellprob*DT)
        Position_of_divisions=np.random.randint(0,cell_array.shape[1],size=Num_of_divisions)
        if Position_of_divisions.shape[0] !=0:
            for j in Position_of_divisions:
                cell_array= np.concatenate( (cell_array[:,:j],cell_array[:,[j]],cell_array[:,j:]),axis=1)
        #%%%Excitability part #
        #Here we implement cells getting stochastically "excited" using a simple coin flip
        #As such it is a similar setup to division
        to_excite = cell_array[0,:]<1.5
        excitetoss=np.random.choice([True, False],size=cell_array.shape[1],p=[DT*Modelvar.excitationprob , 1-DT*Modelvar.excitationprob])
        to_excite=np.logical_and(to_excite,excitetoss)
        cell_array[0,:]=cell_array[0,:]+Modelvar.L*to_excite
        #%%% Chemical reaction eqns #
        cost=np.sum(cell_array[0,:])
        cell_array[0,:], cell_array[1,:] = cell_array[0,:] + mu1(cell_array[0,:],cell_array[1,:],T,(Modelvar.Vtot*(cost))) * DT + sigma1(cell_array[0,:],cell_array[1,:],T) * dW1(DT,cell_array.shape[1]), \
                                     cell_array[1,:] + mu2(cell_array[0,:],cell_array[1,:],T)*DT + sigma2(cell_array[0,:],cell_array[1,:],T)*dW2(DT,cell_array.shape[1])
        #%%% Differentiation part # 
        #Checks if a cell can differentiate (if the concentrations are over a particular value)
        to_differentiate_g = cell_array[0,:]>crit_value_g
        to_differentiate_h =  cell_array[1,:]>crit_value_h
        to_differentiate =np.logical_and(to_differentiate_g,to_differentiate_h) #Array of True/False to capture differentiation
        #"Tossing a coin to determine if differentiation actually happens
        tossdiff=np.random.choice([True, False],size=cell_array.shape[1],p=[DT*Modelvar.diffprob , 1-DT*Modelvar.diffprob])
        to_differentiate=np.logical_and(to_differentiate,tossdiff)
        cell_array=cell_array[:,~to_differentiate]
        ##########
        lowcells= cell_array[0,:]<1.5
        lowcelly=cell_array[:,lowcells]
        #Recording the relevant values#    
        timearray.append(T)
        cellnumarray.append(cell_array.shape[1])
        costarray.append(cost)
        lowcellarray.append(lowcelly.shape[1])
    return(timearray,cellnumarray,costarray,cell_array,lowcellarray)
#%%

def phase_plane_plot(i, range_x = (-1,3), range_y = (0,8),
                     num_grid_points = 20):
    axs.clear()
    def model(x, t = 0):
        'Toggle Inhibitor model'
        h, g = x
        dh = mu2(g,h,t)
        dg = mu1(g,h,t,Modelvar.Vtot*costarr[i])
        return np.array([dh, dg])
    
    '''
    Simple implementation of the phase plane plot in matplotlib.
    Original function done by Dominik Krzeminski (dokato)'
    Input:
    -----
      *model* : function
        function that takes numpy.array as input with two elements
        representing two state variables
      *range_x* = (-1, 1) : tuple
        range of x axis
      *range_y* = None : tuple
        range of yints* = 50 : int axis; if None, the same range as *range_x*
      *num_grid_points* = 50 : int
        number of samples on grid 
    '''
    
    x_ = np.linspace(range_x[0], range_x[1], num_grid_points)                                                             
    y_ = np.linspace(range_y[0], range_y[1], num_grid_points)                                                             

    grid = np.meshgrid(x_, y_)
        
    dfmat = np.zeros((num_grid_points, num_grid_points, 2))
    for nx in range(num_grid_points):
        for ny in range(num_grid_points):
            df = model([grid[0][nx,ny], grid[1][nx,ny]])
            dfmat[nx, ny, 0] = df[0]
            dfmat[nx, ny, 1] = df[1]
    
    plt.contour(grid[0], grid[1], dfmat[:, :, 0], [0], colors = 'r') 
    plt.contour(grid[0], grid[1], dfmat[:, :, 1], [0], colors = 'g')
    print(i)

#%% Main body of the code
fig,axs=plt.subplots(ncols=2,nrows=2)
timearr,cellnumarr,costarr,cellarr,lowcellarr=tic_splitting(3000,0.1, Modelvar.cellnum,0.1,0.1,2.5,0.94)
axs[0,0].plot(timearr,np.array(cellnumarr))  #-2.4627/(2.5*Modelvar.Vtot)
axs[0,0].set_xlabel("Time (hours)")
axs[0,0].set_ylabel("Number of cells")

axs[1,0].plot(timearr,Modelvar.Vtot*np.array(costarr)-5.6)
axs[1,0].set_xlabel("Time (hours)")
axs[1,0].set_ylabel("Cost")
#axs[1,0].set_ylim([-0.5,0.5])
axs[0,1].plot(timearr,lowcellarr)
axs[0,1].set_xlabel("Time (hours)")
axs[0,1].set_ylabel("Cells in low s.s")
#plt.yscale('log')
plt.show()

#fig,axs=plt.subplots()
#axs.set_xlim(-1,3)
#axs.set_ylim(0,8)
#line, =axs.plot(0,0)
#anim = animation.FuncAnimation(fig, phase_plane_plot ,frames=np.arange(0,len(timearr),10), repeat=False, interval=1,save_count=sys.maxsize)
#writergif = animation.FFMpegWriter(fps=30)
#anim.save("video.mp4",writer=writergif)
