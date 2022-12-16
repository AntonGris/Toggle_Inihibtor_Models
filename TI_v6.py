# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:02:25 2022

@author: Anton Grisheshckin
"""
#Importing packages#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

####################
class modelvar :
    V = 3.6
    n = 3 
    k1 = 2 
    k2 = 2 
    k3 = 0.1
    ψ = 1 
    γ1 = 1 
    γ2 = 1 
    I = 0
    σ1 = 0.02 
    σ2 = 0.02 
##########################

#Deterministic part of the code#
def Hill(top, bottom, power):
    "Creates a Hill function"
    return (top**power)/(top**power+bottom**power)

def f1(g,h,MaxAmpl):
    "Deterministic part of the first equation (dg/dt)"
    return (modelvar.I+(MaxAmpl)*Hill(g,modelvar.k1,modelvar.n)*Hill(modelvar.k2,h,1)-modelvar.γ1*g)

def f2(g,h) :
    "Determenistic part of the second equation (dh/dt)"
    return (modelvar.ψ*(g/(modelvar.k3+g))-modelvar.γ2*h)


def model(x, t = 0):
    'Toggle Inhibitor model'
    h, g = x
    dh = mu2(g,h,t)
    dg = mu1(g,h,t,modelvar.V)
    return np.array([dh, dg])

def phase_plane_plot(model, range_x = (-1,2), range_y = (-0.5,6),
                     num_grid_points = 500, arrow_grid_points = 10):
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
      *arrow_grid_points* : int
        numeber of points for arrows to go on the grid w.r.t the axis (i.e.number of actual arrows is this num sqauared)
    '''
    if range_y is None:
        range_y = range_x
    x_ = np.linspace(range_x[0], range_x[1], num_grid_points)                                                             
    y_ = np.linspace(range_y[0], range_y[1], num_grid_points)                                                             
    
    arrowx_ = np.linspace(range_x[0], range_x[1], arrow_grid_points)
    arrowy_ = np.linspace(range_y[0], range_y[1], arrow_grid_points)
    
    
    grid = np.meshgrid(x_, y_)
    
    arrowgrid=np.meshgrid(arrowx_, arrowy_)
    
    dfmat = np.zeros((num_grid_points, num_grid_points, 2))
    arrowdfmat=np.zeros((arrow_grid_points, arrow_grid_points,2))
    for nx in range(num_grid_points):
        for ny in range(num_grid_points):
            df = model([grid[0][nx,ny], grid[1][nx,ny]])
            dfmat[nx, ny, 0] = df[0]
            dfmat[nx, ny, 1] = df[1]
            
    for nx in range(arrow_grid_points):
        for ny in range(arrow_grid_points):
            arrowdf = model([arrowgrid[0][nx,ny], arrowgrid[1][nx,ny]])
            arrowdfmat[nx, ny, 0] = arrowdf[0]
            arrowdfmat[nx, ny, 1] = arrowdf[1]

    plt.quiver(arrowgrid[0], arrowgrid[1], arrowdfmat[:, :, 0], arrowdfmat[:, :, 1],)
    plt.contour(grid[0], grid[1], dfmat[:, :, 0], [0], colors = 'r')
    plt.contour(grid[0], grid[1], dfmat[:, :, 1], [0], colors = 'g')


###########################################

#Stochastic part of the code#    
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
def sigma1(g: float, h:float,  _t: float) -> float:
    """
    Implement the BM part of dg equation
    """
    return modelvar.σ1

def sigma2(g: float, h:float,  _t: float) -> float:
    """
    Implement the BM part of dh equation
    """
    return modelvar.σ2

def dW1(delta_t: float) -> float:
    """
    Sample a random number at each call.
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

def dW2(delta_t: float) -> float:
    """
    Sample a random number at each call independent of dW1.
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

def run_simulation():
    """
    Return the result of one full simulation.
    """
    T_INIT = 1
    T_END  = 300
    N      = 10000  # Compute 1000 grid points
    DT     = float(T_END - T_INIT) / N
    TS = np.arange(T_INIT, T_END + DT, DT)

    Y_INIT = 0.1
    X_INIT = 0.1
    xs= np.zeros(N+1)
    ys = np.zeros(N + 1)
    xs[0]= X_INIT
    ys[0] = Y_INIT
    for i in range(1, TS.size):
        t = T_INIT + (i - 1) * DT
        y = ys[i - 1]
        x=xs[i-1]
        ys[i] = y + mu1(y,x,t,modelvar.V) * DT + sigma1(y,x,t) * dW1(DT)
        xs[i]= x + mu2(y,x,t)*DT + sigma2(y,x,t)*dW2(DT)
    return xs, ys

def plot_simulations(num_sims: int) -> None:
    """
    Plot several simulations in one image.
    """
    for _ in range(num_sims):
        plt.plot(*run_simulation())
    plt.xlabel("h")
    plt.ylabel("g")
    plt.show()
########################################################
phase_plane_plot(model, range_x = (-1, 2))



if __name__=="__main__":
    NUM_SIMS = 0
    plot_simulations(NUM_SIMS)
for i in np.linspace(0, 1):
    plt.show()