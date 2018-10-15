# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 08:57:40 2018

@author: mducoffe

heuristic for the k center problem
with distance metric d
"""

# Mixed Integer Programming Solver
from ortools.linear_solver import pywraplp
import numpy as np

def create_variables(solver, N=100):
    # create u : N variables
    u = [solver.IntVar(0.0, 1.1, 'u_{}'.format(i)) for i in range(N)]
    # create w: NxN variables
    w = [[solver.IntVar(-0.1, 1.1, 'w_{}_{}'.format(i, j)) for i in range(N)] for j in range(N)]
    # create eta: NxN variables
    e=  [[solver.IntVar(0.0, 1.1, 'e_{}_{}'.format(i, j)) for i in range(N)] for j in range(N)]
    
    return u, w, e
    
def create_constraints(solver, u, w, e, dict_values, delta):
    b = dict_values['b']
    s_0 = dict_values['s_0']

    N = len(u)
    constraints=[]    
    
    constraints_1 = [[solver.Constraint(-solver.infinity(), 0) for i in range(N)] for j in range(N)]
    constraints_2 =[]
    constraints_3 = []
    constraints_4 = [solver.Constraint(1, 1) for i in range(N)] 
    constraint_5 = solver.Constraint(len(s_0)+b, len(s_0)+b)
    for i in range(N):
        constraint_5.SetCoefficient(u[i], 1.)
        
        if i in s_0:
            constraint_3_tmp = solver.Constraint(1, 1)
            constraint_3_tmp.SetCoefficient(u[i], 1.)
            constraints_3.append(constraint_3_tmp)
        
        for j in range(N):
            
            constraints_1[i][j].SetCoefficient(u[i], -1.)
            constraints_1[i][j].SetCoefficient(w[i][j], 1.)
            
            if delta[i][j]==0:
                constraint_2_tmp = solver.Constraint(0, 0)
                constraint_2_tmp.SetCoefficient(w[j][i], 1.)
                constraint_2_tmp.SetCoefficient(e[j][i], -1.)
                constraints_2.append(constraint_2_tmp)
            constraints_4[i].SetCoefficient(w[j][i], 1.)
           
            
    
    constraints+=constraints_1
    constraints+=constraints_2
    constraints+=constraints_3
    #constraints+=constraints_4
    #constraints.append(constraint_5)
    
    return constraints
    
def create_objective(solver, e):
    objective = solver.Objective()
    N = len(e)
    for i in range(N):
        for j in range(N):
            objective.SetCoefficient(e[i][j], 1.)
    objective.SetMinimization()
    
def feasible(dict_values, delta):
    solver = pywraplp.Solver('CoreSetIntegerProblem',
                           pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    N = len(delta)
    u,w, e = create_variables(solver, N)
    constraints=create_constraints(solver, u, w, e, dict_values, delta)
    objective=create_objective(solver, e)
    
    # Invoke the solver and display the results
    result_status=solver.Solve()
    # the solution looks legit
    assert solver.VerifySolution(1e-7, True)
    
    u_values = [u[i].solution_value() for i in range(N)]
    e_values=[[e[j][i].solution_value() for i in range(N)] for j in range(N)]
    e_values = np.array(e_values)
    
    outliers = dict_values['outliers']
    return np.sum(e_values)<=outliers, u_values
    
    """
    for variable in u:
        print('%s=%d'%(variable.name(), variable.solution_value()))
        
    e_values=[[e[j][i].solution_value() for i in range(N)] for j in range(N)]
    e_values = np.array(e_values)
    #print(e_values)
    
    w_values=[[w[j][i].solution_value() for i in range(N)] for j in range(N)]
    w_values = np.array(w_values)
    #print(w_values)
    """
    
#%%
def build_delta(model, x, threshold):
    
    
    N = len(x)
    delta = np.zeros((N,N), dtype='uint8')
    lb = np.inf
    ub = threshold
    for i in range(N):
        for j in range(N):
            distance = np.linalg.norm(x[i] - x[j])
            if distance <= threshold:
                delta[i,j]=1
                delta[j,i]=1
                if distance >ub:
                    ub = distance
            else:
                if distance<lb:
                    lb = distance

    return delta, lb, ub

def greedy_k_center(model, x, c_0, b):
    centers = range(c_0)
    N = len(x)
    candidates = range(c_0, N)
    
    for query in range(b):
        distances = [np.min([np.linalg.norm(x[i]-x[j]) for j in centers]) for i in candidates]
        new_cluster_index = np.argmax(distances)
        centers.append(len(centers)+new_cluster_index)
        candidates.pop(new_cluster_index)
    # recompute distance
    distances = [np.min([np.linalg.norm(x[i]-x[j]) for j in centers]) for i in candidates]
    return np.max(distances)
    

def robust_k_center(model, labelled_data, unlabelled_data, b):
    x_0 = model.predict(labelled_data[0])
    x_1 = model.predict(unlabelled_data[0])
    x = np.concatenate([x_0, x_1], axis=0)
    opt = greedy_k_center(model, x, len(labelled_data[0]),b)
    s_0=range(len(labelled_data[0]))
    N = len(labelled_data[0]) + len(unlabelled_data[0])
    outliers=1e-4*N
    dict_values={'s_0':s_0, 'outliers': outliers, 'b':b}
    ub=opt; lb=opt/2.
    assignment=None
    print(opt)
    while True:
        print(('iter', ub, lb))
        threshold= (ub+lb)/2.
        delta, lb_, ub_ = build_delta(model, x, threshold)
        bool_test, centroids = feasible(dict_values, delta)
        if bool_test:
            ub=ub_
            assignment=centroids
        else:
            lb=lb_
        if lb >=ub or (ub - lb)<5*1e-3:
            break

    n_start = len(labelled_data[0])
    assignment = np.array(assignment[n_start:])
    return np.where(assignment==1), np.where(assignment==0)
 
                                                                                                                                                                                                                                         
 
#%%
def toy_example():
    delta=[]
    delta.append([1,1,1,0,0,0,0,0]) #0
    delta.append([1,1,0,0,0,0,0,0]) #1
    delta.append([1,0,1,0,0,0,0,0]) #2
    delta.append([0,0,0,1,0,0,0,0]) #3
    delta.append([0,0,0,0,1,1,1,0]) #4
    delta.append([0,0,0,0,1,1,0,0]) #5
    delta.append([0,0,0,0,1,0,1,0]) #6
    delta.append([0,0,0,0,0,0,0,1]) #7
    
    delta = np.array(delta).astype(np.uint)
    
    s_0=[4]
    outliers=1
    b=1
    dict_values={'s_0':s_0, 'outliers': outliers, 'b':b}
    
    bool_test, centroids = feasible(dict_values, delta)
    print(bool_test)
    
    
                    
    
  #%%  
if __name__=="__main__":
    # Instantiate a mixed-integer solver, naming it SolveIntegerProblem.
    toy_example()
