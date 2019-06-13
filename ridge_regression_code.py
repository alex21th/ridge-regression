#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries needed for optimization.
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint

# Read dataset file.
data = np.loadtxt("death_rate.dat")
A = data[:,:-1]     # Matrix of explanatory variables.    m·n
y = data[:,-1]      # Vector of the response variable.    m·1

m = (A.shape)[0]    # Number of rows of matrix A.
n = (A.shape)[1]    # Number of columns of matrix A.

w = np.zeros(n)     # Vector of weights.                  n·1
gamma = 0           # Intercept parameter.                1·1
e = np.ones(m)      # Vector of ones.                     m·1

''' OBJECTIVE FUNCTION f(x) '''
def obje_f(x):
    # 'x' is a vector containing:    the vector of weights 'w'
    #                                the intercept parameter 'gamma'
    w = x[:-1]
    gamma = x[-1]
    
    # Returns the evaluation of the objective function f(x).    1·1
    form = np.matmul(A,w) + gamma*e - y    # Aw + ge - y
    return(1/2 * np.dot(form,form))

''' GRADIENT OF THE OBJECTIVE FUNCTION ∇f(x) '''
grad = np.zeros(n+1)    # Initialization of the gradient vector.

def grad_f(x):
    # 'x' is a vector containing:    the vector of weights 'w'
    #                                the intercept parameter 'gamma'
    w = x[:-1]
    gamma = x[-1]
    
    # Returns the evaluation of the gradient vector ∇f(x).    (n+1)·1
    grad[:-1] = np.matmul(np.transpose(A),(np.matmul(A,w) + gamma*e - y))           # A'(Aw + ge - y)
    grad[-1] = np.matmul(np.transpose(e),np.matmul(A,w)) + gamma*m - np.dot(e,y)    # e'Aw + gm - e'y
    return(grad)

''' HESSIAN OF THE OBJECTIVE FUNCTION ∇^(2)f(x) '''
hess = np.zeros((n+1,n+1))                    # Initialization of the hessian matrix.
hess[:n,:n] = np.matmul(np.transpose(A),A)    # Computation of the first 'n·n' inputs.    A'*A
hess[:n,-1] = np.matmul(np.transpose(A),e)    # Computation of the 'n·(n+1)' column.      A'*e
hess[-1,:n] = np.transpose(hess[:n,-1])       # Computation of the '(n+1)·n' row.         e'*A
hess[-1,-1] = m                               # Computation of the '(n+1)·(n+1)' element.   m

def hess_f(x):
    # Returns the evaluation of the hessian matrix ∇^(2)f(x).    (n+1)·(n+1)
    return(hess)

''' CONSTRAINT CONDITION h(x) '''
def cons_f(x):
    # 'x' is a vector containing:    the vector of weights 'w'
    #                                the intercept parameter 'gamma'
    w = x[:-1]
    
    # Returns the evaluation of the constraint condition h(x).    1·1
    return(np.dot(w,w))

''' JACOBIAN OF THE CONSTRAINT ∇h(x) '''
jaco = np.zeros(n+1)    # Initialization of the jacobian matrix.

def cons_J(x):
    # 'x' is a vector containing:    the vector of weights 'w'
    #                                the intercept parameter 'gamma'
    w = x[:-1]
    
    # Returns the evaluation of the jacobian matrix ∇h(x).    (n+1)·1
    jaco[:-1] = 2*w
    return(jaco)

''' HESSIAN OF THE CONSTRAINT ∇h^(2)(x) '''
hessC = 2*np.eye(n+1)
hessC[-1,-1] = 0         # Initialization of the hessian matrix.

def cons_H(x, v):
    # Returns the evaluation of the sum of Lagrange multipliers by the hessian matrix.    1·1
    return(v[0]*hessC)

# Definition of the nonlinear constraint (with the hiperparameter 't').
t = 1.0
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, t, jac = cons_J, hess = cons_H)
x0 = np.zeros(n+1)    # Definition of the starting point.

sol = minimize(obje_f, x0, method = 'trust-constr', jac = grad_f, hess = hess_f,    # Solve problem.
               constraints = [nonlinear_constraint], options = {'verbose': 1})

np.set_printoptions(suppress = True)
print(sol)            # Display of the obtained solution.


# Condition 1.
cons_f(sol.x) - t # <-- g(x*)

# Condition 2.
sol.lagrangian_grad # <-- ∇L(x*)

# Condition 3.
print(sol.v) # <-- mu
(sol.v[0][0])*(cons_f(sol.x)-1) # <-- mu·g(x*)