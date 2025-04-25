#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cupy as cp
import numpy as np
from qiskit.quantum_info import Pauli,PauliList,SparsePauliOp,Operator, Choi, Kraus, random_unitary, Statevector, entropy, partial_trace, DensityMatrix
from qiskit import *

import random
from random import randint

import scipy


import pandas as pd

from itertools import product
from functools import reduce

from tqdm import tqdm

CUPY_ACCELERATORS="cub", "cutensor"


# In[ ]:


def Opent(O,N1,N): ##Gives operator entanglement of a unitary O (given as numpy array) over a bipartition N1|N2 of N qubits
    q=2 #Qubit dimension
    N2=N-N1
    On=O.reshape(q**N1,q**N2,q**N1,q**N2)
    Ons=On.conj()
    opent=1-1/((q**N)**2)*np.real(cp.einsum("abcd,efgh,ebgd,afch",On,On,Ons,Ons,optimize='optimal').get())

    return opent

def Pauli_list(N): ##Generates a list of Pauli operators acting on N qubits as a PauliList qiskit object
    # Define the Pauli terms
    pauli_operators = ['I', 'X', 'Y', 'Z']

    # Generate all possible Pauli strings
    pauli_strings= [''.join(p) for p in product(pauli_operators, repeat=N)]
    pauli_list = PauliList(pauli_strings)

    return pauli_list

def tensor_product(list): #takes a list of qiskit operators and returns their tensor product
    tensor_product = reduce(lambda x, y: x.tensor(y), list)

    return tensor_product

def pauli_gen(N): ##Generates a list of single-site Pauli operators acting on N qubits as a list of of lists of qiskit Operator objects
    si = Operator(Pauli('I'))
    sx = Operator(Pauli('X'))
    sy = Operator(Pauli('Y'))
    sz = Operator(Pauli('Z'))

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor_product(op_list))

        op_list[n] = sy
        sy_list.append(tensor_product(op_list))

        op_list[n] = sz
        sz_list.append(tensor_product(op_list))

    return [sx_list, sy_list, sz_list]


def XYZ_ham(N,Jx,Jy,Jz,h): #Generates the Hamiltonian of a 1D chain of N qubits with XYZ interactions and a magnetic field h and PBC as a qiskit operator (PBC)
    sx_list,sy_list,sz_list=pauli_gen(N)
    H=0

    for n in range(N):
        H+=h*sz_list[n]
        H+=Jx*sx_list[n]@sx_list[(n+1)%N]+Jy*sy_list[n]@sy_list[(n+1)%N]+Jz*sz_list[n]@sz_list[(n+1)%N]
        
    return H

def TFIM_ham(N,g,h): #generate transverse field Ising model (PBC)
    
    sx_list,sy_list,sz_list=pauli_gen(N)
    H=0
    
    for n in range(N):
        H+=-1*sz_list[n]@sz_list[(n+1)%N]-g*sx_list[n]-h*sz_list[n]
    
    return H

# def sample_Pauli(N): ##Generates a random Pauli string acting on N qubits as a PauliList qiskit object
#     # Define the Pauli terms
#     pauli_operators = ['I', 'X', 'Y', 'Z']

#     # Generate all possible Pauli strings
#     pauli_string= ''.join(random.choices(pauli_operators, k=N))
#     pauli = Pauli(pauli_string)

#     return pauli    



    


# In[ ]:


## Compute the operator entanglement generation on Paulis for the PBC XYZ model for some choices of N, Jx=0.75, Jy=0.25, h=0.5 and various Jz around 0 (with sampling)
# dt_list=[0.01,0.05,0.1,0.15,0.2] -> Empirical values for N=2,..,6
# max_time ~= 15 -> Empirical value for N=2,..,6

N=11
Jx=0.75
Jy=0.25
h=0.5
Jz_list=np.linspace(-0.1,0.1,11)


N1=N//2
pauli_list=Pauli_list(N)
H_list=[XYZ_ham(N,Jx,Jy,Jz,h) for Jz in Jz_list]

tmin=1 #Set some minimum time for the evolution
dt=0.2 #Set the time step for the evolution
tol=2*10**(-2)#Set the tolerance for the long-time value mean error

flag=False
t=0
times=[0]
paulent_list=[]
opent_list=[]

pbar=tqdm()

while flag==False:
    Ut_list=[cp.array(scipy.linalg.expm(-1j*H_list[i].data*t)) for i in range(len(Jz_list))]

    flag_sample=False
    tol_sample=2*10**(-2)
    paulent_sample=[]
    min_samples=N**2

    index_list=np.arange(len(pauli_list))

    while flag_sample==False:
        index_sample=np.random.choice(index_list)
        index_list=np.delete(index_list,np.where(index_list==index_sample))

        paulent_sample.append([Opent(Ut_list[j]@cp.array(Operator(pauli_list[index_sample]).data)@Ut_list[j].conj().T,N1,N) for j in range(len(Jz_list))])

        if len(paulent_sample)>min_samples and np.all(1.96*np.std(paulent_sample, axis=0, ddof=1)/len(paulent_sample)**(1/2)< tol_sample) or len(paulent_sample)==len(pauli_list):
            flag_sample=True
        pbar.update(1)
    pbar.close        

    paulent_list.append(np.mean(paulent_sample, axis=0))
    opent_list.append([Opent(Ut_list[j],N1,N) for j in range(len(Jz_list))])

    if t>tmin and np.all(1.96*np.std(paulent_list, axis=0, ddof=1)/len(paulent_list)**(1/2) < tol ) and np.all(1.96*np.std(opent_list, axis=0, ddof=1)/len(paulent_list)**(1/2) < tol ):
        flag=True
        #This checks if after the minimum time the mean values of opent and paulent are stable by considering the standard error of the mean of the time point samples and a 95% confidence interval. 
    else:
        t+=dt
        times.append(t)

data_XYZ={'Jz':Jz_list,'Pauli_entangling':np.mean(paulent_list, axis=0),'std_Pauli_entangling':np.std(paulent_list, axis=0, ddof=1),'opent':np.mean(opent_list, axis=0),'std_opent':np.std(opent_list, axis=0, ddof=1)}
data_XYZ=pd.DataFrame(data_XYZ)
data_XYZ.to_csv('sample_XYZ_N'+str(N)+'_Jx'+str(Jx)+'_Jy'+str(Jy)+'_h'+str(h)+'.csv')

## Save full-time data for posterity
data_XYZ_full={
 ('Jz='+str(Jz_list[i]), 'Pauli_entangling'):np.array(paulent_list)[:,i] for i in range(len(Jz_list))
 }
data_XYZ_full.update({
 ('Jz='+str(Jz_list[i]), 'opent'):np.array(opent_list)[:,i] for i in range(len(Jz_list))
 })
columns = pd.MultiIndex.from_tuples(
 [('Jz='+str(Jz_list[i]), 'Pauli_entangling') for i in range(len(Jz_list))]+[('Jz='+str(Jz_list[i]), 'opent') for i in range(len(Jz_list))],
    names=['Jz', 'Time'])
data_XYZ_full=pd.DataFrame(data_XYZ_full, columns=columns, index=times)
data_XYZ_full.to_csv('sample_XYZ_N'+str(N)+'_Jx'+str(Jx)+'_Jy'+str(Jy)+'_h'+str(h)+'_full_time.csv')

