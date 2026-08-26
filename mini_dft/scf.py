import system
import potentials
import eigensolver
import numpy as np
import hamiltonian
import density


System = system.system('input.yaml')
alpha = 0.5
iter = 0

Potential = potentials.get_potential(System.initial_density,System)
N = 1#暂且设置为等于1
density_old = System.initial_density
density_new = System.initial_density+1
while np.max(np.abs(density_new-density_old))>1e-6:
    iter = iter + 1
    E_total = 0
    Density = 0
    for ik in System.kpoints:
        G = basis.get_G(System,ik)
        E,c = eigensolver.get_eigenvalues(N,System.bands,ik,Potential)
        for i in range(N):
            c_i = c[i]
            E_i = E[i]
            E_total = E_total+E_i
            Density = Density +density.get_density(c_i)
    density_new = (1-alpha)*density_old + alpha*Density
    print('第',iter,'次迭代，E_total为',E_total)