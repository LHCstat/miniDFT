import numpy as np
import basis
import hamiltonian
from scipy.sparse.linalg import LinearOperator, eigsh
import system
def get_eigenvalues(N,bands,k,potential):
    #给出c的维度数量、占据带数量、k点、势能，返回本征值和本征向量

    # 2. 封装为线性算子（告诉 eigsh 如何计算 H @ c）
    A = LinearOperator(
        shape=(N, N),
        matvec=lambda x: hamiltonian.applyHc(potential, x, k),   # 输入 c，返回 H @ c
        dtype=float               # 如果波函数是复数的；实数则用 float
    )

    # 3. 调用求解 H c = E c
    #    k: 你想求几个本征值（比如占据带数）
    eigenvalues, eigenvectors = eigsh(
        A,
        k=bands,          # 特征值个数
        which='SA',           # SA = 最小值（基态）
        tol=1e-8,
        maxiter=1000
    )

    # eigenvectors 的列是特征向量，转置一下方便后续用 (num_bands, N)
    eigenvectors = eigenvectors.T
    return eigenvalues, eigenvectors