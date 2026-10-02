import numpy as np
from scipy.linalg import block_diag, sqrtm
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import scipy.io
from itertools import combinations

# ----------------------------
# Helper functions
# ----------------------------

def canonicalM(N):
    if N % 2 != 0:
        raise ValueError("Matrix dimension N must be even.")

    matrix = np.zeros((N, N), dtype=complex)
    mid = N // 2

    matrix[0, 0] = 1
    matrix[0, mid] = 1
    matrix[1, 0] = -1j
    matrix[1, mid] = 1j

    for i in range(2, N, 2):
        matrix[i] = np.roll(matrix[i - 2], 1)
        matrix[i + 1] = np.roll(matrix[i - 1], 1)

    return matrix


def takeSubMatrix(CM):
    return CM[0:4, 0:4]


def takeSub(CM, i, j):
    return np.array([
        [CM[i, i], CM[i, i+1], CM[i, j], CM[i, j+1]],
        [CM[i+1, i], CM[i+1, i+1], CM[i+1, j], CM[i+1, j+1]],
        [CM[j, i], CM[j, j+1], CM[j, j], CM[j, j+1]],
        [CM[j+1, i], CM[j+1, i+1], CM[j+1, j], CM[j+1, j+1]],
    ])

def takeSubonly2(CM, i, j):
    return np.array([[CM[i, j], CM[i, j+1]], [CM[i+1, j], CM[i+1, j+1]]])

def computeValue(CM):
    A = CM[0:2, 0:2]
    B = CM[2:4, 2:4]
    AB = CM[0:2, 2:4]

    delta = np.linalg.det(A) + np.linalg.det(B) - 2 * np.linalg.det(AB)
    detCM = np.linalg.det(CM)

    ptev = delta / 2 - np.sqrt(delta**2 - 4 * detCM) / 2
    result = -np.log2(ptev) / 2

    return max(result, 0)

def build_adj_numpy(edges, weights, node_index, N):
    A = np.zeros((N, N), dtype=complex)

    for (edge, w) in zip(edges, weights):
        if w != 0:
            i = node_index[edge[0]]
            j = node_index[edge[1]]
            A[i, j] += w
            A[j, i] += w  # undirected graph

    return A

def symplectic_form(n_modes):
    M = np.array([[0, 1], [-1, 0]])
    return np.kron(np.eye(n_modes), M)

def symplectic_eigenvalues(CM):
    #Omega = symplectic_form(CM.shape[0] // 2)
    Omega = np.kron(np.array([[0, 1], [-1, 0]]), np.eye(CM.shape[0] // 2))
    eigvals = np.linalg.eigvals(1j * Omega @ CM)
    eigvals = np.sort(np.abs(eigvals))
    return eigvals[::2]  # take one from each pair

def beamsplitter_hamiltonian_allpairs(n_modes):
    """
    Build the beam splitter Hamiltonian for ALL pairs of modes,
    in xp-interleaved ordering (x1, p1, x2, p2, ..., xn, pn).

    For each pair (k, m) with k < m:
        H[2k,   2m+1] = -1   (x_k  couples to p_m)
        H[2m+1, 2k  ] = -1
        H[2k+1, 2m  ] = +1   (p_k  couples to x_m)
        H[2m,   2k+1] = +1
    """
    from itertools import combinations

    dim = 2 * n_modes
    H = np.zeros((dim, dim))

    for k, m in combinations(range(n_modes), 2):
        x_k = 2 * k
        p_k = 2 * k + 1
        x_m = 2 * m
        p_m = 2 * m + 1

        H[x_k, p_m] = -1
        H[p_m, x_k] = -1
        H[p_k, x_m] =  1
        H[x_m, p_k] =  1

    return H

# ----------------------------
# Main script
# ----------------------------

K = canonicalM(16)

a = 1
r = 0.5
alpha = 0
n = (2 * 0 + 1)

#theta = np.pi * (1 - 2 * np.random.rand(3)) 
theta = np.array([0, np.pi, -np.pi])/2*0
phases = np.exp(1j * theta)
b=np.pi/4*0

# Symplectic matrix
M = np.array([[0, 1], [-1, 0]])
k4 = 1
#k3 = 2
#k2 = 4
#k1 = 8
#k3 = 0.5
#k2 = 0.25
#k1 = 0.125
#k3 = 1.25
#k2 = 1.5
#k1 = 1.75
k1=k4
k2=k4
k3=k4
couplings = np.diag([k1, k2, k3, k4, k4, k3, k2, k1])
Omega = np.kron(np.eye(8), M)
#Omega = np.kron(M, np.eye(8))
#Coupling = Omega.T
Coupling = np.kron(couplings, M).T
print(Coupling)
init = a * np.eye(16)
diffusion = (Omega @ Coupling) @ init @ (Omega @ Coupling).T

# ----------------------------
# Graph definition
# ----------------------------

edges = [
    (4, 5),  # -1, 1
    (3, 6),  # -2, 2
    (2, 7),  # -3, 3
    (1, 8),  # -4, 4
    (5, 6),  # 1, 2
    (4, 7),  # -1, 3
    (3, 8),  # -2, 4
    (4, 3),  # -1, -2
    (5, 2),  # 1, -3
    (6, 1),  # 2, -4
]

nodes = [1, 2, 3, 4, 5, 6, 7, 8]
node_index = {node: i for i, node in enumerate(nodes)}
N = len(nodes)

weights1 = np.concatenate([np.ones(4), np.zeros(6)])
weights2 = np.concatenate([np.zeros(4), np.ones(3), np.zeros(3)])
weights3 = np.concatenate([np.zeros(7), np.ones(3)])

Adj1 = build_adj_numpy(edges, weights1, node_index, N)
Adj2 = build_adj_numpy(edges, weights2, node_index, N)
Adj3 = build_adj_numpy(edges, weights3, node_index, N)
bs = np.fliplr(np.eye(16))

r1=r*phases[0]
r2=r*phases[1]*0
r3=r*phases[2]*0

Adj = r1 * Adj1 + r2 * Adj2 + r3 * Adj3
# ----------------------------
# Hamiltonian
# ----------------------------

#Add a beam splitter between what? 

H_BS = np.zeros((16, 16), dtype=complex)
# Symmetric beam splitting

"""
# Between modes 1 and 8 (0 and 7 in 0-indexing)
H_BS[0, 15] = -1
H_BS[15, 0] = -1
H_BS[1, 14] = 1
H_BS[14, 1] = 1

#Bep.pi/4*0tween modes 2 and 7 (1 and 6 in 0-indexing)
H_BS[2, 13] = -1
H_BS[13, 2] = -1
H_BS[3, 12] = 1
H_BS[12, 3] = 1

#Between modes 3 and 6 (2 and 5 in 0-indexing)
H_BS[4, 11] = -1
H_BS[11, 4] = -1    
H_BS[5, 10] = 1
H_BS[10, 5] = 1

#between modes 4 and 5 (3 and 4 in 0-indexing)
H_BS[6, 9] = -1
H_BS[9, 6] = -1
H_BS[7, 8] = 1
H_BS[8, 7] = 1

"""
"""
# Between modes 2 and 4 
H_BS[2, 7] = -1
H_BS[7, 2] = -1
H_BS[3, 6] = 1
H_BS[6, 3] = 1

# Between modes 5 and 7
H_BS[8, 13] = -1
H_BS[13, 8] = -1
H_BS[9, 12] = 1
H_BS[12, 9] = 1

# Between modes 4 and 6
H_BS[6, 11] = -1
H_BS[11, 6] = -1
H_BS[7, 10] = 1 
H_BS[10, 7] = 1

# Between modes 3 and 5
H_BS[4, 9] = -1
H_BS[9, 4] = -1
H_BS[5, 8] = 1 
H_BS[8, 5] = 1

# Between modes 1 and 3
H_BS[0, 5] = -1
H_BS[5, 0] = -1
H_BS[1, 4] = 1
H_BS[4, 1] = 1

# Between modes 6 and 8
H_BS[10, 15] = -1
H_BS[15, 10] = -1
H_BS[11, 14] = 1
H_BS[14, 11] = 1"""

# Between modes 1 and 2
H_BS[0, 3] = -1
H_BS[3, 0] = -1
H_BS[1, 2] = 1
H_BS[2, 1] = 1

# Between modes 2 and 3
H_BS[2, 5] = -1
H_BS[5, 2] = -1
H_BS[3, 4] = 1
H_BS[4, 3] = 1

# Between modes 3 and 4
H_BS[4, 7] = -1
H_BS[7, 4] = -1
H_BS[5, 6] = 1  
H_BS[6, 5] = 1

# Between modes 4 and 5
H_BS[6, 9] = -1
H_BS[9, 6] = -1
H_BS[7, 8] = 1      
H_BS[8, 7] = 1

# Between modes 5 and 6
H_BS[8, 11] = -1
H_BS[11, 8] = -1
H_BS[9, 10] = 1
H_BS[10, 9] = 1

# Between modes 6 and 7
H_BS[10, 13] = -1   
H_BS[13, 10] = -1
H_BS[11, 12] = 1
H_BS[12, 11] = 1

# Between modes 7 and 8
H_BS[12, 15] = -1
H_BS[15, 12] = -1
H_BS[13, 14] = 1
H_BS[14, 13] = 1

H_BS *= 1j*1*0

n_modes = 8
H_BS = beamsplitter_hamiltonian_allpairs(n_modes)

plt.figure()
plt.imshow(H_BS)
plt.colorbar(label="BEAM SPLITTER HAMILTONIAN")
plt.title("Beam Splitter Hamiltonian Matrix")

H = K @ (block_diag(1j * Adj, -1j * np.conj(Adj)))@ (K.T) / 2 + H_BS

Ap = Omega @ H + (Omega @ Coupling) @ Omega @ Coupling.T / 2

# ----------------------------
# Monitoring
# ----------------------------
#*(1/np.cos(b)**2 + n * np.tan(b)**2)
heterodyne = np.eye(16)

invs = sqrtm(np.linalg.inv(init + heterodyne))

B = Coupling @ Omega @ invs
E = Omega @ Coupling @ init @ invs
D = diffusion - E @ E.T

A = Ap + E @ B.T
gB = -B @ B.T

# ----------------------------
# Riccati equation
# ----------------------------

P = solve_continuous_are(A, B, D, np.eye(B.shape[1]), e=None, s=None, balanced=True)
plt.figure()
plt.imshow(np.real(P))
plt.colorbar(label="Riccati Solution P")
plt.title("Riccati Solution Matrix P")
print(P)
# ----------------------------
# Post-processing
# ----------------------------

CM3 = np.real(takeSub(P, 6, 8))
x3 = computeValue(CM3)
m3 = 1 / np.sqrt(np.linalg.det(CM3))

print("CM3 =", CM3)
print("x3 =", x3)
print("m3 =", m3)


N = 8  # number of modes
heatmap = np.zeros((N, N))

for i_idx, i in enumerate(range(0, 16, 2)):
    for j_idx, j in enumerate(range(0, 16, 2)):
        CM_ij = takeSub(P, i, j)
        heatmap[i_idx, j_idx] = computeValue(CM_ij)
        
plt.figure()
heatmap = 0.5 * (heatmap + heatmap.T)  # symmetrize for better visualization§
plt.imshow(heatmap)
plt.colorbar(label="Log-Negativity")
plt.xlabel("Mode j")
plt.ylabel("Mode i")
plt.title("Pairwise Entanglement Map")


# DIAGONALIZE

#eigvals, eigvecs = np.linalg.eig(np.real(P))

#print("Eigenvalues:", eigvals)
#print("Eigenvectors:", eigvecs)

#Covariance matrix in the eigenbasis
#P_diag = np.diag(sorted(eigvals))
# Compute symplectic eigenvalues
#nu = symplectic_eigenvalues(np.real(P))
#print("Symplectic eigenvalues:", nu)

# Build Williamson diagonal form
# Each eigenvalue appears twice
#P_diag = np.diag(np.repeat(nu, 2))

#plt.figure()
#plt.imshow(np.real(P_diag))
#plt.colorbar(label="Eigenbasis values")
#plt.title("CM in eigenbasis")

N = 8  # number of modes
heatmap = np.zeros((N, N))

for i_idx, i in enumerate(range(0, 16, 2)):
    for j_idx, j in enumerate(range(0, 16, 2)):
        CM_ij = takeSub(P, i, j)
        heatmap[i_idx, j_idx] = (np.sqrt(np.linalg.det(CM_ij))**(-1))
        
plt.figure()
plt.imshow(heatmap)
plt.colorbar(label="Purity")
plt.xlabel("Mode j")
plt.ylabel("Mode i")
plt.title("Pairwise Purity Map")
print((np.sqrt(np.linalg.det(P)))**(-1))





# Assumes you have: subcovariance(cov, modes) -> 4x4 np.ndarray
# and a covariance matrix `cov` with n modes in xp ordering.
 
n = P.shape[0] // 2
 
# Matrix to store determinants, NaN on diagonal (undefined for j==k)
det_matrix = np.full((n, n), np.nan)
 
for j in range(n):
    for k in range(n):
        if j != k:
            sigma_jk = takeSubonly2(P, 2*j, 2*k)
            d = np.linalg.det(sigma_jk)
            det_matrix[j, k] = d

fails = det_matrix <= 0
 
print("det(sigma_jk) matrix:")
print(np.array2string(det_matrix, precision=3, suppress_small=True))
print()
print("det(sigma_jk) <= 0 (False):")
print(fails)

# EXPERIMENT
print("Experimental data:")

data = scipy.io.loadmat('/home/vxmikael/cluster/cleanmats/V_clean_0.mat')
cov = data['V_clean']

n = cov.shape[0] // 2

# Matrix to store determinants, NaN on diagonal (undefined for j==k)
det_matrix = np.full((n, n), np.nan)
 
for j in range(n):
    for k in range(n):
        if j != k:
            sigma_jk = takeSubonly2(cov, 2*j, 2*k)
            d = np.linalg.det(sigma_jk)
            det_matrix[j, k] = d

fails = det_matrix <= 0
 
print("det(sigma_jk) matrix:")
print(np.array2string(det_matrix, precision=3, suppress_small=True))
print()
print("det(sigma_jk) <= 0 (False):")
print(fails)

plt.show()








"""

def build_collective_relative_basis(N):
    
    #Builds S such that:
    #r1 = sum_i x_i
    #r2 = x2 - x3
    #r3 = x3 - x4
    #...
    #TRY r1 = x1-x2
    

    S = np.zeros((N, N))

    # r1 = sum x_i
    #S[0, :] = 1.0

    # relative differences
    for i in range(1, N):
        S[i, i-1] = 1.0
        if i < N:
            S[i, i] = -1.0

    return S

def build_r_matrix(N):
    S = np.zeros((N, N))
    for i in range(N - 1):
        S[i, i] = 1
        S[i, i + 1] = -1
    S[N - 1, N - 1] = 1  # last mode
    return S

def build_q_matrix(N):
    S = np.zeros((N, N))
    for i in range(N - 1):
        S[i, i] = 1
        S[i, i + 1] = -1
    S[N - 1, N - 1] = 1
    return S

def transform_covariance(V, S):
    N = S.shape[0]
    S_full = np.block([
        [S, np.zeros((N, N))],
        [np.zeros((N, N)), S]
    ])

    return S_full @ V @ S_full.T

#S = build_collective_relative_basis(8)

S_full = np.block([
    [build_r_matrix(8), np.zeros_like(build_r_matrix(8))],
    [np.zeros_like(build_q_matrix(8)), build_q_matrix(8)]
])

def xp_transform(N):
    M = 2 * N
    S = np.zeros((M, M))

    # r_i = x_i + x_{i+1}
    for i in range(N - 1):
        S[i, 2*i]     = 1   # x_i
        S[i, 2*(i+1)] = 1   # x_{i+1}

    # last position mode
    S[N - 1, 2*(N - 1)] = 1

    # q_i = p_i - p_{i+1}
    for i in range(N - 1):
        S[N + i, 2*i + 1]     = 1   # p_i
        S[N + i, 2*(i+1) + 1] = -1  # p_{i+1}

    # last momentum mode
    S[2*N - 1, 2*(N - 1) + 1] = 1

    return S

S_full = xp_transform(8)

S_inv = np.linalg.inv(S_full)

A_tilde = S_full @ A @ S_inv
B_tilde = S_full @ B
D_tilde = S_full @ D @ S_full.T


P_tilde = solve_continuous_are(
    A_tilde,
    B_tilde,
    D_tilde,
    np.eye(B.shape[1]),
    balanced=True
)

plt.figure()
plt.imshow(np.real(P_tilde))
plt.colorbar(label="Riccati Solution P")
plt.title("Riccati Solution Matrix P")

N = 8  # number of modes
heatmap = np.zeros((N, N))

for i_idx, i in enumerate(range(0, 16, 2)):
    for j_idx, j in enumerate(range(0, 16, 2)):
        CM_ij = takeSub(P_tilde, i, j)
        heatmap[i_idx, j_idx] = computeValue(CM_ij)
heatmap = 0.5 * (heatmap + heatmap.T)  # symmetrize for better visualization
plt.figure()
plt.imshow(heatmap)
plt.colorbar(label="Log-Negativity")
plt.xlabel("Mode j")
plt.ylabel("Mode i")
plt.title("Pairwise Entanglement Map")

N = 8  # number of modes
heatmap = np.zeros((N, N))

for i_idx, i in enumerate(range(0, 16, 2)):
    for j_idx, j in enumerate(range(0, 16, 2)):
        CM_ij = takeSub(P_tilde, i, j)
        heatmap[i_idx, j_idx] = (np.sqrt(np.linalg.det(CM_ij))**(-1))
        
heatmap = 0.5 * (heatmap + heatmap.T)  # symmetrize for better visualization
plt.figure()
plt.imshow(heatmap)
plt.colorbar(label="Purity")
plt.xlabel("Mode j")
plt.ylabel("Mode i")
plt.title("Pairwise Purity Map")
print((np.sqrt(np.linalg.det(P_tilde)))**(-1))
plt.show()
"""
