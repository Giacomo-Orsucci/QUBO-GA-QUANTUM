import numpy as np

def generate_global_friendly_qubo(n=16, density=0.25):
    Q = np.zeros((n, n))
    # 1. Diagonale costante (Tutti i nodi vogliono accendersi)
    np.fill_diagonal(Q, -1.0)
    
    # 2. Legami positivi casuali (Conflitti)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < density:
                val = np.random.uniform(0.5, 1.5)
                Q[i, j] = val
                Q[j, i] = val
                
    # Salvataggio in formato Amburgo
    i_idx, j_idx = np.where(np.triu(Q) != 0)
    weights = Q[i_idx, j_idx]
    np.savez("global_friendly_16.npz", i=i_idx, j=j_idx, Jij=weights)
    print("File 'global_friendly_16.npz' generato con successo.")

generate_global_friendly_qubo()