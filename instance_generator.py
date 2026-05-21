import numpy as np
import os


#In realtà ho notato che la densità risultante è soggetta a varianza non trascurabile. Da valutare se è un pro o un contro
def generate_qubo(n=5, density=0.25, seed=42):
    # Fissiamo il seed per la riproducibilità scientifica
    np.random.seed(seed)
    
    Q = np.zeros((n, n))
    # 1. Diagonale costante
    np.fill_diagonal(Q, -1.0)
    
    # 2. Legami positivi casuali
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < density:
                val = np.random.uniform(0.5, 1.5)
                Q[i, j] = val
                Q[j, i] = val
                
    # Creazione cartella di output se non esiste
    os.makedirs("./my_QUBO_instances/scaling_tests/friendly", exist_ok=True)
    
    # Salvataggio dinamico
    file_name = f"./my_QUBO_instances/scaling_tests/global_friendly_{n}x{n}_d{int(density*100)}_s{seed}.npz"
    
    i_idx, j_idx = np.where(np.triu(Q) != 0)
    weights = Q[i_idx, j_idx]
    np.savez(file_name, i=i_idx, j=j_idx, Jij=weights)
    print(f"[OK] Generata istanza: {file_name}")

# Esempio di generazione in batch
for size in range(5, 11): # Genera da 5x5 a 10x10
    generate_qubo(n=size, density=0.30, seed=100)