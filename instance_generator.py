import numpy as np
import os
from scipy.spatial.distance import pdist, squareform


#TO DO: better commenting.

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


import numpy as np
import os
from scipy.spatial.distance import pdist, squareform

def generate_verisimilar_jade_qubo(n=5, target_degree=2.5, min_dist=4.0, rydberg_radius=12.0, max_hw_radius=50.0, seed=42):
    """
    Genera matrici QUBO UDG realistiche per l'hardware Jade.
    Adatta l'area di posizionamento in base a 'n' per garantire che il grafo 
    abbia interazioni reali (matrice QUBO popolata), rispettando i vincoli fisici.
    """
    np.random.seed(seed)
    
    # 1. Calcolo dell'Area Operativa Dinamica
    # In un Random Geometric Graph, per avere un grado medio (target_degree),
    # il raggio di posizionamento deve scalare con la radice di N.
    # Formula derivata da: <k> = N * (Area_Rydberg / Area_Generazione)
    effective_radius = rydberg_radius * np.sqrt(n / target_degree)
    
    # Assicuriamoci che l'area non superi MAI i limiti fisici della macchina (50 µm)
    # e che non sia troppo piccola per farci stare gli atomi a distanza min_dist
    min_required_radius = np.sqrt(n) * (min_dist / 2.0) * 1.5 
    effective_radius = np.clip(effective_radius, min_required_radius, max_hw_radius)
    
    print(f"Generazione {n}x{n} | Raggio Operativo calcolato: {effective_radius:.2f} µm (Max Hardware: {max_hw_radius} µm)")
    
    # 2. Generazione Coordinate con Rejection Sampling
    positions = []
    attempts = 0
    max_attempts = 20000
    
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        
        # Posizioniamo gli atomi all'interno dell'Area Operativa (non di tutto l'hardware)
        # Usiamo coordinate polari per distribuirli uniformemente in un cerchio
        r = effective_radius * np.sqrt(np.random.rand())
        theta = np.random.rand() * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Controllo Distanza Minima Hardware (Jade: 4 µm)
        if len(positions) > 0:
            pos_array = np.array(positions)
            distances_to_others = np.linalg.norm(pos_array - np.array([x, y]), axis=1)
            if np.any(distances_to_others < min_dist):
                continue # Troppo vicino, scarta l'atomo
                
        positions.append([x, y])
        
    if len(positions) < n:
        raise ValueError(f"Impossibile posizionare {n} atomi! Prova ad abbassare il target_degree o min_dist.")
        
    positions = np.array(positions)
    
    # 3. Costruzione Matrice QUBO
    dist_matrix = squareform(pdist(positions))
    Q = np.zeros((n, n))
    
    # Diagonale: -1.0 (Voglia globale di accendersi)
    np.fill_diagonal(Q, -1.0)
    
    # Archi (Penalità): 2.0 se la distanza è inferiore al Raggio di Rydberg
    adjacency_mask = (dist_matrix < rydberg_radius) & (dist_matrix > 0)
    Q[adjacency_mask] = 2.0
    
    # 4. Salvataggio Compatibile con il tuo 'load_hamburg_matrix'
    output_dir = "./my_QUBO_instances/scaling_tests/jade_udg"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/jade_udg_{n}x{n}_R{int(rydberg_radius)}_s{seed}.npz"
    
    # Estrazione Upper Triangle
    i_idx, j_idx = np.where(np.triu(Q) != 0)
    weights = Q[i_idx, j_idx]
    
    np.savez(file_name, i=i_idx, j=j_idx, Jij=weights, positions=positions)
    
    # Statistiche di validazione
    num_edges = np.sum(adjacency_mask) / 2
    density = num_edges / (n * (n - 1) / 2) if n > 1 else 0
    actual_degree = (num_edges * 2) / n
    print(f"-> [OK] Salvato. Archi totali: {int(num_edges)} | Grado medio: {actual_degree:.1f} | Densità: {density:.2f}\n")

# --- TEST DI SCALING ---
if __name__ == "__main__":
    for size in range(30, 31):
        # target_degree=3.0 garantisce che ogni nodo, in media, faccia a cazzotti 
        # (violi il blocco di Rydberg) con altri 3 nodi. 
        # È una difficoltà perfetta per testare il tuo embedding!
        generate_verisimilar_jade_qubo(
            n=size, 
            target_degree=3.0, 
            min_dist=4.0, 
            rydberg_radius=12.0, 
            seed=42 + size
        )