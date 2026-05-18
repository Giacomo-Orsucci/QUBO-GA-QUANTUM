import os
import numpy as np

def load_hamburg_matrix(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        i_indices, j_indices, weights = data['i'], data['j'], data['Jij']
        n_nodes = int(max(np.max(i_indices), np.max(j_indices))) + 1
        Q = np.zeros((n_nodes, n_nodes))
        for r, c, w in zip(i_indices, j_indices, weights):
            r, c = int(r), int(c)
            Q[r, c] = w
            if r != c: Q[c, r] = w
        return Q

def main():
    # Il percorso del file originale
    file_path = "./qubo-bench/qubo-benchmark-main/generate/compsup/instances/2d_(4, 4)_precision256/seed00.npz"
    
    # La bitstring dominante restituita dalla QPU (Falso Minimo Locale)
    # Assicurati di copiare quella esatta dal tuo grafico a barre!
    qpu_bitstring = "0100001001000000" 
    
    # Il Ground State matematico perfetto (da paper/benchmark)
    optimal_energy = -16.41
    
    try:
        Q = load_hamburg_matrix(file_path)
    except FileNotFoundError:
        print(f"Errore: File non trovato in {file_path}")
        return

    # Trasforma la stringa in un vettore di 0 e 1
    x = np.array([int(b) for b in qpu_bitstring])
    
    # Calcolo dell'energia QUBO: E = x^T * Q * x
    qpu_energy = x.T @ Q @ x
    
    # Calcolo dell'Approximation Ratio
    # Più è vicino a 1.0 (o 100%), migliore è l'approssimazione
    approx_ratio = qpu_energy / optimal_energy
    
    print("\n" + "="*50)
    print(" ANALISI ENERGETICA DELLA SOLUZIONE QUANTISTICA")
    print("="*50)
    print(f"Bitstring analizzata : {qpu_bitstring}")
    print(f"Nodi attivi (Atomi)  : {np.sum(x)} su {len(x)}")
    print("-" * 50)
    print(f"Energia Ottima Assoluta : {optimal_energy:.4f}")
    print(f"Energia Trovata da QPU  : {qpu_energy:.4f}")
    print("-" * 50)
    print(f"Approximation Ratio     : {approx_ratio:.2%}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()