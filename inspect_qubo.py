import numpy as np
import os

#File to visualize QUBO matrices from imported dataset. As we can see from the filepath below 
#is necessary to import the desired dataset (in our case QUBO benchmark from university of Hamburgh: https://github.com/ml-uhh/qubo-benchmark/tree/main).

# Percorso della matrice QUBO 16x16 2d_4x4 con seed00
file_path = "./qubo-bench/qubo-benchmark-main/generate/compsup/instances/2d_(4, 4)_precision256/seed00.npz"
file_path = "./qubo-bench/qubo-benchmark-main/instances/compsup/2d/2d_(4, 4)_precision256_seed18.npz"

def inspect_and_print_matrix(path):
    if not os.path.exists(path):
        print(f"Errore: Il file {path} non esiste.")
        return

    try:
        with np.load(path, allow_pickle=True) as data:
            i_idx = data['i']
            j_idx = data['j']
            weights = data['Jij']
            
            n_nodes = int(max(np.max(i_idx), np.max(j_idx))) + 1
            Q = np.zeros((n_nodes, n_nodes))
            
            for r, c, w in zip(i_idx, j_idx, weights):
                r, c = int(r), int(c)
                Q[r, c] = w
                if r != c:
                    Q[c, r] = w
            
            print(f"=== ISPEZIONE VALORI MATRICE: {os.path.basename(path)} ===")
            print(f"Dimensione: {n_nodes}x{n_nodes}\n")

            # --- Stampa della Matrice Numerica ---
            # Header con i numeri di colonna
            header = "     " + "".join([f"{c:6}" for c in range(n_nodes)])
            print(header)
            print("-" * len(header))

            for r in range(n_nodes):
                # Numero di riga
                line = f"{r:2} | "
                for c in range(n_nodes):
                    val = Q[r, c]
                    if val == 0:
                        # Grigio per gli zeri
                        line += f"\033[90m{val:6.2f}\033[0m"
                    elif val > 0:
                        # Rosso per i valori positivi (repulsione/penalità)
                        line += f"\033[91m{val:6.2f}\033[0m"
                    else:
                        # Verde per i valori negativi (attrazione/premio)
                        line += f"\033[92m{val:6.2f}\033[0m"
                print(line)

            # --- Riepilogo Statistico ---
            off_diag = Q[np.triu_indices(n_nodes, k=1)]
            off_diag = off_diag[off_diag != 0]
            
            print(f"\n--- Statistiche ---")
            print(f"Valore quadratico minimo: {np.min(off_diag):.4f}")
            print(f"Valore quadratico massimo: {np.max(off_diag):.4f}")
            print(f"Deviazione Standard: {np.std(off_diag):.4f}")

            return Q

    except Exception as e:
        print(f"Errore durante la lettura: {e}")

# Esecuzione
Q_matrix = inspect_and_print_matrix(file_path)