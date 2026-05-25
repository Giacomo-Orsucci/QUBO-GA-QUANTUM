import numpy as np
import os

# Percorso della matrice QUBO 16x16 2d_4x4 con seed00
#file_path = "./qubo-bench/qubo-benchmark-main/instances/compsup/2d/2d_(4, 4)_precision256_seed18.npz"
#file_path = "./my_QUBO_instances/scaling_tests/friendly/global_friendly_6x6_d46.7_s100.npz"
#file_path = "./my_QUBO_instances/tutorial_5x5.npz"
file_path = "./my_QUBO_instances/scaling_tests/jade_udg/jade_udg_7x7_R12_s49.npz"

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
            header = "     " + "".join([f"{c:6}" for c in range(n_nodes)])
            print(header)
            print("-" * len(header))

            for r in range(n_nodes):
                line = f"{r:2} | "
                for c in range(n_nodes):
                    val = Q[r, c]
                    if val == 0:
                        line += f"\033[90m{val:6.2f}\033[0m"
                    elif val > 0:
                        line += f"\033[91m{val:6.2f}\033[0m"
                    else:
                        line += f"\033[92m{val:6.2f}\033[0m"
                print(line)

            # --- Calcolo Statistiche Avanzate ---
            # Valori fuori diagonale (termini quadratici)
            off_diag = Q[np.triu_indices(n_nodes, k=1)]
            actual_edges = off_diag[off_diag != 0]
            
            # Valori sulla diagonale (termini lineari)
            diag_vals = np.diag(Q)
            
            # 1. Densità
            n_possible_edges = (n_nodes * (n_nodes - 1)) / 2
            n_actual_edges = len(actual_edges)
            density = n_actual_edges / n_possible_edges if n_possible_edges > 0 else 0
            
            # 2. Gradi dei nodi (quanti archi non nulli ha ogni nodo, escludendo sé stesso)
            Q_off_diag_only = Q.copy()
            np.fill_diagonal(Q_off_diag_only, 0)
            node_degrees = np.sum(Q_off_diag_only != 0, axis=1)
            
            # 3. Dynamic Range
            abs_edges = np.abs(actual_edges)
            dynamic_range = np.max(abs_edges) / np.min(abs_edges) if (len(abs_edges) > 0 and np.min(abs_edges) > 0) else 0

            print(f"\n--- Statistiche per Embedding ---")
            print(f"Densità del Grafo:      {density * 100:.1f}% ({n_actual_edges}/{int(n_possible_edges)} archi)")
            print(f"Grado Nodi (Min/Max):   {np.min(node_degrees)} / {np.max(node_degrees)} (Media: {np.mean(node_degrees):.1f})")
            print(f"Range Pesi Quadratici:  Da {np.min(actual_edges):.4f} a {np.max(actual_edges):.4f}")
            print(f"Dynamic Range (Max/Min): {dynamic_range:.1f}x")
            print(f"Deviazione Std (Archi): {np.std(actual_edges):.4f}")
            print(f"Media Diagonale (Lin):  {np.mean(np.abs(diag_vals)):.4f}")

            return Q

    except Exception as e:
        print(f"Errore durante la lettura: {e}")

# Esecuzione
Q_matrix = inspect_and_print_matrix(file_path)