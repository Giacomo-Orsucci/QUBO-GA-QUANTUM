import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from os import getenv
from qat.qlmaas.connection import QLMaaSConnection

#This is the script to retrieve asynchronously the remote result indicating n_atoms, job_id and
#benchmark file path

# --- 0. SETUP INIZIALE ---
N_ATOMS = 16
JOB_ID = "SJob208583"
FILE_PATH = "./qubo-bench/qubo-benchmark-main/generate/compsup/instances/2d_(4, 4)_precision256/seed00.npz"

# --- 1. FUNZIONI DI SUPPORTO (LOADER E SOLUTORE CLASSICO) ---
def load_hamburg_matrix(file_path):
    """Legge la matrice QUBO dal file .npz del benchmark."""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            i_indices = data['i']
            j_indices = data['j']
            weights = data['Jij']
            
            n_nodes = int(max(np.max(i_indices), np.max(j_indices))) + 1
            Q = np.zeros((n_nodes, n_nodes))
            
            for r, c, w in zip(i_indices, j_indices, weights):
                r, c = int(r), int(c)
                Q[r, c] = w
                if r != c:
                    Q[c, r] = w
            return Q
    except Exception as e:
        print(f"Errore nel caricamento della matrice: {e}")
        return None

def find_classical_ground_state(Q):
    """Esegue un calcolo Brute Force classico per trovare la soluzione ottima esatta."""
    "Per le dimensioni finora trattate possiamo usare il brute force"
    print("  -> Calcolo del Vero Ground State Matematico (Brute Force)...")
    min_energy = float('inf')
    ground_states = []
    
    for bits in product([0, 1], repeat=len(Q)):
        x = np.array(bits)
        energy = x.T @ Q @ x
        if energy < min_energy:
            min_energy = energy
            ground_states = ["".join(map(str, bits))]
        elif energy == min_energy:
            ground_states.append("".join(map(str, bits)))
            
    return ground_states, min_energy

# --- 2. ELABORAZIONE CLASSICA ---
Q = load_hamburg_matrix(FILE_PATH)
if Q is None:
    exit()

veri_ground_states, vera_energia = find_classical_ground_state(Q)
print(f"  -> [OK] Trovate {len(veri_ground_states)} soluzioni ottime matematiche con energia: {vera_energia:.4f}")

# --- 3. RECUPERO JOB QUANTISTICO ---
print(f"Connessione a Jülich in corso per cercare il Job: {JOB_ID}")
try:
    connection = QLMaaSConnection()
    if JOB_ID == "":
        raise RuntimeError("specify the environment variable JOB_ID")

    status = connection.get_status(JOB_ID)
    results = connection.get_result(JOB_ID)
    print(f"  -> Stato attuale del Job: {status}")
    
except Exception as e:
    print(f"\nErrore durante il recupero. Assicurati che l'ID sia corretto. Dettagli: {e}")
    exit()

if results is None:
    raise Exception("Nessun risultato ottenuto dall'emulatore.")
    
# Estrazione probabilità
samples = {}
for sample in results.raw_data:
    bitstring = sample.state.bitstring.zfill(N_ATOMS)
    samples[bitstring] = sample.probability

samples = dict(sorted(samples.items(), key=lambda item: item[1], reverse=True))

# --- 4. PLOT IBRIDO (QPU + MATH) ---
print("  -> Generazione Plot dell'Istogramma Quantistico...")

# Prendiamo solo i top 30
top_samples = dict(list(samples.items())[:30]) 

# Logica di colorazione: Verde se è il Ground State vero, Blu altrimenti
bar_colors = ['green' if bit in veri_ground_states else 'blue' for bit in top_samples.keys()]

plt.figure(figsize=(14, 7))
bars = plt.bar(top_samples.keys(), top_samples.values(), color=bar_colors, alpha=0.7)

# Legenda personalizzata
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.7, label='Misurazione QPU (Falsi Minimi Locali)'),
    Patch(facecolor='green', alpha=0.7, label=f'Vero Ground State Matematico ({vera_energia:.2f})')
]
plt.legend(handles=legend_elements)

plt.xlabel("Configurazioni Misurate (Bitstrings)")
plt.ylabel("Probabilità (Frequenza QPU)")
plt.title(f"Top 30 Soluzioni Emulate vs Soluzione Ottima - 2d_4x4 (16 atomi)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Controllo finale da console
print("\n--- ANALISI SCONTRINO ---")
for gs in veri_ground_states:
    if gs in top_samples:
        print(f"Vero Ground State {gs} trovato nella Top 30 al rank {list(top_samples.keys()).index(gs) + 1} con probabilità {top_samples[gs]:.4f}")
    elif gs in samples:
        print(f"Vero Ground State {gs} trovato fuori dalla Top 30 (Probabilità: {samples[gs]:.6f})")
    else:
        print(f"Vero Ground State {gs} MAI campionato dalla QPU (Probabilità 0.0)")

plt.show()

print("\nMetadati del Job:")
print(results.meta_data)