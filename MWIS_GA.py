import os
import time
import numpy as np
import pandas as pd
import pygad
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import dataclasses

from pulser import InterpolatedWaveform, Pulse, Sequence, Register
from pulser_myqlm import IsingAQPU

# ==========================================
# SETUP HARDWARE
# ==========================================
try:
    from pulser.devices import Jade as target_device
except ImportError:
    from pulser.devices import AnalogDevice
    target_device = dataclasses.replace(AnalogDevice, name="FakeJade")

# ==========================================
# PARSER DEI .NPZ 
# ==========================================
def load_hamburg_matrix(file_path):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            i, j, w = data['i'], data['j'], data['Jij']
            n_nodes = int(max(np.max(i), np.max(j))) + 1
            Q = np.zeros((n_nodes, n_nodes))
            for r, c, weight in zip(i, j, w):
                r, c = int(r), int(c)
                Q[r, c] = weight
                if r != c: Q[c, r] = weight
            return Q
    except Exception as e:
        print(f"  [ERRORE LETTURA] {file_path}: {e}")
        return None

# ==========================================
# FASE 2 HELPER: MICRO-GA LOCALE PER UDG
# ==========================================
def run_local_udg_ga(sub_Q, nodes, min_dist, r_blockade):
    n_local = len(nodes)
    
    # Solo i legami > 0 (conflitti) devono subire il Blockade
    Target_A = (sub_Q > 0).astype(int) 
    
    def fitness_func(ga_inst, sol, sol_idx):
        coords = np.array(sol, dtype=np.float64).reshape((n_local, 2))
        dist_matrix = squareform(pdist(coords))
        error = 0.0
        
        for i in range(n_local):
            for j in range(i+1, n_local):
                d = dist_matrix[i, j]
                if d < min_dist:
                    error += (min_dist - d) * 10000.0
                
                if Target_A[i, j] == 1: 
                    if d > r_blockade - 0.2: error += (d - (r_blockade - 0.2)) * 10.0
                else: 
                    if d <= r_blockade + 0.2: error += ((r_blockade + 0.2) - d) * 10.0
                    
        return 1.0 / (error + 1e-6)
        
    ga = pygad.GA(num_generations=250, num_parents_mating=10, fitness_func=fitness_func,
                  sol_per_pop=50, num_genes=n_local * 2, gene_type=float,
                  gene_space={'low': -10.0, 'high': 10.0}, suppress_warnings=True)
    ga.run()
    best_sol = ga.best_solution()[0]
    return np.array(best_sol, dtype=np.float64).reshape((n_local, 2))

# ==========================================
# PIPELINE GERARCHICA CLASSICA (Compilazione)
# ==========================================
def compile_problem(Q):
    N_ATOMS = len(Q)
    device = target_device
    MIN_DIST = device.min_atom_distance
    R_BLOCKADE = 8.5
    BOX_SIZE = R_BLOCKADE * 2.0
    
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)
    
    # --- FASE 1: GRID OFFSET GA ---
    print("\n  [FASE 1] Ottimizzazione Griglia: Minimizzazione tagli logici...")
    G = nx.from_numpy_array(np.abs(Q_off_diag))
    spring_pos = nx.spring_layout(G, scale=20.0)
    base_coords = np.array([spring_pos[i] for i in range(N_ATOMS)])
    
    def grid_fitness(ga_instance, solution, solution_idx):
        dx, dy = solution
        cut_penalty = 0.0
        cell_ids = [(int(np.floor((base_coords[i][0] - dx) / BOX_SIZE)), 
                     int(np.floor((base_coords[i][1] - dy) / BOX_SIZE))) for i in range(N_ATOMS)]
            
        for i in range(N_ATOMS):
            for j in range(i+1, N_ATOMS):
                if Q_off_diag[i, j] != 0 and cell_ids[i] != cell_ids[j]:
                    cut_penalty += abs(Q_off_diag[i, j])
        return 1.0 / (cut_penalty + 1e-6)

    ga_grid = pygad.GA(num_generations=150, num_parents_mating=10, fitness_func=grid_fitness,
                       sol_per_pop=50, num_genes=2, gene_type=float,
                       gene_space=[{'low': -BOX_SIZE, 'high': BOX_SIZE}, {'low': -BOX_SIZE, 'high': BOX_SIZE}],
                       suppress_warnings=True)
    ga_grid.run()
    dx, dy = ga_grid.best_solution()[0]
    cut_error = (1.0 / ga_grid.best_solution()[1]) - 1e-6
    
    cells = {}
    for i in range(N_ATOMS):
        c_id = (int(np.floor((base_coords[i][0] - dx) / BOX_SIZE)), int(np.floor((base_coords[i][1] - dy) / BOX_SIZE)))
        if c_id not in cells: cells[c_id] = []
        cells[c_id].append(i)
        
    print(f"    -> Trovati {len(cells)} sotto-grafi. Errore residuo (tagli): {cut_error:.2f}")

    # --- FASE 2: LOCAL UDG GAs ---
    print("  [FASE 2] Compilazione Spaziale Locale per ogni cella...")
    local_layouts = {}
    for c_id, nodes in cells.items():
        if len(nodes) == 1:
            local_layouts[c_id] = {nodes[0]: np.array([0.0, 0.0])}
        else:
            sub_Q = Q_off_diag[np.ix_(nodes, nodes)]
            loc_coords = run_local_udg_ga(sub_Q, nodes, MIN_DIST, R_BLOCKADE)
            loc_coords -= np.mean(loc_coords, axis=0)
            local_layouts[c_id] = {nodes[i]: loc_coords[i] for i in range(len(nodes))}

    # --- FASE 4: GLOBAL MERGE GA (Previsione Classica) ---
    print("  [FASE 4] Simulazione Merge Classico (Valutazione Energia Totale)...")
    def merge_fit(ga_inst, sol, idx):
        x = np.array(sol)
        energy = x.T @ Q @ x
        return -energy 
        
    ga_merge = pygad.GA(num_generations=100, num_parents_mating=10, fitness_func=merge_fit,
                        sol_per_pop=50, num_genes=N_ATOMS, gene_type=int, gene_space=[0, 1], suppress_warnings=True)
    ga_merge.run()
    best_sol = ga_merge.best_solution()[0]
    best_energy = -ga_merge.best_solution()[1]
    
    print(f"    -> Stima GA di Ricucitura: Energia {best_energy:.4f} ({sum(best_sol)} atomi attivi)")
    return cells, local_layouts, cut_error

# ==========================================
# FASE 3: ESECUZIONE QUANTISTICA MULTIPLA
# ==========================================
def submit_quantum_subgraphs(Q, cells, local_layouts, qpu_emulator):
    print("  [FASE 3] Invio pacchetti quantistici separati (1 Job per Cella)...")
    device = target_device
    MIN_DIST = device.min_atom_distance
    
    job_ids = {}
    
    for c_id, layout in local_layouts.items():
        nodes = list(layout.keys())
        coords = list(layout.values())
        
        if len(nodes) < 2:
            print(f"    -> Cella {c_id}: Nodo singolo isolato (salto QPU)")
            job_ids[c_id] = ("ISOLATED_NODE", len(nodes))
            continue
            
        sub_Q = Q[np.ix_(nodes, nodes)]
        sub_Q_off_diag = sub_Q.copy()
        np.fill_diagonal(sub_Q_off_diag, 0)
        
        V_max = device.interaction_coeff / (MIN_DIST**6)
        Q_max = np.max(np.abs(sub_Q_off_diag)) if np.any(sub_Q_off_diag) else 1.0
        scale_factor = V_max / Q_max
        
        Q_target = sub_Q_off_diag * scale_factor
        
        qubits = {f"q{nodes[i]}": coords[i].tolist() for i in range(len(nodes))}
        reg = Register(qubits)
        
        ideal_omega = np.median(Q_target[Q_target > 0]) if np.any(Q_target > 0) else 1.0
        Omega = min(ideal_omega, device.channels["rydberg_global"].max_amp / 1.2)
        
        # FIX ADIABATICO: forza necessaria per accendere gli atomi
        scaled_delta = Omega * 2.5

        adiabatic_pulse = Pulse(
            InterpolatedWaveform(4000, [1e-9, Omega, 1e-9]),
            InterpolatedWaveform(4000, [-scaled_delta, 0, scaled_delta]), 0
        )

        seq = Sequence(reg, device)
        seq.declare_channel("ising", "rydberg_global")
        seq.add(adiabatic_pulse, "ising")
        job = IsingAQPU.convert_sequence_to_job(seq, nbshots=0)

        for attempt in range(3):
            try:
                async_job = qpu_emulator.submit(job)
                j_id = async_job.batch_id if hasattr(async_job, 'batch_id') else str(async_job)
                print(f"    -> [SUCCESSO] Cella {c_id} (Nodi: {len(nodes)}) inviata! ID: {j_id}")
                # Salviamo anche quanti nodi ci sono, per il padding durante l'estrazione
                job_ids[c_id] = (j_id, len(nodes))
                break
            except Exception as e:
                time.sleep(5)
                if attempt == 2: job_ids[c_id] = ("ERRORE_INVIO", len(nodes))
                
    return job_ids

if __name__ == "__main__":
    print("Inizializzazione dell'emulatore remoto AnalogQPU...")
    try:
        from qlmaas.qpus import AnalogQPU
        qpu_emulator = AnalogQPU()
    except ImportError:
        try:
            from qat.qlmaas.qpus import QLMaaSQPU
            qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
        except ImportError:
            # Fallback locale se non c'è connessione ai server Atos
            print("  [!] Server remoto non trovato, uso emulatore locale.")
            qpu_emulator = IsingAQPU()

    # Specifica il file che hai appena generato
    FILE_PATH = "global_friendly_16.npz"
    print(f"\nAnalisi di: {FILE_PATH}")
    
    Q = load_hamburg_matrix(FILE_PATH)
    if Q is not None:
        cells, local_layouts, cut_error = compile_problem(Q)
        job_ids = submit_quantum_subgraphs(Q, cells, local_layouts, qpu_emulator)
        
        print("\n--- Riepilogo per il Fetch Script ---")
        for c_id, data in job_ids.items():
            if data[0] != "ISOLATED_NODE":
                print(f"ID: {data[0]} | Qubit attesi: {data[1]}")