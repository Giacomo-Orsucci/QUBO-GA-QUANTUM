import os
import time
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
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
    target_device = dataclasses.replace(AnalogDevice, name="FakeJade", max_radial_distance=50)

def load_hamburg_matrix(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        i, j, w = data['i'], data['j'], data['Jij']
        n = int(max(np.max(i), np.max(j))) + 1
        Q = np.zeros((n, n))
        for r, c, weight in zip(i, j, w):
            r, c = int(r), int(c)
            Q[r, c] = weight
            if r != c: Q[c, r] = weight
        return Q

# ==========================================
# FASE 1: GRID PARTITIONING STATICO
# ==========================================
def static_grid_partitioning(Q, r_blockade=8.5):
    N_ATOMS = len(Q)
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)
    
    G = nx.from_numpy_array(np.abs(Q_off_diag))
    spring_pos = nx.spring_layout(G, scale=20.0, seed=42) 
    base_coords = np.array([spring_pos[i] for i in range(N_ATOMS)])
    
    # Taglio statico (Algoritmo 1)
    box_size = r_blockade * 2.1 
    min_x, min_y = np.min(base_coords, axis=0)
    
    cells = {}
    cut_penalty = 0.0
    cell_ids = []
    
    for i in range(N_ATOMS):
        x, y = base_coords[i]
        cx = int(np.floor((x - min_x) / box_size))
        cy = int(np.floor((y - min_y) / box_size))
        cell_ids.append((cx, cy))
        if (cx, cy) not in cells: cells[(cx, cy)] = []
        cells[(cx, cy)].append(i)
        
    for i in range(N_ATOMS):
        for j in range(i+1, N_ATOMS):
            if Q_off_diag[i, j] != 0 and cell_ids[i] != cell_ids[j]:
                cut_penalty += abs(Q_off_diag[i, j])
                
    print(f"\n[FASE 1] Grid Partitioning Statico completato.")
    print(f"  -> Trovati {len(cells)} sotto-grafi (Celle).")
    print(f"  -> Peso dei legami inter-cella spezzati: {cut_penalty:.2f}")
    
    return cells, base_coords

# ==========================================
# FASE 2: INVIO SINGOLO PER OGNI SOTTOGRAFO
# ==========================================
def submit_individual_subgraphs(Q, cells, base_coords, qpu_emulator):
    device = target_device
    MIN_DIST = device.min_atom_distance
    
    jobs_dict = {}
    print("\n[FASE 2] Risoluzione AHS separata per ogni Sottografo...")
    
    for c_id, nodes in cells.items():
        print(f"  -> Preparazione cella {c_id} ({len(nodes)} atomi)...")
        
        # 1. Estrazione coordinate e scalatura locale (per evitare il DistanceError)
        local_coords = np.array([base_coords[n] for n in nodes])
        
        if len(nodes) > 1:
            min_d = np.min(pdist(local_coords))
            if min_d < MIN_DIST:
                scale_up = (MIN_DIST + 1.0) / min_d 
                local_center = np.mean(local_coords, axis=0)
                local_coords = (local_coords - local_center) * scale_up
                
        # Centriamo il registro locale
        local_coords[:, 0] -= np.mean(local_coords[:, 0])
        local_coords[:, 1] -= np.mean(local_coords[:, 1])
        
        # 2. Estrazione del sub-QUBO per calcolare l'energia corretta
        sub_Q = Q[np.ix_(nodes, nodes)]
        sub_Q_off_diag = sub_Q.copy()
        np.fill_diagonal(sub_Q_off_diag, 0)
        
        V_max = device.interaction_coeff / (MIN_DIST**6)
        Q_max = np.max(np.abs(sub_Q_off_diag))
        scale_factor = V_max / Q_max if Q_max > 0 else 1.0
        
        Q_target = sub_Q_off_diag * scale_factor
        scaled_delta = np.mean(np.abs(np.diag(sub_Q))) * scale_factor
        
        # Creiamo il registro SOLO con gli atomi di questa cella.
        # Rinominiamo i qubit con il loro vero indice globale per riconoscerli dopo
        qubits = {f"q{nodes[i]}": local_coords[i].tolist() for i in range(len(nodes))}
        
        # Eccezione Pulser: Se c'è un solo atomo nella cella, non possiamo creare Sequence.
        # Lo gestiamo a livello logico (un nodo singolo isolato, se ha peso negativo, fa parte del MWIS).
        if len(nodes) < 2:
            print(f"    [!] Cella con un solo atomo. Soluzione logica banale, salto invio QPU.")
            jobs_dict[c_id] = "ISOLATED_NODE"
            continue
            
        reg = Register(qubits)
        
        ideal_omega = np.median(Q_target[Q_target > 0]) if np.any(Q_target > 0) else 1.0
        Omega = min(ideal_omega, device.channels["rydberg_global"].max_amp / 1.2)

        T = 4 * 1000 
        adiabatic_pulse = Pulse(InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
                                InterpolatedWaveform(T, [-scaled_delta, 0, scaled_delta]), 0)

        seq = Sequence(reg, device)
        seq.declare_channel("ising", "rydberg_global")
        seq.add(adiabatic_pulse, "ising")

        job = IsingAQPU.convert_sequence_to_job(seq, nbshots=0)
        
        # 3. Invio asincrono
        for attempt in range(3):
            try:
                async_job = qpu_emulator.submit(job)
                job_id = async_job.batch_id if hasattr(async_job, 'batch_id') else str(async_job)
                print(f"    -> [SUCCESSO] Inviato a Jülich! Job ID: {job_id}")
                jobs_dict[c_id] = job_id
                break
            except Exception as e:
                print(f"    [ATTESA] Errore di rete ({e}). Ritento...")
                time.sleep(5)
                
    return jobs_dict

def main():
    try:
        from qlmaas.qpus import AnalogQPU
        qpu_emulator = AnalogQPU()
    except ImportError:
        from qat.qlmaas.qpus import QLMaaSQPU
        qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
        
    file_path = "./qubo-bench/qubo-benchmark-main/generate/compsup/instances/2d_(4, 4)_precision256/seed00.npz"
    Q = load_hamburg_matrix(file_path)
    
    cells, base_coords = static_grid_partitioning(Q)
    jobs_dict = submit_individual_subgraphs(Q, cells, base_coords, qpu_emulator)
    
    print("\n[RIEPILOGO JOB INVIATI]")
    for c_id, j_id in jobs_dict.items():
        print(f"Cella {c_id}: {j_id}")

if __name__ == "__main__":
    main()