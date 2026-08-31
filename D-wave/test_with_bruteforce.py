# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite


#Little script to test a little D-Wave platforms with bruteforce classical bench (so for little instances).


# ==========================================
# 0. CONFIGURATION
# ==========================================
#FILE_PATH = "../my_QUBO_instances/tutorial_5x5.npz"
FILE_PATH = "../my_QUBO_instances/scaling_tests/jade_udg/jade_udg_25x25_R12_s67.npz"
OUTPUT_CSV = "./new_csv/dwave_experiment_registry_udg_25x25.csv"

NUM_READS = 1000  #suggestion: 250 for debugging and 1000 or above to benhcmark
USE_HYBRID = False

# ==========================================
# 1. CLASSICAL PARSERS & SOLVERS
# ==========================================
def load_hamburg_matrix_dense(file_path):
    """Loads the dense Q matrix for exact classical calculation."""
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
        print(f"  [READ ERROR] Unable to load dense matrix {file_path}: {e}")
        return None

def load_hamburg_matrix_for_dwave_aligned(file_path):
    """Loads the sparse Q dictionary optimized and aligned for D-Wave."""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            i_indices = data['i']
            j_indices = data['j']
            weights = data['Jij']
            
            Q_dict = {}
            for r, c, w in zip(i_indices, j_indices, weights):
                r, c = int(r), int(c)
                
                if r == c:
                    if (r, r) in Q_dict:
                        Q_dict[(r, r)] += float(w)
                    else:
                        Q_dict[(r, r)] = float(w)
                else:
                    if r > c:
                        r, c = c, r
                    if (r, c) in Q_dict:
                        Q_dict[(r, c)] += 2.0 * float(w)
                    else:
                        Q_dict[(r, c)] = 2.0 * float(w)
            return Q_dict
    except Exception as e:
        print(f"  [READ ERROR] Unable to load sparse dict {file_path}: {e}")
        return None

def find_classical_ground_state(Q):
    """Executes a classical Brute Force calculation. Use only on small instances (< 20-25 nodes)."""
    print("  -> Calculating the True Mathematical Ground State (Brute Force)...")
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

# ==========================================
# 2. MAIN EXECUTION & PROCESSING
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print(f" D-WAVE BENCHMARK: {os.path.basename(FILE_PATH)}")
    print("="*50)

    # --- A. Classical Math Calculation ---
    Q_dense = load_hamburg_matrix_dense(FILE_PATH)
    if Q_dense is None:
        exit()
        
    n_atoms = len(Q_dense)
    true_ground_states, true_energy = find_classical_ground_state(Q_dense)
    print(f"  -> [OK] Found {len(true_ground_states)} mathematical optimal solutions with energy: {true_energy:.4f}")

    # --- B. D-Wave Quantum Execution ---
    Q_dict = load_hamburg_matrix_for_dwave_aligned(FILE_PATH)
    
    if USE_HYBRID:
        print("\n  -> Submitting to LeapHybridSampler...")
        sampler = LeapHybridSampler()
        sampleset = sampler.sample_qubo(Q_dict)
    else:
        print("\n  -> Submitting to pure QPU via EmbeddingComposite...")
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample_qubo(Q_dict, num_reads=NUM_READS, return_embedding=True)

    # --- C. Data Extraction & Formatting ---
    # Convert D-Wave dict samples to bitstrings and calculate relative frequencies (probabilities)
    dwave_samples = {}
    
    for sample, energy, num_occ in sampleset.data(['sample', 'energy', 'num_occurrences']):
        # Sort keys to ensure the bitstring matches the node order 0, 1, 2...
        bitstring = "".join(str(int(sample[i])) for i in sorted(sample.keys()))
        if bitstring in dwave_samples:
            dwave_samples[bitstring] += num_occ
        else:
            dwave_samples[bitstring] = num_occ

    # Normalize occurrences into probabilities
    total_reads = sum(dwave_samples.values())
    for bs in dwave_samples:
        dwave_samples[bs] /= total_reads

    # Sort dictionary by highest probability
    dwave_samples = dict(sorted(dwave_samples.items(), key=lambda item: item[1], reverse=True))

    # Identify the best state found by D-Wave
    best_solution = sampleset.first
    top_state_bitstring = "".join(str(int(best_solution.sample[i])) for i in sorted(best_solution.sample.keys()))
    top_state_prob = dwave_samples.get(top_state_bitstring, 0.0)
    top_state_energy = best_solution.energy

    print("\n--- ENERGY ANALYSIS ---")
    print(f"Most probable sampled state: {top_state_bitstring}")
    print(f"  -> Probability (Frequency): {top_state_prob:.4f}")
    print(f"  -> Calculated Energy: {top_state_energy:.4f}")
    print(f"  -> Energy gap from GS: {top_state_energy - true_energy:.4f}\n")

    # --- D. Metadata Extraction ---
    timing_data = sampleset.info.get('timing', {})
    qpu_time = timing_data.get('qpu_access_time', sampleset.info.get('run_time', 0.0))
    
    embedding = sampleset.info.get('embedding_context', {}).get('embedding', sampleset.info.get('embedding', {}))
    physical_qubits = sum(len(c) for c in embedding.values()) if embedding else n_atoms

    # ==========================================
    # 3. CSV LOGGING
    # ==========================================
    # Prepare data payload
    quantum_data = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Instance": os.path.basename(FILE_PATH),
        "N_Nodes": n_atoms,
        "Solver": "LeapHybrid" if USE_HYBRID else "Pure_QPU",
        "Num_Reads": 1 if USE_HYBRID else NUM_READS,
        "Top_State_QPU": top_state_bitstring,
        "Top_Prob_QPU": round(top_state_prob, 4),
        "Energy_QPU": round(top_state_energy, 4),
        "Gap_From_GS": round(top_state_energy - true_energy, 4),
        "Target_Reached": "YES" if top_state_bitstring in true_ground_states else "NO",
        "QPU_Access_Time_us": qpu_time,
        "Physical_Qubits_Used": physical_qubits
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    if os.path.exists(OUTPUT_CSV):
        df_history = pd.read_csv(OUTPUT_CSV)
        df_history = pd.concat([df_history, pd.DataFrame([quantum_data])], ignore_index=True)
    else:
        df_history = pd.DataFrame([quantum_data])

    df_history.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Archive '{OUTPUT_CSV}' updated and synchronized.\n")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    print("  -> Generating Quantum Histogram Plot...")
    
    # Take only the top 30 sampled states for the plot
    top_samples = dict(list(dwave_samples.items())[:30]) 

    # Color green if the bitstring is a true mathematical ground state
    bar_colors = ['green' if bit in true_ground_states else 'blue' for bit in top_samples.keys()]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(top_samples.keys(), top_samples.values(), color=bar_colors, alpha=0.7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='QPU Measurement (False Local Minima/Excited States)'),
        Patch(facecolor='green', alpha=0.7, label=f'True Mathematical Ground State ({true_energy:.2f})')
    ]
    plt.legend(handles=legend_elements)

    plt.xlabel("Measured Configurations (Bitstrings)")
    plt.ylabel(f"Probability (Frequency over {NUM_READS} reads)")
    file_name = os.path.basename(FILE_PATH)

    solver_name = "Hybrid" if USE_HYBRID else "QPU"
    plt.title(f"D-Wave {solver_name} Top 30 Emulated vs Math Optimal - {file_name} ({n_atoms} variables)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    for gs in true_ground_states:
        if gs in top_samples:
            print(f"True Ground State {gs} found in Top 30 at rank {list(top_samples.keys()).index(gs) + 1} with frequency {top_samples[gs]:.4f}")
        elif gs in dwave_samples:
            print(f"True Ground State {gs} found outside Top 30 (Frequency: {dwave_samples[gs]:.6f})")
        else:
            print(f"True Ground State {gs} NEVER sampled by QPU (Frequency 0.0)")

    plt.show()