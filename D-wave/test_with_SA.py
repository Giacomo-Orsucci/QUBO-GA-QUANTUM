# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT



#This is the same script of test_with_bruteforce but with Simulated Annealing to be able
#to test instances with >= 25-30 qubits.

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
import neal  


#Little script to test a little D-Wave platforms with SA bench.


# ==========================================
# 0. CONFIGURATION
# ==========================================
FILE_PATH = "../my_QUBO_instances/scaling_tests/jade_udg/jade_udg_100x100_R12_s142.npz"
#FILE_PATH = "../my_QUBO_instances/scaling_tests/jade_udg/jade_udg_50x50_R12_s99.npz" # Example for large instances
OUTPUT_CSV = "./new_csv/dwave_experiment_SA_100x100.csv"

NUM_READS = 2000 
USE_HYBRID = False # True if Pure Quantum QPU is not able to embed 50+ nodes

# ==========================================
# 1. PARSERS & CLASSICAL HEURISTIC SOLVER
# ==========================================
def load_hamburg_matrix_for_dwave_aligned(file_path):
    """Loads the sparse Q dictionary optimized and aligned for D-Wave & Neal."""
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

def find_classical_heuristic_ground_state(Q_dict, num_reads=1000):
    """
    Replaces brute force for large instances. Uses CPU Simulated Annealing.
    Returns the best state(s) and minimum energy found by the classical algorithm.
    """
    print(f"  -> Calculating Classical Baseline (Simulated Annealing - {num_reads} sweeps)...")
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
    
    best_energy = sampleset.first.energy
    best_states = []
    
    # Extract all states that share the lowest energy found by SA
    for sample, energy in sampleset.data(['sample', 'energy']):
        if np.isclose(energy, best_energy, atol=1e-5):
            bitstring = "".join(str(int(sample[i])) for i in sorted(sample.keys()))
            if bitstring not in best_states:
                best_states.append(bitstring)
        else:
            break # Sampleset is sorted by energy, so we can stop looking
            
    return best_states, best_energy

# ==========================================
# 2. MAIN EXECUTION & PROCESSING
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print(f" D-WAVE BENCHMARK (LARGE INSTANCE): {os.path.basename(FILE_PATH)}")
    print("="*50)

    # Load matrix once for both Classical SA and Quantum D-Wave
    Q_dict = load_hamburg_matrix_for_dwave_aligned(FILE_PATH)
    if Q_dict is None:
        exit()
        
    n_atoms = max(max(r, c) for r, c in Q_dict.keys()) + 1

    # --- A. Classical Heuristic Baseline ---
    sa_best_states, sa_best_energy = find_classical_heuristic_ground_state(Q_dict, num_reads=NUM_READS)
    print(f"  -> [OK] Classical SA found lowest energy: {sa_best_energy:.4f} (Degeneracy: {len(sa_best_states)})")

    # --- B. D-Wave Quantum Execution ---
    if USE_HYBRID:
        print("\n  -> Submitting to LeapHybridSampler...")
        sampler = LeapHybridSampler()
        sampleset = sampler.sample_qubo(Q_dict)
    else:
        print("\n  -> Submitting to pure QPU via EmbeddingComposite...")
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample_qubo(Q_dict, num_reads=NUM_READS, return_embedding=True)

    # --- C. Data Extraction & Formatting ---
    dwave_samples = {}
    for sample, energy, num_occ in sampleset.data(['sample', 'energy', 'num_occurrences']):
        bitstring = "".join(str(int(sample[i])) for i in sorted(sample.keys()))
        if bitstring in dwave_samples:
            dwave_samples[bitstring] += num_occ
        else:
            dwave_samples[bitstring] = num_occ

    total_reads = sum(dwave_samples.values())
    for bs in dwave_samples:
        dwave_samples[bs] /= total_reads

    dwave_samples = dict(sorted(dwave_samples.items(), key=lambda item: item[1], reverse=True))

    best_solution = sampleset.first
    top_state_bitstring = "".join(str(int(best_solution.sample[i])) for i in sorted(best_solution.sample.keys()))
    top_state_prob = dwave_samples.get(top_state_bitstring, 0.0)
    top_state_energy = best_solution.energy

    print("\n--- ENERGY ANALYSIS ---")
    print(f"Most probable sampled state: {top_state_bitstring}")
    print(f"  -> Probability (Frequency): {top_state_prob:.4f}")
    print(f"  -> Calculated Energy: {top_state_energy:.4f}")
    print(f"  -> Energy gap from SA Baseline: {top_state_energy - sa_best_energy:.4f}\n")

    # --- D. Metadata Extraction ---
    timing_data = sampleset.info.get('timing', {})
    qpu_time = timing_data.get('qpu_access_time', sampleset.info.get('run_time', 0.0))
    
    embedding = sampleset.info.get('embedding_context', {}).get('embedding', sampleset.info.get('embedding', {}))
    physical_qubits = sum(len(c) for c in embedding.values()) if embedding else n_atoms

    # ==========================================
    # 3. CSV LOGGING
    # ==========================================
    quantum_data = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Instance": os.path.basename(FILE_PATH),
        "N_Nodes": n_atoms,
        "Solver": "LeapHybrid" if USE_HYBRID else "Pure_QPU",
        "Num_Reads": 1 if USE_HYBRID else NUM_READS,
        "Top_State_QPU": top_state_bitstring,
        "Top_Prob_QPU": round(top_state_prob, 4),
        "Energy_QPU": round(top_state_energy, 4),
        "Gap_From_SA": round(top_state_energy - sa_best_energy, 4),
        "Matched_SA_Baseline": "YES" if top_state_bitstring in sa_best_states else "NO",
        "QPU_Access_Time_us": qpu_time,
        "Physical_Qubits_Used": physical_qubits
    }

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
    
    top_samples = dict(list(dwave_samples.items())[:30]) 

    # --- NEW COLOR LOGIC BASED ON ENERGY ---
    bar_colors = []
    for bitstring in top_samples.keys():
        # Convert the string into a list of integers
        x = [int(b) for b in bitstring]
        
        # Calculate the energy of the bitstring using our sparse Q_dict
        # We multiply the weight 'w' by the value of the nodes 'r' and 'c'
        energy = sum(w * x[r] * x[c] for (r, c), w in Q_dict.items())
        
        # If the calculated energy matches the SA one (using isclose for floats), it becomes green
        if np.isclose(energy, sa_best_energy, atol=1e-5):
            bar_colors.append('green')
        else:
            bar_colors.append('blue')
    # ---------------------------------------------------

    plt.figure(figsize=(14, 7))
    bars = plt.bar(top_samples.keys(), top_samples.values(), color=bar_colors, alpha=0.7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='QPU Measurement (Sub-optimal or Excited States)'),
        Patch(facecolor='green', alpha=0.7, label=f'Matched SA Baseline Energy ({sa_best_energy:.2f})')
    ]
    plt.legend(handles=legend_elements)

    plt.xlabel("Measured Configurations (Bitstrings)")
    plt.ylabel(f"Probability (Frequency over {NUM_READS} reads)")
    file_name = os.path.basename(FILE_PATH)

    solver_name = "Hybrid" if USE_HYBRID else "QPU"
    plt.title(f"D-Wave {solver_name} vs Classical SA Baseline - {file_name} ({n_atoms} variables)")
    
    # Hide x-axis labels if the bitstrings are too long (e.g., > 30 chars) to prevent overlap
    if n_atoms > 30:
        plt.xticks([])
        plt.xlabel(f"Measured Configurations (Bitstrings hidden for readability due to n={n_atoms})")
    else:
        plt.xticks(rotation=45, ha='right')
        
    plt.tight_layout()

    # Print check for SA states
    for gs in sa_best_states:
        if gs in top_samples:
            print(f"SA Baseline State {gs[:10]}... found in Top 30 at rank {list(top_samples.keys()).index(gs) + 1} with frequency {top_samples[gs]:.4f}")
        elif gs in dwave_samples:
            print(f"SA Baseline State {gs[:10]}... found outside Top 30 (Frequency: {dwave_samples[gs]:.6f})")
        else:
            print(f"SA Baseline State {gs[:10]}... NEVER sampled by QPU (Frequency 0.0)")

    plt.show()