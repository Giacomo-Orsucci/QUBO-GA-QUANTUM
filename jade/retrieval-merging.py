# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

import os
import time
import json
import numpy as np
import pandas as pd
from os import getenv
from qat.qlmaas.connection import QLMaaSConnection
import re


import neal #to use SA

# --- 0. INITIAL SETUP ---
FITNESS_TYPE = "Improved_Topological_MAE" 
FILE_PATH = ".././my_QUBO_instances/scaling_tests/jade_udg/jade_udg_30x30_R12_s72.npz"


# Extract the clean instance name (e.g., "jade_udg_15x15_R12_s57")
instance_name = os.path.splitext(os.path.basename(FILE_PATH))[0]

OUTPUT_CSV = f"./retrieval_csv-SA-GREEDY-INIT-PARTITIONED/experiment_registry_distributed_{instance_name}.csv"
REGISTRY_FILE = f"./distributed_json/cluster_jobs_registry_{instance_name}.json"


#Possible improvement: find the best energy solution among n first solution and stop getting only the

# --- QUBO INSTANCE LOADER ---
def load_matrix(file_path):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            i_indices, j_indices, weights = data['i'], data['j'], data['Jij']
            n_nodes = int(max(np.max(i_indices), np.max(j_indices))) + 1
            Q = np.zeros((n_nodes, n_nodes))
            for r, c, w in zip(i_indices, j_indices, weights):
                r, c = int(r), int(c)
                Q[r, c] = w
                if r != c: Q[c, r] = w
            return Q
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return None

def greedy_merge(Q_global, local_results_dict):
    print("\n  -> [PHASE 3] Starting Greedy Merging of local solutions...")
    proposed_ones = [int(node) for node, bit in local_results_dict.items() if bit == 1]
    proposed_ones.sort(key=lambda x: Q_global[x, x])
    
    final_solution = np.zeros(len(Q_global), dtype=int)
    accepted_nodes = []
    
    for node in proposed_ones:
        conflict = False
        for acc_node in accepted_nodes:
            if Q_global[node, acc_node] > 0:
                conflict = True
                break
        
        if not conflict:
            final_solution[node] = 1
            accepted_nodes.append(node)
            
    return final_solution

# --- 2. CLASSICAL BENCHMARK (USING D-WAVE NEAL) ---
def run_neal_sa_benchmark(Q):
    """
    Uses the dwave-neal library to perform Simulated Annealing on the QUBO matrix.
    """
    print("\n  -> [BENCHMARK] Running D-Wave Neal Simulated Annealing...")
    start_time = time.time()
    
    # D-Wave expects the QUBO as a dictionary: {(i, j): weight}
    qubo_dict = {}
    N = len(Q)
    for i in range(N):
        for j in range(i, N): # Only upper triangle is needed
            if Q[i, j] != 0:
                qubo_dict[(i, j)] = Q[i, j]
                
    # Initialize the sampler
    sampler = neal.SimulatedAnnealingSampler()
    
    # Run the sampling
    # num_reads = number of independent SA runs. 1000 is a solid number for finding good minimums.
    response = sampler.sample_qubo(qubo_dict, num_reads=1000)
    
    # Get the best sample (the one with the lowest energy)
    best_sample = response.first.sample
    best_energy = response.first.energy
    
    # Convert the dictionary output back to a numpy array for consistency
    best_x = np.zeros(N, dtype=int)
    for idx, bit_val in best_sample.items():
        best_x[int(idx)] = bit_val
        
    execution_time = time.time() - start_time
    print(f"     [OK] Neal SA finished in {execution_time:.2f}s. Best Energy: {best_energy:.4f}")
    
    return best_x, best_energy, execution_time

def run_neal_sa_tuning(Q, num_reads=1000, initial_state=None):
    """
    SA fine tuning starting from greedy merged solution.
    """
    start_time = time.time()
    
    #Q matrix translation in D-Wave SA (for fine tuning)
    qubo_dict = {}
    N = len(Q)
    for i in range(N):
        for j in range(i, N): 
            if Q[i, j] != 0:
                qubo_dict[(i, j)] = Q[i, j]
                
    sampler = neal.SimulatedAnnealingSampler()
    
    if initial_state is not None:
        initial_state_dict = {i: int(initial_state[i]) for i in range(N)}
        # Beta range (inverse of T). We start form 5.0. 
        # Low temperature to respect our original solution.
        response = sampler.sample_qubo(
            qubo_dict, 
            num_reads=num_reads, 
            initial_states=[initial_state_dict] * num_reads,
            beta_range=[5.0, 100.0] 
        )
    
    best_sample = response.first.sample
    best_energy = response.first.energy
    
    best_x = np.zeros(N, dtype=int)
    for idx, bit_val in best_sample.items():
        best_x[int(idx)] = bit_val
        
    return best_x, best_energy, time.time() - start_time


# --- 3. LOAD GLOBAL DATA & REGISTRY ---
Q_global = load_matrix(FILE_PATH)
if Q_global is None: exit()
N_ATOMS_GLOBAL = len(Q_global)

print(f"--- DISTRIBUTED RETRIEVAL & BENCHMARK: {os.path.basename(FILE_PATH)} ({N_ATOMS_GLOBAL} nodes) ---")

try:
    with open(REGISTRY_FILE, "r") as f:
        job_registry = json.load(f)
except FileNotFoundError:
    print(f"ERROR: File '{REGISTRY_FILE}' not found. Please run the submission script first.")
    exit()

# --- 4. MULTI-JOB RETRIEVAL ---
print("\n  -> [PHASE 2] Connecting to Jülich to retrieve cluster data...")
try:
    connection = QLMaaSConnection()
except Exception as e:
    print(f"Connection Error: {e}")
    exit()

global_results_map = {}
all_jobs_completed = True

for job_id, global_nodes in job_registry.items():
    # 1. Intercepets locally solved tasks
    if job_id.startswith("LOCAL_EXACT_"):
        best_bitstring = job_id.split("_")[-1]
        print(f"     [OK] Loaded exact local solution for {len(global_nodes)} nodes (Bits: {best_bitstring})")
        for local_idx, bit_str in enumerate(best_bitstring):
            global_idx = global_nodes[local_idx]
            global_results_map[global_idx] = int(bit_str)
        continue
        
    print(f"     Checking Job {job_id} (Cluster of {len(global_nodes)} nodes)...")
    status = connection.get_status(job_id)
    
    if status != 'done':
        print(f"     [!] Job {job_id} is still in status: {status}. Try again later.")
        all_jobs_completed = False
        continue
        
    results = connection.get_result(job_id)
    
    best_prob = -1
    best_bitstring = None
    
    for sample in results.raw_data:
        if sample.probability > best_prob:
            best_prob = sample.probability
            bits_only = re.sub(r'[^01]', '', str(sample.state))
            best_bitstring = bits_only.zfill(len(global_nodes))
            
    if best_bitstring:
        for local_idx, bit_str in enumerate(best_bitstring):
            global_idx = global_nodes[local_idx]
            global_results_map[global_idx] = int(bit_str)
            
    print(f"     [OK] Job {job_id} downloaded. (Highest Probability: {best_prob:.3f})")

if not all_jobs_completed:
    print("\n[WARNING] Not all jobs are completed. Execution halted to wait for pending results.")
    exit()

for i in range(N_ATOMS_GLOBAL):
    if i not in global_results_map:
        global_results_map[i] = 0

# ---------------------------------------------------------
# FASE 3: GREEDY MERGING and POST-PROCESSING
# ---------------------------------------------------------
print("\n  -> [PHASE 3] Starting Greedy Merging of local solutions...")
final_global_bitstring = greedy_merge(Q_global, global_results_map)
qpu_energy = final_global_bitstring.T @ Q_global @ final_global_bitstring

print("\n  -> [TUNING] Refining QPU Hybrid Solution with SA...")
# post processing tuning with SA on greedy merged solution.
tuned_bitstring, tuned_energy, tuned_time = run_neal_sa_tuning(
    Q_global, 
    num_reads=500, # Not so much readings, we are confident the greedy merge found something near the solution
    initial_state=final_global_bitstring
)

print("\n  -> [BENCHMARK-BASELINE] Running D-Wave Neal Simulated Annealing from scratch...")
sa_bitstring, sa_energy, sa_time = run_neal_sa_benchmark(Q_global)

# ---------------------------------------------------------
# FINAL RESULTS
# ---------------------------------------------------------
print("\n" + "="*60)
print(" EXPERIMENT RESULTS & COMPARISON")
print("="*60)
print(f" [1] Quantum Hybrid (AHS + Greedy)   : {qpu_energy:.4f}")
print(f" [2] Hybrid Tuned (QPU + SA)         : {tuned_energy:.4f}")
print(f" [3] Classical SA Benchmark          : {sa_energy:.4f}")
print("-" * 60)
    
if tuned_energy <= sa_energy:
    print(" SUCCESS: Hybrid Tuned matches or beats Pure Classical SA!")
else:
    print(" Pure Classical SA still found a better global minimum.")
    print("="*60)

# --- 7. CSV LOGGING ---
master_job_signature = f"DISTRIBUTED_{len(job_registry)}_JOBS"

experiment_data = {
    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "Instance": os.path.basename(FILE_PATH),
    "N_Nodes_Global": N_ATOMS_GLOBAL,
    "N_Clusters": len(job_registry),
    "Juelich_Master_ID": master_job_signature,
    "Fitness_Metric": FITNESS_TYPE,
    "Energy_Hybrid_QPU": round(qpu_energy, 4),
    "Energy_Hybrid_Tuned": round(tuned_energy, 4),
    "Energy_Classic_SA": round(sa_energy, 4),
    "SA_Tuning_Time_s": round(tuned_time, 2),
    "SA_Execution_Time_s": round(sa_time, 2)
}

if os.path.exists(OUTPUT_CSV):
    df_history = pd.read_csv(OUTPUT_CSV)
    df_new = pd.DataFrame([experiment_data])
    df_updated = pd.concat([df_history, df_new], ignore_index=True)
else:
    df_updated = pd.DataFrame([experiment_data])

df_updated.to_csv(OUTPUT_CSV, index=False)
print(f"\n[CSV] Comparative global data successfully saved in '{OUTPUT_CSV}'.")