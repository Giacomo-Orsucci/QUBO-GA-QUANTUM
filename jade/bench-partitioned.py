# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

import os
import time
import json
import numpy as np
import pygad
import pandas as pd
import itertools 
import dataclasses
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering

from pulser import InterpolatedWaveform, Pulse, Sequence, Register
from pulser_myqlm import IsingAQPU

# ==========================================
# 1. HARDWARE SETUP (Fake Jade)
# ==========================================
try:
    from pulser.devices import Jade as target_device
except ImportError:
    from pulser.devices import AnalogDevice
    try:
        target_device = dataclasses.replace(AnalogDevice, name="FakeJade", max_radial_distance=50)
    except TypeError:
        target_device = dataclasses.replace(AnalogDevice, name="FakeJade", maximum_radial_distance=50)
    print("[WARNING] Jade profile injected artificially (Radius extended to 50 µm).")

device = target_device
MIN_DIST = device.min_atom_distance
MAX_RADIUS = device.max_radial_distance if hasattr(device, 'max_radial_distance') else 50

TARGET_FILE = ".././my_QUBO_instances/scaling_tests/jade_udg/jade_udg_30x30_R12_s72.npz"
n_restarts = 5
# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_matrix(file_path):
    """Loads the QUBO matrix from an .npz file."""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            i_indices, j_indices, weights = data['i'], data['j'], data['Jij']
            num_nodes = int(max(np.max(i_indices), np.max(j_indices))) + 1
            q_matrix = np.zeros((num_nodes, num_nodes))
            for r, c, w in zip(i_indices, j_indices, weights):
                r, c = int(r), int(c)
                q_matrix[r, c] = w
                if r != c: q_matrix[c, r] = w
            return q_matrix
    except Exception as e:
        print(f"  [READ ERROR] Unable to load {file_path}: {e}")
        return None
    
def solve_classically_exact(q_matrix):
    """Solves small QUBO instances (1 or 2 nodes) using exact brute force."""
    num_nodes = len(q_matrix)
    best_energy = float('inf')
    best_state = None
    for bits in itertools.product([0, 1], repeat=num_nodes):
        state_array = np.array(bits)
        energy = state_array.T @ q_matrix @ state_array
        if energy < best_energy:
            best_energy = energy
            best_state = bits
    return best_state

# ==========================================
# 3. TOPOLOGICAL CLUSTERING (Grid Partitioning equivalent)
# ==========================================
def partition_qubo_topological(q_matrix, target_cluster_size=5):
    """
    Divides the QUBO matrix into strongly connected clusters using Spectral Clustering.
    This acts as the 'Grid Partitioning' step for non-spatial graphs.
    """
    num_nodes = len(q_matrix)
    
    # Determine the number of clusters needed to hit the target size
    num_clusters = max(1, num_nodes // target_cluster_size)
    
    if num_clusters == 1:
        return {0: list(range(num_nodes))}

    # Create the affinity matrix using the absolute values of the connections.
    # We ignore the diagonal (linear weights) as it doesn't define topology.
    affinity_matrix = np.abs(q_matrix.copy())
    np.fill_diagonal(affinity_matrix, 0)
    
    print(f"  -> Starting Spectral Clustering to create {num_clusters} clusters...")
    
    # Random state fixed for reproducibility during testing
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
    labels = clustering.fit_predict(affinity_matrix)
    
    # Map the cluster ID to the original global node indices
    clusters = {i: [] for i in range(num_clusters)}
    for node_idx, cluster_id in enumerate(labels):
        clusters[cluster_id].append(node_idx)
        
    return clusters

# ==========================================
# 4. GENETIC ENGINE (Local Embedding Optimization)
# ==========================================
def optimize_embedding(q_matrix, num_restarts=10):
    """
    Runs the Genetic Algorithm to find the optimal 2D coordinates for a SUB-GRAPH.
    """
    num_atoms = len(q_matrix)
    q_off_diag = q_matrix.copy()
    np.fill_diagonal(q_off_diag, 0)

    # Physical Scale Factor Calculation
    v_max_allowed = device.interaction_coeff / (MIN_DIST**6)
    q_max_off_diag = np.max(q_off_diag)
    scale_space = v_max_allowed / q_max_off_diag if q_max_off_diag > 0 else float('inf')

    channel = device.channels["rydberg_global"]
    max_detuning = channel.max_abs_detuning if channel.max_abs_detuning is not None else 40.0
    avg_linear_weight = np.mean(np.abs(np.diag(q_matrix)))
    scale_laser = max_detuning / avg_linear_weight if avg_linear_weight > 0 else float('inf')

    scale_factor = min(scale_space, scale_laser)
    q_target = q_off_diag * scale_factor

    def improved_topological_mae_fitness_func(ga_instance, solution, solution_idx):
        coords = np.reshape(solution, (num_atoms, 2))
        distances = pdist(coords)
        
        # Calculate physical interactions based on distance
        v_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        v_triu = v_physical[np.triu_indices(num_atoms, k=1)]
        q_triu = q_target[np.triu_indices(num_atoms, k=1)]
        
        penalty = 0.0

        # Penalize layout if atoms are closer than the hardware allows
        if np.any(distances < MIN_DIST):
            penalty += np.sum(np.clip(MIN_DIST - distances, 0, None)) * 100000.0 
            
        # Penalize layout if atoms are placed outside the maximum laser radius
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            penalty += np.sum(np.clip(radii - MAX_RADIUS, 0, None)) * 100000.0

        max_q = np.max(np.abs(q_triu)) if np.max(np.abs(q_triu)) > 0 else 1.0
        v_triu_clipped = np.clip(v_triu, 0, max_q * 5.0)

        # Calculate error on active vs inactive bonds
        active_bonds = np.abs(q_triu) > 1e-5
        error_active = np.sum(np.abs(v_triu_clipped[active_bonds] - q_triu[active_bonds]))
        error_inactive = np.sum(v_triu_clipped[~active_bonds]) * 2.0 
        
        base_error = error_active + error_inactive
        return 1.0 / (base_error + penalty + 1e-6)

    gene_space = [{'low': -MAX_RADIUS, 'high': MAX_RADIUS} for _ in range(num_atoms * 2)]
    best_overall_fitness = -float('inf')
    best_overall_coords = None

    print(f"  -> Starting Multi-Start ({num_restarts} runs)...")
    for run_idx in range(num_restarts):
        ga = pygad.GA(
            num_generations=1500, 
            num_parents_mating=20, 
            fitness_func=improved_topological_mae_fitness_func, 
            sol_per_pop=200, 
            num_genes=num_atoms * 2,
            gene_space=gene_space,
            parent_selection_type="tournament",
            K_tournament=4, 
            keep_elitism=10, 
            crossover_type="uniform",
            mutation_type="adaptive",
            mutation_probability=[0.20, 0.05], 
            random_mutation_min_val=-2.0,  
            random_mutation_max_val=2.0,   
            allow_duplicate_genes=False,
            suppress_warnings=True
        )
        ga.run()
        run_best_solution, run_best_fitness, _ = ga.best_solution()
        
        if run_best_fitness > best_overall_fitness:
            best_overall_fitness = run_best_fitness
            best_overall_coords = np.reshape(run_best_solution, (num_atoms, 2))
            print(f"    [Run {run_idx+1}] New Record! Fitness: {run_best_fitness:.4f}")

    print("GA executed. Next step: job preparation")
            
    return best_overall_coords, best_overall_fitness, scale_factor

# ==========================================
# 5. ADIABATIC QUANTUM ENGINE 
# ==========================================
def run_quantum_job(q_matrix, coords, scale_factor, qpu_emulator):
    """
    Submits a job to the quantum emulator.
    The global laser is now optimized specifically for this sub-graph.
    """
    num_atoms = len(q_matrix)
    q_off_diag = q_matrix.copy()
    np.fill_diagonal(q_off_diag, 0)
    q_target = q_off_diag * scale_factor
    
    avg_linear_weight = np.mean(np.abs(np.diag(q_matrix)))
    scaled_delta = avg_linear_weight * scale_factor
    
    qubits = {f"q{i}": c for i, c in enumerate(coords)}
    reg = Register(qubits)
    
    ideal_omega = np.median(q_target[q_target > 0]) if np.any(q_target > 0) else 1.0
    channel_max_amp = device.channels["rydberg_global"].max_amp
    omega = min(ideal_omega, channel_max_amp / 1.2) if channel_max_amp else ideal_omega

    total_time = 4000 
    delta_initial = -scaled_delta
    delta_final = scaled_delta

    adiabatic_pulse = Pulse(
        InterpolatedWaveform(total_time, [1e-9, omega, 1e-9]),
        InterpolatedWaveform(total_time, [delta_initial, 0, delta_final]),
        0,
    )

    seq = Sequence(reg, device)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")

    job = IsingAQPU.convert_sequence_to_job(seq, nbshots=0)
            
    # --- ASYNCHRONOUS MODIFICATION (NUCLEAR OPTION) ---
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            async_job = qpu_emulator.submit(job)
            
            # Opzione Nucleare: NESSUN controllo su attributi (niente .batch_id).
            # Convertiamo subito la memoria grezza in stringa.
            str_rep = repr(async_job)
            
            if "SJob" in str_rep:
                job_id = "SJob" + str_rep.split("SJob")[1].split()[0].strip(">'\")")
            else:
                # Fallback di sicurezza assoluta per non bloccare mai il ciclo
                job_id = str_rep 
                
            return job_id
            
        except Exception as e:
            print(f"     [!] Submission error (Attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                return "SUBMISSION_ERROR"

# ==========================================
# 6. ORCHESTRATOR RUN
# ==========================================
if __name__ == "__main__":
    print("Initializing virtual QPU...")
    try:
        from qlmaas.qpus import AnalogQPU
        qpu_emulator = AnalogQPU()
    except ImportError:
        try:
            from qat.qlmaas.qpus import QLMaaSQPU
            qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
        except ImportError:
            print("[WARNING] Remote server not found, using local emulator...")
            qpu_emulator = IsingAQPU()

    print(f"\n--- COMPILING DISTRIBUTED CASE: {TARGET_FILE} ---")
    
    q_global_matrix = load_matrix(TARGET_FILE)
    if q_global_matrix is None:
        exit()

    # Partitioning: default target size set to 5 for fast 15x15/30x30 tests
    clusters = partition_qubo_topological(q_global_matrix, target_cluster_size=5)
    
    job_registry = {} 
    ga_times_log = [] # List to collect GA execution metrics
    
    for cluster_id, global_nodes in clusters.items():
        print(f"\n--- Processing Cluster {cluster_id + 1}/{len(clusters)} ({len(global_nodes)} nodes) ---")
        
        sub_q_matrix = q_global_matrix[np.ix_(global_nodes, global_nodes)]
        
        # Handling very small clusters (trivial classical cases)
        if len(global_nodes) < 3:
            print("  -> Cluster too small. Solving exact classically...")
            best_bits = solve_classically_exact(sub_q_matrix)
            bit_string = "".join(map(str, best_bits))
            
            fake_job_id = f"LOCAL_EXACT_{bit_string}"
            job_registry[fake_job_id] = global_nodes
            
            # Record "EXACT" in fitness since it's solved without spatial approximations
            ga_times_log.append({
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Instance": os.path.basename(TARGET_FILE),
                "Cluster_ID": cluster_id + 1,
                "Nodes": len(global_nodes),
                "Job_ID": fake_job_id,
                "GA_Time_s": 0.0,
                "Fitness": "EXACT" 
            })
            continue
            
        start_time = time.time()
        
        # Extracting coordinates, fitness, and scale factor
        coords, fitness, scale_factor = optimize_embedding(sub_q_matrix, num_restarts=n_restarts) 
        ga_execution_time = time.time() - start_time
        
        job_id = run_quantum_job(sub_q_matrix, coords, scale_factor, qpu_emulator)
        job_registry[job_id] = global_nodes
        
        print(f"  -> Cluster {cluster_id + 1} submitted. Job ID: {job_id}. GA Time: {ga_execution_time:.2f}s | Fitness: {fitness:.4e}")
        
        # Save cluster metrics including the "Fitness" column
        ga_times_log.append({
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Instance": os.path.basename(TARGET_FILE),
            "Cluster_ID": cluster_id + 1,
            "Nodes": len(global_nodes),
            "Job_ID": job_id,
            "GA_Time_s": round(ga_execution_time, 2),
            "Fitness": round(fitness, 6) # Rounded to 6 decimal places
        })

    print("\n" + "="*50)
    print(" PHASE 1 COMPLETED: All clusters are currently executing.")

    # Extract the clean instance name (e.g., "jade_udg_15x15_R12_s57")
    instance_name = os.path.splitext(os.path.basename(TARGET_FILE))[0]
    
    JSON_FILE = f"./distributed_json/cluster_jobs_registry_{instance_name}.json"
    # --- JSON REGISTRY SAVING (Int64 conversion fix) ---
    safe_registry = {j_id: [int(n) for n in nodes] for j_id, nodes in job_registry.items()}
    with open(JSON_FILE, "w") as f:
        json.dump(safe_registry, f)
    print(" [JSON] Job registry successfully saved.")

    # --- GA EXECUTION TIMES CSV SAVING ---
    
    
    # Insert the instance name into the CSV file path
    GA_CSV_FILE = f"./distributed_time_csv/ga_times_{instance_name}.csv"
    os.makedirs(os.path.dirname(GA_CSV_FILE), exist_ok=True) 
    
    if os.path.exists(GA_CSV_FILE):
        df_history = pd.read_csv(GA_CSV_FILE)
        df_new = pd.DataFrame(ga_times_log)
        df_updated = pd.concat([df_history, df_new], ignore_index=True)
    else:
        df_updated = pd.DataFrame(ga_times_log)
        
    df_updated.to_csv(GA_CSV_FILE, index=False)
    print(f" [CSV] GA execution times saved in '{GA_CSV_FILE}'.")
    print("="*50)