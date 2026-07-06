# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

import os
import time
import json
import numpy as np
import pandas as pd
import itertools 
import dataclasses
import random
import math
import socket
import io
import re
from contextlib import redirect_stdout, redirect_stderr
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

# Target file for the scaling test instance
TARGET_FILE = ".././my_QUBO_instances/scaling_tests/jade_udg/jade_udg_30x30_R12_s72.npz"
instance_name = os.path.splitext(os.path.basename(TARGET_FILE))[0]

OUTPUT_CSV = f".././new_csv/experiment_registry_SA_partitioned_{instance_name}.csv"
TARGET_CLUSTER_SIZE = 15

# ==========================================
# 2. DYNAMIC SIMULATED ANNEALING PARAMETERS
# ==========================================
def get_sa_hyperparameters(n_nodes):
    """
    Dynamically scales the thermodynamic parameters based on 
    the spatial complexity (number of atoms) of the cluster.
    """
    if n_nodes <= 8:
        
        return {"restarts": 5, "t_init": 80.0, "t_min": 0.1, "cooling": 0.95, "steps": 300}
    elif n_nodes <= 12:
        return {"restarts": 5, "t_init": 100.0, "t_min": 0.01, "cooling": 0.98, "steps": 1000}
    elif n_nodes <= 16:
        return {"restarts": 5, "t_init": 1000.0, "t_min": 0.01, "cooling": 0.99, "steps": 1200}
    else:
        return {"restarts": 5, "t_init": 2000.0, "t_min": 0.01, "cooling": 0.995, "steps": 2000}

# ==========================================
# 3. HELPER FUNCTIONS
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
    """Solves small QUBO instances using exact brute force."""
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
# 4. TOPOLOGICAL CLUSTERING (Recursive Size-Capped)
# ==========================================
def partition_qubo_topological(q_matrix, target_cluster_size=10, tolerance=2):
    """
    Divides the QUBO matrix into strongly connected clusters.
    Uses a recursive approach to guarantee that NO cluster exceeds 
    (target_cluster_size + tolerance) in size.
    """
    max_allowed_size = target_cluster_size + tolerance
    
    def recursive_split(current_nodes):
        # If the block is already within limits, keep it as is.
        if len(current_nodes) <= max_allowed_size:
            return [current_nodes]
        
        # If it's too large, calculate how many pieces to divide it into.
        n_clusters = max(2, round(len(current_nodes) / target_cluster_size))
        print(f"    -> Cluster of {len(current_nodes)} nodes is too large. Splitting into {n_clusters} sub-clusters...")
        
        # Extract the sub-matrix for these nodes only
        sub_q = q_matrix[np.ix_(current_nodes, current_nodes)]
        affinity = np.abs(sub_q.copy())
        np.fill_diagonal(affinity, 0)
        
        # Spectral cut
        clustering = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed', 
            assign_labels='kmeans', 
            random_state=42
        )
        labels = clustering.fit_predict(affinity)
        
        # Group results and recursively check the new pieces
        final_pieces = []
        for i in range(n_clusters):
            # Map sub-cluster nodes back to their original global indices
            piece_nodes = [current_nodes[idx] for idx, label in enumerate(labels) if label == i]
            final_pieces.extend(recursive_split(piece_nodes))
            
        return final_pieces

    print(f"  -> Starting Spectral Clustering (Target: {target_cluster_size}, Max Nodes: {max_allowed_size})...")
    
    all_nodes = list(range(len(q_matrix)))
    safe_clusters = recursive_split(all_nodes)
    
    cluster_dict = {i: nodes for i, nodes in enumerate(safe_clusters)}
    
    print(f"  -> Graph successfully partitioned into {len(cluster_dict)} safe clusters.")
    return cluster_dict

# ==========================================
# 5. SA ENGINE (GRASP + Quenching + Early Stopping)
# ==========================================
def optimize_embedding(Q, sa_params):
    """
    Runs Simulated Annealing to find the optimal 2D coordinates for a SUB-GRAPH.
    Includes an Early Stopping mechanism to halt runs if a good enough solution is found.
    """
    # Extract dynamic parameters
    num_restarts = sa_params["restarts"]
    t_init = sa_params["t_init"]
    t_min = sa_params["t_min"]
    cooling_rate = sa_params["cooling"]
    steps_per_temp = sa_params["steps"]

    num_atoms = len(Q)
    q_off_diag = Q.copy()
    np.fill_diagonal(q_off_diag, 0)

    # Physical Scale Factor Calculation
    v_max_allowed = device.interaction_coeff / (MIN_DIST**6)
    q_max_off_diag = np.max(q_off_diag)
    scale_space = v_max_allowed / q_max_off_diag if q_max_off_diag > 0 else float('inf')

    channel = device.channels["rydberg_global"]
    max_detuning = channel.max_abs_detuning if channel.max_abs_detuning is not None else 40.0
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scale_laser = max_detuning / avg_linear_weight if avg_linear_weight > 0 else float('inf')

    scale_factor = min(scale_space, scale_laser)
    q_target = q_off_diag * scale_factor

    q_triu_full = q_target.copy() 
    np.fill_diagonal(q_triu_full, 0)

    def calculate_energy(coords):
        distances = pdist(coords)
        v_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        
        v_triu = v_physical[np.triu_indices(num_atoms, k=1)]
        q_triu = q_target[np.triu_indices(num_atoms, k=1)]
        
        penalty = 0.0
        
        if np.any(distances < MIN_DIST):
            penalty += np.sum(np.clip(MIN_DIST - distances, 0, None)) * 100000.0 
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            penalty += np.sum(np.clip(radii - MAX_RADIUS, 0, None)) * 100000.0

        max_q = np.max(np.abs(q_triu)) if np.max(np.abs(q_triu)) > 0 else 1.0
        v_triu_clipped = np.clip(v_triu, 0, max_q * 5.0)

        active_bonds = np.abs(q_triu) > 1e-5
        
        error_active = np.sum(np.abs(v_triu_clipped[active_bonds] - q_triu[active_bonds]))
        error_inactive = np.sum(v_triu_clipped[~active_bonds]) * 2.0 
        
        return error_active + error_inactive + penalty

    def perturb_coordinates(coords, temp, max_temp):
        new_coords = coords.copy()
        move_type = random.random()
        
        jitter_scale = (temp / max_temp) * 3.0 + 0.2 

        if move_type < 0.3:
            idx = random.randint(0, num_atoms - 1)
            new_coords[idx][0] += random.uniform(-jitter_scale, jitter_scale)
            new_coords[idx][1] += random.uniform(-jitter_scale, jitter_scale)
        elif move_type < 0.9:
            idx1, idx2 = random.sample(range(num_atoms), 2)
            new_coords[idx1], new_coords[idx2] = new_coords[idx2].copy(), new_coords[idx1].copy()
        else:
            idx = random.randint(0, num_atoms - 1)
            r = random.uniform(0, MAX_RADIUS * 0.8) 
            theta = random.uniform(0, 2 * np.pi)
            new_coords[idx] = [r * np.cos(theta), r * np.sin(theta)]
            
        return new_coords

    best_overall_energy = float('inf')
    best_overall_coords = None

    print(f"  -> Starting SA with GRASP Init & Quenching ({num_restarts} runs max)...")
    
    for run_idx in range(num_restarts):
        
        # --- GRASP Placer ---
        current_coords = np.zeros((num_atoms, 2))
        node_weights = np.sum(np.abs(q_triu_full), axis=1) + np.sum(np.abs(q_triu_full), axis=0)
        sorted_nodes = np.argsort(node_weights)[::-1].tolist()
        
        top_candidates = sorted_nodes[:min(3, len(sorted_nodes))]
        boss_node = random.choice(top_candidates)
        sorted_nodes.remove(boss_node)
        
        current_coords[boss_node] = [0.0, 0.0]
        placed_nodes = [boss_node]
        
        for node in sorted_nodes:
            best_target = None
            max_bond = 0
            for p in placed_nodes:
                bond = abs(q_target[min(node, p), max(node, p)])
                if bond > max_bond:
                    max_bond = bond
                    best_target = p
            
            if best_target is not None and max_bond > 1e-5:
                angle = random.uniform(0, 2 * np.pi)
                r_orbit = MIN_DIST * random.uniform(1.2, 1.8) 
                current_coords[node] = [
                    current_coords[best_target][0] + r_orbit * np.cos(angle),
                    current_coords[best_target][1] + r_orbit * np.sin(angle)
                ]
            else:
                angle = random.uniform(0, 2 * np.pi)
                current_coords[node] = [
                    (MAX_RADIUS - MIN_DIST) * np.cos(angle),
                    (MAX_RADIUS - MIN_DIST) * np.sin(angle)
                ]
            placed_nodes.append(node)

        current_energy = calculate_energy(current_coords)
        best_run_coords = current_coords.copy()
        best_run_energy = current_energy
        current_temp = t_init
        
        # --- COOLING PHASE ---
        while current_temp > t_min:
            for step in range(steps_per_temp):
                candidate_coords = perturb_coordinates(current_coords, current_temp, t_init)
                candidate_energy = calculate_energy(candidate_coords)
                
                delta_e = candidate_energy - current_energy
                
                if delta_e < 0 or random.random() < math.exp(-delta_e / current_temp):
                    current_coords = candidate_coords
                    current_energy = candidate_energy
                    
                    if current_energy < best_run_energy:
                        best_run_energy = current_energy
                        best_run_coords = current_coords.copy()
            
            current_temp *= cooling_rate
            
        # --- QUENCHING PHASE ---
        for _ in range(1500): 
            idx = random.randint(0, num_atoms - 1)
            cand_coords = best_run_coords.copy()
            
            cand_coords[idx][0] += random.uniform(-0.3, 0.3) 
            cand_coords[idx][1] += random.uniform(-0.3, 0.3)
            
            cand_e = calculate_energy(cand_coords)
            
            if cand_e < best_run_energy:
                best_run_energy = cand_e
                best_run_coords = cand_coords.copy()
            
        if best_run_energy < best_overall_energy:
            best_overall_energy = best_run_energy
            best_overall_coords = best_run_coords.copy()
            print(f"    [Run {run_idx+1}] New Record! Energy: {best_overall_energy:.4f}")

        # --- EARLY STOPPING CONDITION (DYNAMIC) ---
        # The acceptable error scales with the size of the cluster
        dynamic_target = num_atoms * 12.0 
        
        if best_overall_energy < dynamic_target:
            print(f"    [Early Stop] Target energy achieved ({best_overall_energy:.4f} < {dynamic_target}). Skipping remaining runs.")
            break

    print("  -> SA executed successfully. Next step: job preparation.")
    final_fitness = 1.0 / (best_overall_energy + 1e-6)
            
    return best_overall_coords, best_overall_energy, final_fitness, scale_factor
# ==========================================
# 6. ADIABATIC QUANTUM ENGINE (Strictly Remote QLMaaS)
# ==========================================
def run_quantum_job(q_matrix, coords, scale_factor):
    """
    Submits a job STRICTLY to the remote quantum emulator (Jülich QLMaaS).
    No local fallback allowed. Retries connection persistently on network failure.
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
    print("  -> Submitting job strictly to Jülich QLMaaS...")
            
    max_retries = 5  # Increased to 5 since we have no local fallback
    for attempt in range(max_retries):
        captured_output = io.StringIO()
        original_timeout = socket.getdefaulttimeout()
        
        try:
            # 10 seconds timeout to avoid hanging, but enough for the server to reply
            socket.setdefaulttimeout(10.0)
            
            # STRICTLY REMOTE: No IsingAQPU local fallback!
            from qlmaas.qpus import AnalogQPU
            fresh_qpu = AnalogQPU()
            
            socket.setdefaulttimeout(original_timeout)
            
            with redirect_stdout(captured_output), redirect_stderr(captured_output):
                async_job = fresh_qpu.submit(job)
            
            output_str = captured_output.getvalue()
            if output_str and "Submitted a new batch" not in output_str:
                print(output_str, end="")
                
            raw_id = getattr(async_job, "batch_id", None) or getattr(async_job, "job_id", None)
            if raw_id is not None:
                id_str = str(raw_id)
                return f"SJob{id_str}" if id_str.isdigit() else id_str
            
            return "UNKNOWN_JOB_ID"
            
        except Exception as exception_error:
            # Reset timeout immediately
            socket.setdefaulttimeout(original_timeout)
            output_str = captured_output.getvalue()
            
            # Extreme fallback: if it printed the ID right before timing out
            match = re.search(r'(SJob\d+)', output_str)
            if match:
                recovered_job_id = match.group(1)
                print(f"     -> [RECOVERED] Timeout avoided! Intercepted assigned ID: {recovered_job_id}")
                return recovered_job_id
                
            print(f"     [!] Jülich server unreachable. No local fallback allowed. Reconnecting... (Attempt {attempt+1}/{max_retries})")
            
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait 5 seconds before hammering the server again
            else:
                print("     [FATAL] Maximum retries reached. Submission failed.")
                return "SUBMISSION_ERROR"

# ==========================================
# 7. ORCHESTRATOR RUN
# ==========================================
if __name__ == "__main__":
    print(f"\n--- COMPILING DISTRIBUTED CASE: {TARGET_FILE} ---")
    
    q_global_matrix = load_matrix(TARGET_FILE)
    if q_global_matrix is None:
        exit()

    clusters = partition_qubo_topological(q_global_matrix, target_cluster_size=TARGET_CLUSTER_SIZE)
    
    job_registry = {} 
    sa_times_log = [] 
    
    for cluster_id, global_nodes in clusters.items():
        print(f"\n--- Processing Cluster {cluster_id + 1}/{len(clusters)} ({len(global_nodes)} nodes) ---")
        
        sub_q_matrix = q_global_matrix[np.ix_(global_nodes, global_nodes)]
        
        if len(global_nodes) < 5: 
            print("  -> Cluster too small. Solving exact classically...")
            best_bits = solve_classically_exact(sub_q_matrix)
            bit_string = "".join(map(str, best_bits))
            
            fake_job_id = f"LOCAL_EXACT_{bit_string}"
            job_registry[fake_job_id] = global_nodes
            
            # Formatting for exact mini-clusters (null values for SA)
            sa_times_log.append({
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Instance": os.path.basename(TARGET_FILE),
                "Cluster_ID": cluster_id + 1,
                "Nodes": len(global_nodes),
                "SA_Restarts": 0,
                "SA_T_Init": 0.0,
                "SA_T_Min": 0.0,
                "SA_Cooling": 0.0,
                "SA_Steps": 0,
                "Best_Energy_Cost": 0.0,
                "Spatial_Fitness": "EXACT",
                "Scale_Factor": 0.0,
                "Classical_Time_s": 0.0,
                "Job_ID": fake_job_id,
                "Fitness_Metric": "EXACT_SOLVER" 
            })
            continue
            
        start_time = time.time()
        
        # 1. FETCH DYNAMIC PARAMETERS FOR THIS SPECIFIC CLUSTER
        current_sa_params = get_sa_hyperparameters(len(global_nodes))
        
        # 2. PASS THE DICTIONARY TO THE SA ENGINE
        coords, best_energy, fitness, scale_factor = optimize_embedding(sub_q_matrix, current_sa_params) 
        sa_execution_time = time.time() - start_time
        
        job_id = run_quantum_job(sub_q_matrix, coords, scale_factor)
        job_registry[job_id] = global_nodes
        
        print(f"  -> Cluster {cluster_id + 1} submitted. Job ID: {job_id}. SA Time: {sa_execution_time:.2f}s | Cost: {best_energy:.4f}")
        
        # 3. LOG THE ACTUAL DYNAMIC PARAMETERS USED
        sa_times_log.append({
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Instance": os.path.basename(TARGET_FILE),
            "Cluster_ID": cluster_id + 1,
            "Nodes": len(global_nodes),
            "SA_Restarts": current_sa_params["restarts"],
            "SA_T_Init": current_sa_params["t_init"],
            "SA_T_Min": current_sa_params["t_min"],
            "SA_Cooling": current_sa_params["cooling"],
            "SA_Steps": current_sa_params["steps"],
            "Best_Energy_Cost": round(best_energy, 4),
            "Spatial_Fitness": round(fitness, 6),
            "Scale_Factor": round(scale_factor, 4),
            "Classical_Time_s": round(sa_execution_time, 2),
            "Job_ID": job_id,
            "Fitness_Metric": "Improved_Topological_MAE_SA_GRASP" 
        })

    print("\n" + "="*50)
    print(" PHASE 1 COMPLETED: All clusters are currently executing.")

    instance_name = os.path.splitext(os.path.basename(TARGET_FILE))[0]
    
    JSON_FILE = f"./distributed_json/cluster_jobs_registry_{instance_name}.json"
    safe_registry = {j_id: [int(n) for n in nodes] for j_id, nodes in job_registry.items()}
    os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True) 
    
    with open(JSON_FILE, "w") as f:
        json.dump(safe_registry, f)
    print(f" [JSON] Job registry successfully saved in {JSON_FILE}.")

    # --- EXECUTION TIMES CSV SAVING ---
    SA_CSV_FILE = f"./distributed_time_csv/sa_times_{instance_name}.csv"
    os.makedirs(os.path.dirname(SA_CSV_FILE), exist_ok=True) 
    
    if os.path.exists(SA_CSV_FILE):
        df_history = pd.read_csv(SA_CSV_FILE)
        df_new = pd.DataFrame(sa_times_log)
        df_updated = pd.concat([df_history, df_new], ignore_index=True)
    else:
        df_updated = pd.DataFrame(sa_times_log)
        
    df_updated.to_csv(SA_CSV_FILE, index=False)
    print(f" [CSV] Distributed SA execution metrics saved in '{SA_CSV_FILE}'.")
    print("="*50)