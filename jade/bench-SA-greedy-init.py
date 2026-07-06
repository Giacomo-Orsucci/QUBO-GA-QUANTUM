# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

import os
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt 
import dataclasses
import random
import math
import io
import re
from contextlib import redirect_stdout, redirect_stderr

from pulser import InterpolatedWaveform, Pulse, Sequence, Register
from pulser_myqlm import IsingAQPU


# THIS IS THE SCRIPT ABLE TO SCALE UP TO 20X20 AND EMBED OUR INSTANCE WITH SUCCESS.

# Architecture: GRASP-Initialized Simulated Annealing with Zero-Temperature Quenching.

# IT REALLY DEPENDS ON THE PARAMETRIZATION, BUT FOR THE MOMENT IS THE BEST VERSION.

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

# Target file for the 20x20 scaling test instance
TARGET_FILE = ".././my_QUBO_instances/scaling_tests/jade_udg/jade_udg_20x20_R12_s62_d5.0.npz"

instance_name = os.path.splitext(os.path.basename(TARGET_FILE))[0]

# --- CSV APPEND SAVING LOGIC ---
OUTPUT_CSV = f".././new_csv/experiment_registry_SA_{instance_name}.csv"

"""
# ==========================================
# 2. SIMULATED ANNEALING HYPERPARAMETERS
# ==========================================
num_restarts = 15      # Use 15 for 20x20, 10 is enough for 15x15
T_INIT = 50000.0       # High initial temp to easily overcome 100k penalties
T_MIN = 0.1            # Deep freeze temperature
COOLING_RATE = 0.99    # Use 0.99 for 20x20, 0.98 for 15x15
STEPS_PER_TEMP = 2000  # Use 2000 for 20x20, 1000 for 15x15
"""

# ==========================================
# 2. SIMULATED ANNEALING HYPERPARAMETERS (Deep-Focus Tuning)
# ==========================================
num_restarts = 5       # REDUCED FROM 15 TO 5: Saves a huge amount of overall time.
T_INIT = 80.0          # Starting less hot to avoid destroying the good GRASP positions.
T_MIN = 0.01           
COOLING_RATE = 0.995   # CRUCIAL: Extremely slow cooling. Gives atoms time to gently "relax".
STEPS_PER_TEMP = 1200  # Balanced. With cooling at 0.995, we'll have many more thermal "steps".

# ==========================================
# 3. QUBO PARSER (.npz)
# ==========================================
def load_matrix(file_path):
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
        print(f"  [READ ERROR] Unable to load {file_path}: {e}")
        return None

# ==========================================
# 4. SA ENGINE (GRASP + Quenching)
# ==========================================
def optimize_embedding(Q, num_restarts=10):
    N_ATOMS = len(Q)
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)

    # --- Scale Factor Computation ---
    V_max_allowed = device.interaction_coeff / (MIN_DIST**6)
    Q_max_off_diag = np.max(Q_off_diag)
    scale_space = V_max_allowed / Q_max_off_diag if Q_max_off_diag > 0 else float('inf')

    channel = device.channels["rydberg_global"]
    max_detuning = channel.max_abs_detuning if channel.max_abs_detuning is not None else 40.0
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scale_laser = max_detuning / avg_linear_weight if avg_linear_weight > 0 else float('inf')

    scale_factor = min(scale_space, scale_laser)
    Q_target = Q_off_diag * scale_factor

    Q_triu_full = Q_target.copy() 
    np.fill_diagonal(Q_triu_full, 0)

    # --- ENERGY FUNCTION ---
    def calculate_energy(coords):
        distances = pdist(coords)
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        penalty = 0.0
        
        if np.any(distances < MIN_DIST):
            penalty += np.sum(np.clip(MIN_DIST - distances, 0, None)) * 100000.0 
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            penalty += np.sum(np.clip(radii - MAX_RADIUS, 0, None)) * 100000.0

        max_q = np.max(np.abs(Q_triu)) if np.max(np.abs(Q_triu)) > 0 else 1.0
        V_triu_clipped = np.clip(V_triu, 0, max_q * 5.0)

        active_bonds = np.abs(Q_triu) > 1e-5
        
        error_active = np.sum(np.abs(V_triu_clipped[active_bonds] - Q_triu[active_bonds]))
        error_inactive = np.sum(V_triu_clipped[~active_bonds]) * 2.0 
        
        return error_active + error_inactive + penalty

    # --- PERTURBATION FUNCTION ---
    #def perturb_coordinates(coords, temp, max_temp):
    #    new_coords = coords.copy()
     #   move_type = random.random()
        """
        jitter_scale = (temp / max_temp) * (MAX_RADIUS / 2) + 0.5 

        if move_type < 0.6:
            idx = random.randint(0, N_ATOMS - 1)
            new_coords[idx][0] += random.uniform(-jitter_scale, jitter_scale)
            new_coords[idx][1] += random.uniform(-jitter_scale, jitter_scale)
        elif move_type < 0.8:
            idx1, idx2 = random.sample(range(N_ATOMS), 2)
            new_coords[idx1], new_coords[idx2] = new_coords[idx2].copy(), new_coords[idx1].copy()
        else:
            idx = random.randint(0, N_ATOMS - 1)
            r = random.uniform(0, MAX_RADIUS)
            theta = random.uniform(0, 2 * np.pi)
            new_coords[idx] = [r * np.cos(theta), r * np.sin(theta)]
            
        return new_coords

"""

# alternative emerged from experimenting dense problems:

# --- PERTURBATION FUNCTION ---
    def perturb_coordinates(coords, temp, max_temp):
        new_coords = coords.copy()
        move_type = random.random()
        
        # little Jitter: max 3 µm to not destroy clusters
        jitter_scale = (temp / max_temp) * 3.0 + 0.2 

        if move_type < 0.3:
            # 30% fine-tuning, micro adjustments
            idx = random.randint(0, N_ATOMS - 1)
            new_coords[idx][0] += random.uniform(-jitter_scale, jitter_scale)
            new_coords[idx][1] += random.uniform(-jitter_scale, jitter_scale)
            
        elif move_type < 0.9:
            # 60% SWAP
            idx1, idx2 = random.sample(range(N_ATOMS), 2)
            new_coords[idx1], new_coords[idx2] = new_coords[idx2].copy(), new_coords[idx1].copy()
            
        else:
            # 10% cautious repositioning
            idx = random.randint(0, N_ATOMS - 1)
            r = random.uniform(0, MAX_RADIUS * 0.8) 
            theta = random.uniform(0, 2 * np.pi)
            new_coords[idx] = [r * np.cos(theta), r * np.sin(theta)]
            
        return new_coords


    best_overall_energy = float('inf')
    best_overall_coords = None

    print(f"  -> Starting SA with GRASP Init & Quenching ({num_restarts} runs)...")
    
    for run_idx in range(num_restarts):
        
        # --- SMART INITIALIZATION (GRASP Placer) ---
        current_coords = np.zeros((N_ATOMS, 2))
        node_weights = np.sum(np.abs(Q_triu_full), axis=1) + np.sum(np.abs(Q_triu_full), axis=0)
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
                bond = abs(Q_target[min(node, p), max(node, p)])
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
        T = T_INIT
        
        # --- COOLING PHASE ---
        while T > T_MIN:
            for step in range(STEPS_PER_TEMP):
                candidate_coords = perturb_coordinates(current_coords, T, T_INIT)
                candidate_energy = calculate_energy(candidate_coords)
                
                delta_e = candidate_energy - current_energy
                
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    current_coords = candidate_coords
                    current_energy = candidate_energy
                    
                    if current_energy < best_run_energy:
                        best_run_energy = current_energy
                        best_run_coords = current_coords.copy()
            
            T *= COOLING_RATE
            
        # --- QUENCHING PHASE ---
        for _ in range(1500): 
            idx = random.randint(0, N_ATOMS - 1)
            cand_coords = best_run_coords.copy()
            
            cand_coords[idx][0] += random.uniform(-0.3, 0.3) 
            cand_coords[idx][1] += random.uniform(-0.3, 0.3)
            
            cand_e = calculate_energy(cand_coords)
            
            if cand_e < best_run_energy:
                best_run_energy = cand_e
                best_run_coords = cand_coords.copy()
            
        print(f"    [Run {run_idx+1}] Best Energy (post-Quenching): {best_run_energy:.4f}")
        
        if best_run_energy < best_overall_energy:
            best_overall_energy = best_run_energy
            best_overall_coords = best_run_coords.copy()
            print(f"      -> New Global Record! Energy: {best_overall_energy:.4f}")

    final_fitness = 1.0 / (best_overall_energy + 1e-6)

    return best_overall_coords, best_overall_energy, final_fitness, scale_factor

# ==========================================
# 5. ADIABATIC QUANTUM ENGINE
# ==========================================
def run_quantum_job(Q, coords, scale_factor):
    N_ATOMS = len(Q)
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)
    Q_target = Q_off_diag * scale_factor
    
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scaled_delta = avg_linear_weight * scale_factor
    
    qubits = {f"q{i}": c for i, c in enumerate(coords)}
    reg = Register(qubits)
    
    reg.draw(blockade_radius=MIN_DIST, draw_half_radius=True, draw_graph=False)
    
    ideal_omega = np.median(Q_target[Q_target > 0]) if np.any(Q_target > 0) else 1.0
    channel_max_amp = device.channels["rydberg_global"].max_amp
    Omega = min(ideal_omega, channel_max_amp / 1.2) if channel_max_amp else ideal_omega

    T = 4000 
    delta_0 = -scaled_delta
    delta_f = scaled_delta

    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )

    seq = Sequence(reg, device)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")

    job = IsingAQPU.convert_sequence_to_job(seq, nbshots=0)
    print("  -> Submitting job to quantum emulator...")


    print("Initializing virtual QPU...")
    
    try:
        from qlmaas.qpus import AnalogQPU
        qpu_emulator = AnalogQPU()
        print("  -> Successfully connected to QLMaaS server (AnalogQPU).")
    except Exception as e_primary:
        print(f"[WARNING] Primary connection failed: {e_primary}")
        try:
            from qat.qlmaas.qpus import QLMaaSQPU
            qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
            print("  -> Successfully connected to QLMaaS server (QLMaaSQPU).")
        except Exception as e_secondary:
            print(f"[WARNING] Secondary connection failed: {e_secondary}")
            print("[INFO] Falling back to local emulator (IsingAQPU)...")
            
            qpu_emulator = IsingAQPU()
    
    # --- ASYNCHRONOUS MODIFICATION WITH SAFE JIT CONNECTION & STDOUT INTERCEPTOR ---
    max_retries = 3
    for attempt in range(max_retries):
        captured_output = io.StringIO()
        
        try:
            with redirect_stdout(captured_output), redirect_stderr(captured_output):
                
                async_job = qpu_emulator.submit(job)
            
            output_str = captured_output.getvalue()
            
            if output_str and "Submitted a new batch" not in output_str:
                print(output_str, end="")
                
            raw_id = getattr(async_job, "batch_id", None)
            if not raw_id:
                raw_id = getattr(async_job, "job_id", None)
                
            if raw_id is not None:
                id_str = str(raw_id)
                if id_str.isdigit():
                    return f"SJob{id_str}"
                return id_str
            
            return "UNKNOWN_JOB_ID"
            
        except Exception as exception_error:
            output_str = captured_output.getvalue()
            
            
            match = re.search(r'(SJob\d+)', output_str)
            if match:
                recovered_job_id = match.group(1)
                print(f"     -> [RECOVERED] Server timeout avoided! Intercepted assigned ID: {recovered_job_id}")
                return recovered_job_id
                
            
            if output_str:
                print(output_str, end="")
                
            print(f"     [!] Network timeout/drop detected. Reconnecting... (Attempt {attempt+1}) - Error: {exception_error}")
            
            if attempt < max_retries - 1:
                time.sleep(3) 
            else:
                return "SUBMISSION_ERROR"

# ==========================================
# 6. ORCHESTRATOR RUN (FAIL-SAFE CHECKPOINTING)
# ==========================================
if __name__ == "__main__":
    
    print(f"\n--- COMPILING BASE CASE: {TARGET_FILE} ---")
    
    Q = load_matrix(TARGET_FILE)
    if Q is not None:
        
        # --- PHASE 1: CLASSICAL COMPUTATION ---
        start_classic_time = time.time()
        coords, best_energy, fitness, scale = optimize_embedding(Q, num_restarts=num_restarts)
        classic_time = time.time() - start_classic_time
        
        # --- PHASE 2: FAIL-SAFE PRE-SAVE ---
        run_info = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Instance": os.path.basename(TARGET_FILE),
            "N_Nodes": len(Q),
            "SA_Restarts": num_restarts,           # NEW LOGGED FIELD
            "SA_T_Init": T_INIT,                   # NEW LOGGED FIELD
            "SA_T_Min": T_MIN,                     # NEW LOGGED FIELD
            "SA_Cooling": COOLING_RATE,            # NEW LOGGED FIELD
            "SA_Steps": STEPS_PER_TEMP,            # NEW LOGGED FIELD
            "Best_Energy_Cost": round(best_energy, 4), 
            "Spatial_Fitness": round(fitness, 6),
            "Scale_Factor": round(scale, 4),
            "Classical_Time_s": round(classic_time, 2),
            "Juelich_Job_ID": "PENDING",  
            "Fitness_Metric": "Improved_Topological_MAE_SA_GRASP"
        }
        
        if os.path.exists(OUTPUT_CSV):
            df_history = pd.read_csv(OUTPUT_CSV)
            df_new = pd.DataFrame([run_info])
            df_updated = pd.concat([df_history, df_new], ignore_index=True)
        else:
            df_updated = pd.DataFrame([run_info])
            
        df_updated.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[FAIL-SAFE] Classical optimization data (Cost: {best_energy:.4f}) safely stored in '{OUTPUT_CSV}'.")
        
        # --- PHASE 3: REMOTE QUANTUM SUBMISSION ---
        job_id = run_quantum_job(Q, coords, scale)
        
        print("\n" + "="*50)
        print(f" OPERATION COMPLETED")
        print(f" Your Job ID is: {job_id}")
        print(f" Use the 'retrieve.py' script to extract the data.")
        print("="*50)
        
        # --- PHASE 4: POST-SAVE UPDATE ---
        if job_id != "SUBMISSION_ERROR":
            df_history = pd.read_csv(OUTPUT_CSV)
            df_history.at[df_history.index[-1], 'Juelich_Job_ID'] = job_id
            df_history.to_csv(OUTPUT_CSV, index=False)
            print(f"[INFO] Registry seamlessly updated. Job ID '{job_id}' linked to the run.")
        else:
            print(f"[WARNING] Submission failed. Data is preserved but Job ID remains 'PENDING'.")