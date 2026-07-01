# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

import os
import time
import numpy as np
import pandas as pd
import pygad
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt 
import dataclasses
import random
import math

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

# Insert the QUBO problem you are interested in embedding and executing on AnalogQPU.

# TARGET_FILE ="./my_QUBO_instances/scaling_tests/friendly/global_friendly_5x5_d50_s100.npz"
# TARGET_FILE ="./my_QUBO_instances/tutorial_5x5.npz"
# TARGET_FILE ="./my_QUBO_instances/scaling_tests/friendly/global_friendly_6x6_d46.7_s100.npz"
TARGET_FILE = ".././my_QUBO_instances/scaling_tests/jade_udg/jade_udg_20x20_R12_s62.npz"

# --- CSV APPEND SAVING LOGIC ---
OUTPUT_CSV = f".././new_csv/experiment_registry_20x20_SA.csv"

# ==========================================
# 2. QUBO PARSER (.npz)
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
# 3. SA ENGINE (Embedding Optimization)
# ==========================================
def optimize_embedding(Q, num_restarts=10):
    N_ATOMS = len(Q)
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)

    # --- Scale Factor Computation (Identical to original) ---
    V_max_allowed = device.interaction_coeff / (MIN_DIST**6)
    Q_max_off_diag = np.max(Q_off_diag)
    scale_space = V_max_allowed / Q_max_off_diag if Q_max_off_diag > 0 else float('inf')

    channel = device.channels["rydberg_global"]
    max_detuning = channel.max_abs_detuning if channel.max_abs_detuning is not None else 40.0
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scale_laser = max_detuning / avg_linear_weight if avg_linear_weight > 0 else float('inf')

    scale_factor = min(scale_space, scale_laser)
    Q_target = Q_off_diag * scale_factor

    # --- ENERGY FUNCTION (Former Fitness Improved Topological MAE) ---
    def calculate_energy(coords):
        distances = pdist(coords)
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        penalty = 0.0

        # Hard spatial constraints
        if np.any(distances < MIN_DIST):
            penalty += np.sum(np.clip(MIN_DIST - distances, 0, None)) * 100000.0 
            
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            penalty += np.sum(np.clip(radii - MAX_RADIUS, 0, None)) * 100000.0

        # Difference 1: Clipping
        max_q = np.max(np.abs(Q_triu)) if np.max(np.abs(Q_triu)) > 0 else 1.0
        V_triu_clipped = np.clip(V_triu, 0, max_q * 5.0)

        # Difference 2: Crosstalk vs Active Bonds
        active_bonds = np.abs(Q_triu) > 1e-5
        
        error_active = np.sum(np.abs(V_triu_clipped[active_bonds] - Q_triu[active_bonds]))
        error_inactive = np.sum(V_triu_clipped[~active_bonds]) * 2.0 
        
        # In SA we minimize the error, we do not calculate the inverse fitness
        total_energy = error_active + error_inactive + penalty
        return total_energy

    # --- PERTURBATION FUNCTION (The "Moves") ---
    def perturb_coordinates(coords, temp, max_temp):
        new_coords = coords.copy()
        move_type = random.random()
        
        # Movement amplitude reduces with temperature
        jitter_scale = (temp / max_temp) * (MAX_RADIUS / 2) + 0.5 

        if move_type < 0.6:
            # 1. JITTER (60%): Micro-displacement of an atom (Refinement)
            idx = random.randint(0, N_ATOMS - 1)
            new_coords[idx][0] += random.uniform(-jitter_scale, jitter_scale)
            new_coords[idx][1] += random.uniform(-jitter_scale, jitter_scale)
            
        elif move_type < 0.8:
            # 2. SWAP (20%): Swap two atoms' positions (Topology)
            idx1, idx2 = random.sample(range(N_ATOMS), 2)
            new_coords[idx1], new_coords[idx2] = new_coords[idx2].copy(), new_coords[idx1].copy()
            
        else:
            # 3. JUMP (20%): Completely reposition an atom (Exploration)
            idx = random.randint(0, N_ATOMS - 1)
            r = random.uniform(0, MAX_RADIUS)
            theta = random.uniform(0, 2 * np.pi)
            new_coords[idx] = [r * np.cos(theta), r * np.sin(theta)]

        return new_coords

    # --- SIMULATED ANNEALING PARAMETERS ---
    # These are the equivalents of (population, generations, mutation)
    T_INIT = 50000.0       # Initial temperature (high to overcome strong local penalties)
    T_MIN = 0.1            # Final temperature
    COOLING_RATE = 0.98    # Cooling factor (e.g., 0.99 = slow, 0.90 = fast)
    STEPS_PER_TEMP = 800   # How many configurations to explore per temperature level

    best_overall_energy = float('inf')
    best_overall_coords = None

    print(f"  -> Starting Simulated Annealing ({num_restarts} runs)...")
    
    for run_idx in range(num_restarts):
        # 1. Random initialization (within the maximum circle)
        current_coords = np.zeros((N_ATOMS, 2))
        for i in range(N_ATOMS):
            r = random.uniform(0, MAX_RADIUS)
            theta = random.uniform(0, 2 * np.pi)
            current_coords[i] = [r * np.cos(theta), r * np.sin(theta)]
            
        current_energy = calculate_energy(current_coords)
        
        # Tracking variables for the single run
        best_run_coords = current_coords.copy()
        best_run_energy = current_energy
        
        T = T_INIT
        
        # 2. Cooling Cycle
        while T > T_MIN:
            for step in range(STEPS_PER_TEMP):
                # Generate a neighbor
                candidate_coords = perturb_coordinates(current_coords, T, T_INIT)
                candidate_energy = calculate_energy(candidate_coords)
                
                # Calculate delta Energy
                delta_e = candidate_energy - current_energy
                
                # Metropolis Acceptance Criterion
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    current_coords = candidate_coords
                    current_energy = candidate_energy
                    
                    # Update the best result found so far
                    if current_energy < best_run_energy:
                        best_run_energy = current_energy
                        best_run_coords = current_coords.copy()
            
            # Lower the temperature
            T *= COOLING_RATE
            
        print(f"    [Run {run_idx+1}] Best Energy: {best_run_energy:.4f}")
        
        # Update the global record
        if best_run_energy < best_overall_energy:
            best_overall_energy = best_run_energy
            best_overall_coords = best_run_coords.copy()
            print(f"      -> New Global Record! Energy: {best_overall_energy:.4f}")

    # We convert the Energy (which is the error) back into "Fitness"
    # so as not to break your CSV logging which expects fitness
    final_fitness = 1.0 / (best_overall_energy + 1e-6)

    return best_overall_coords, final_fitness, scale_factor

# ==========================================
# 4. ADIABATIC QUANTUM ENGINE
# ==========================================
def run_quantum_job(Q, coords, scale_factor, qpu_emulator):
    N_ATOMS = len(Q)
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)
    Q_target = Q_off_diag * scale_factor
    
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scaled_delta = avg_linear_weight * scale_factor
    
    qubits = {f"q{i}": c for i, c in enumerate(coords)}
    reg = Register(qubits)
    
    # Visual inspection of the layout
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
    
    # --- ASYNCHRONOUS MODIFICATION ---
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            async_job = qpu_emulator.submit(job)
            try:
                job_id = async_job.batch_id
            except AttributeError:
                job_id = str(async_job) 
                
            print(f"  -> [SUCCESS] Job accepted! Assigned ID: {job_id}")
            return job_id
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                return "SUBMISSION_ERROR"

# ==========================================
# 5. ORCHESTRATOR RUN (Async with CSV logging)
# ==========================================
if __name__ == "__main__":
    print("Initializing virtual QPU...")
    try:
        from qlmaas.qpus import AnalogQPU
        qpu_emulator = AnalogQPU()
        print("  -> Successfully connected to QLMaaS server (AnalogQPU).")
    except ImportError:
        try:
            from qat.qlmaas.qpus import QLMaaSQPU
            qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
            print("  -> Successfully connected to QLMaaS server (QLMaaSQPU).")
        except ImportError:
            print("[WARNING] Remote server not found, falling back to local emulator (IsingAQPU)...")
            from pulser_myqlm import IsingAQPU
            qpu_emulator = IsingAQPU()


    print(f"\n--- COMPILING BASE CASE: {TARGET_FILE} ---")
    
    Q = load_matrix(TARGET_FILE)
    if Q is not None:
        # Calculate classical phase time for CSV
        start_classic_time = time.time()
        coords, fitness, scale = optimize_embedding(Q, num_restarts=10) # base = 5, boost=10-15
        classic_time = time.time() - start_classic_time
        
        # Asynchronous Execution
        job_id = run_quantum_job(Q, coords, scale, qpu_emulator)
        
        print("\n" + "="*50)
        print(f" OPERATION COMPLETED")
        print(f" Your Job ID is: {job_id}")
        print(f" Use the 'retrieve.py' script to extract the data.")
        print("="*50)
        
        # Build the dictionary with current metadata
        run_info = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Instance": os.path.basename(TARGET_FILE),
            "N_Nodes": len(Q),
            "Spatial_Fitness": round(fitness, 6),
            "Scale_Factor": round(scale, 4),
            "Classical_Time_s": round(classic_time, 2),
            "Juelich_Job_ID": job_id,
            "Fitness_Metric": "Improved_Topological_MAE"  # Change to "MSE" when testing the other metric
        }
        
        # If the file already exists, load the history and append the new row.
        # If it doesn't exist, Pandas will create it from scratch with headers.
        if os.path.exists(OUTPUT_CSV):
            df_history = pd.read_csv(OUTPUT_CSV)
            df_new = pd.DataFrame([run_info])
            df_updated = pd.concat([df_history, df_new], ignore_index=True)
        else:
            df_updated = pd.DataFrame([run_info])
            
        df_updated.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[INFO] Run data successfully saved in '{OUTPUT_CSV}'")