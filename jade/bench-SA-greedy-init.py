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

from pulser import InterpolatedWaveform, Pulse, Sequence, Register
from pulser_myqlm import IsingAQPU


# THIS IS THE SCRIPT ABLE TO SCALE UP TO 20X20 AND EMBED OUR INSTANCE WITH SUCCESS.
# Architecture: GRASP-Initialized Simulated Annealing with Zero-Temperature Quenching.

# ==========================================
# 1. HARDWARE SETUP (Fake Jade)
# ==========================================
# We attempt to load the 'Jade' device profile from Pulser.
# If unavailable, we generate a synthetic analog device matching Jade's specs,
# specifically overriding the maximum radial distance to 50 µm to simulate
# the actual laser field constraints of neutral atom QPUs.
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
MIN_DIST = device.min_atom_distance # The Rydberg blockade radius boundary
MAX_RADIUS = device.max_radial_distance if hasattr(device, 'max_radial_distance') else 50

# Target file for the 20x20 scaling test instance
TARGET_FILE = ".././my_QUBO_instances/scaling_tests/jade_udg/jade_udg_20x20_R12_s62.npz"

# --- CSV APPEND SAVING LOGIC ---
OUTPUT_CSV = f".././new_csv/experiment_registry_20x20_SA.csv"

# ==========================================
# QUBO PARSER (.npz)
# ==========================================
# Reads the sparse matrix representation from the .npz file and reconstructs 
# the full symmetric N x N adjacency matrix Q representing the QUBO problem.
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
# 3. SA ENGINE (GRASP + Quenching)
# ==========================================
def optimize_embedding(Q, num_restarts=10):
    N_ATOMS = len(Q)
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)

    # --- Scale Factor Computation ---
    # Maps abstract QUBO weights to physical frequencies (MHz) constraint-checked against:
    # 1. scale_space: The maximum possible Van der Waals interaction based on the Rydberg blockade.
    # 2. scale_laser: The maximum possible global detuning provided by the device channel.
    V_max_allowed = device.interaction_coeff / (MIN_DIST**6)
    Q_max_off_diag = np.max(Q_off_diag)
    scale_space = V_max_allowed / Q_max_off_diag if Q_max_off_diag > 0 else float('inf')

    channel = device.channels["rydberg_global"]
    max_detuning = channel.max_abs_detuning if channel.max_abs_detuning is not None else 40.0
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scale_laser = max_detuning / avg_linear_weight if avg_linear_weight > 0 else float('inf')

    # The final mapping factor is the strictest bottleneck between space and laser power.
    scale_factor = min(scale_space, scale_laser)
    Q_target = Q_off_diag * scale_factor

    # Full upper triangular matrix cache used for GRASP degree calculations
    Q_triu_full = Q_target.copy() 
    np.fill_diagonal(Q_triu_full, 0)

    # --- ENERGY FUNCTION (The Physical Judge) ---
    def calculate_energy(coords):
        # O(N^2) pairwise distance calculation
        distances = pdist(coords)
        # Apply the C6 / r^6 Rydberg physical potential decay
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        penalty = 0.0
        
        # 1. Hard Spatial Constraints (Wall collisions)
        # Massive penalties (100k) for violating the physical limits of the QPU
        if np.any(distances < MIN_DIST):
            penalty += np.sum(np.clip(MIN_DIST - distances, 0, None)) * 100000.0 
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            penalty += np.sum(np.clip(radii - MAX_RADIUS, 0, None)) * 100000.0

        # Cap physical values to avoid gradient explosions near MIN_DIST
        max_q = np.max(np.abs(Q_triu)) if np.max(np.abs(Q_triu)) > 0 else 1.0
        V_triu_clipped = np.clip(V_triu, 0, max_q * 5.0)

        # 2. Topological Mean Absolute Error
        active_bonds = np.abs(Q_triu) > 1e-5
        
        # Penalty for failure to match target QUBO weights (stretching edges)
        error_active = np.sum(np.abs(V_triu_clipped[active_bonds] - Q_triu[active_bonds]))
        # Penalty for crosstalk (atoms that should be independent but are physically too close)
        # Multiplied by 2.0 to aggressively enforce graph sparsification
        error_inactive = np.sum(V_triu_clipped[~active_bonds]) * 2.0 
        
        return error_active + error_inactive + penalty

    # --- PERTURBATION FUNCTION (Neighborhood Generator) ---
    def perturb_coordinates(coords, temp, max_temp):
        new_coords = coords.copy()
        move_type = random.random()
        
        # Search radius scales down dynamically as the system cools
        jitter_scale = (temp / max_temp) * (MAX_RADIUS / 2) + 0.5 

        if move_type < 0.6:
            # Jitter (60%): Micro-displacement of a single atom for local fine-tuning
            idx = random.randint(0, N_ATOMS - 1)
            new_coords[idx][0] += random.uniform(-jitter_scale, jitter_scale)
            new_coords[idx][1] += random.uniform(-jitter_scale, jitter_scale)
        elif move_type < 0.8:
            # Swap (20%): Exchange positions of two atoms to escape topological dead-ends
            idx1, idx2 = random.sample(range(N_ATOMS), 2)
            new_coords[idx1], new_coords[idx2] = new_coords[idx2].copy(), new_coords[idx1].copy()
        else:
            # Jump (20%): Complete random repositioning within the laser field 
            idx = random.randint(0, N_ATOMS - 1)
            r = random.uniform(0, MAX_RADIUS)
            theta = random.uniform(0, 2 * np.pi)
            new_coords[idx] = [r * np.cos(theta), r * np.sin(theta)]
            
        return new_coords

    # --- SIMULATED ANNEALING PARAMETRIZATION ---
    T_INIT = 50000.0       # High initial temp to easily overcome 100k penalties
    T_MIN = 0.1            # Deep freeze temperature
    COOLING_RATE = 0.99    # Extremely slow cooling schedule for dense layouts
    STEPS_PER_TEMP = 1000  # Markov Chain Monte Carlo (MCMC) steps per epoch

    best_overall_energy = float('inf')
    best_overall_coords = None

    print(f"  -> Starting SA with GRASP Init & Quenching ({num_restarts} runs)...")
    
    for run_idx in range(num_restarts):
        
        # --- 1. SMART INITIALIZATION (GRASP Placer) ---
        # Prevents deterministic clustering traps by pseudo-randomizing the central "Boss" node.
        current_coords = np.zeros((N_ATOMS, 2))
        
        # Calculate total connectivity (degree) for each node
        node_weights = np.sum(np.abs(Q_triu_full), axis=1) + np.sum(np.abs(Q_triu_full), axis=0)
        sorted_nodes = np.argsort(node_weights)[::-1].tolist()
        
        # GRASP: random choice among the top 3 most connected candidates
        # Ensures macro-diversity across the different restarts.
        top_candidates = sorted_nodes[:min(3, len(sorted_nodes))]
        boss_node = random.choice(top_candidates)
        
        # Remove the selected boss from the queue
        sorted_nodes.remove(boss_node)
        
        # Anchor the boss strictly at the coordinate origin (center of the laser)
        current_coords[boss_node] = [0.0, 0.0]
        placed_nodes = [boss_node]
        
        # Attempt to layout the remaining nodes sequentially
        for node in sorted_nodes:
            best_target = None
            max_bond = 0
            
            # Find the strongest topological link to an ALREADY placed node
            for p in placed_nodes:
                bond = abs(Q_target[min(node, p), max(node, p)])
                if bond > max_bond:
                    max_bond = bond
                    best_target = p
            
            if best_target is not None and max_bond > 1e-5:
                # If connected, place it in a radial orbit around its target
                # Enforces a safe radius strictly larger than MIN_DIST to avoid initial explosion
                angle = random.uniform(0, 2 * np.pi)
                r_orbit = MIN_DIST * random.uniform(1.2, 1.8) 
                current_coords[node] = [
                    current_coords[best_target][0] + r_orbit * np.cos(angle),
                    current_coords[best_target][1] + r_orbit * np.sin(angle)
                ]
            else:
                # If isolated (or 0-weight), banish it to the outer boundary of the laser field
                angle = random.uniform(0, 2 * np.pi)
                current_coords[node] = [
                    (MAX_RADIUS - MIN_DIST) * np.cos(angle),
                    (MAX_RADIUS - MIN_DIST) * np.sin(angle)
                ]
            placed_nodes.append(node)
        # --------------------------------------------

        current_energy = calculate_energy(current_coords)
        best_run_coords = current_coords.copy()
        best_run_energy = current_energy
        
        T = T_INIT
        
        # --- 2. COOLING PHASE (Simulated Annealing) ---
        while T > T_MIN:
            for step in range(STEPS_PER_TEMP):
                candidate_coords = perturb_coordinates(current_coords, T, T_INIT)
                candidate_energy = calculate_energy(candidate_coords)
                
                delta_e = candidate_energy - current_energy
                
                # Metropolis-Hastings Acceptance Criterion
                # Always accept improvements. Accept worsening moves with probability exp(-dE/T)
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    current_coords = candidate_coords
                    current_energy = candidate_energy
                    
                    # Track local optima specific to this single run
                    if current_energy < best_run_energy:
                        best_run_energy = current_energy
                        best_run_coords = current_coords.copy()
            
            # Geometric temperature decay
            T *= COOLING_RATE
            
        # --- 3. QUENCHING PHASE (Zero-Temperature Greedy Descent) ---
        # Acts as a final polish. Simulates an instantaneous drop to T=0.
        # Only strict improvements are accepted to iron out sub-micrometer topological noise.
        for _ in range(1500): 
            idx = random.randint(0, N_ATOMS - 1)
            cand_coords = best_run_coords.copy()
            
            # Ultra-fine jitter restricted to 0.3 µm max movement
            cand_coords[idx][0] += random.uniform(-0.3, 0.3) 
            cand_coords[idx][1] += random.uniform(-0.3, 0.3)
            
            cand_e = calculate_energy(cand_coords)
            
            # Strict monotonic improvement only
            if cand_e < best_run_energy:
                best_run_energy = cand_e
                best_run_coords = cand_coords.copy()
        # ---------------------------------------------------------
            
        print(f"    [Run {run_idx+1}] Best Energy (post-Quenching): {best_run_energy:.4f}")
        
        # Update the overall global record across all restarts
        if best_run_energy < best_overall_energy:
            best_overall_energy = best_run_energy
            best_overall_coords = best_run_coords.copy()
            print(f"      -> New Global Record! Energy: {best_overall_energy:.4f}")

    # Convert the absolute energy penalty back to a maximizing 'fitness' metric for the CSV logger
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
    
    # Calculate uniform global detuning for the adiabatic sweep
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scaled_delta = avg_linear_weight * scale_factor
    
    # Map the classical coordinates into the Pulser Quantum Register
    qubits = {f"q{i}": c for i, c in enumerate(coords)}
    reg = Register(qubits)
    
    # Render the topology visually (disabled for headless execution)
    reg.draw(blockade_radius=MIN_DIST, draw_half_radius=True, draw_graph=False)
    
    # Extract the optimal Rabi frequency (Omega) based on the target coupling
    ideal_omega = np.median(Q_target[Q_target > 0]) if np.any(Q_target > 0) else 1.0
    channel_max_amp = device.channels["rydberg_global"].max_amp
    Omega = min(ideal_omega, channel_max_amp / 1.2) if channel_max_amp else ideal_omega

    # Time parameters for the adiabatic evolution
    T = 4000 
    delta_0 = -scaled_delta
    delta_f = scaled_delta

    # Define the laser pulses (Amplitude and Detuning interpolation)
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )

    # Assemble the quantum sequence
    seq = Sequence(reg, device)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")

    # Convert to remote/local job executable
    job = IsingAQPU.convert_sequence_to_job(seq, nbshots=0)
    print("  -> Submitting job to quantum emulator...")
    
    # Remote execution wrapper with graceful degradation and 3 retry attempts
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
# 5. ORCHESTRATOR RUN
# ==========================================
if __name__ == "__main__":
    print("Initializing virtual QPU...")
    
    # Connection Cascade: 
    # 1. AnalogQPU (Primary Remote) -> 2. QLMaaSQPU (Secondary Remote) -> 3. IsingAQPU (Local Simulation)
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
            from pulser_myqlm import IsingAQPU
            qpu_emulator = IsingAQPU()

    print(f"\n--- COMPILING BASE CASE: {TARGET_FILE} ---")
    
    Q = load_matrix(TARGET_FILE)
    if Q is not None:
        # Time the classical embedding phase separately
        start_classic_time = time.time()
        coords, fitness, scale = optimize_embedding(Q, num_restarts=10)
        classic_time = time.time() - start_classic_time
        
        # Trigger the quantum emulation phase
        job_id = run_quantum_job(Q, coords, scale, qpu_emulator)
        
        print("\n" + "="*50)
        print(f" OPERATION COMPLETED")
        print(f" Your Job ID is: {job_id}")
        print(f" Use the 'retrieve.py' script to extract the data.")
        print("="*50)
        
        # Structure the metadata payload for CSV tracking
        run_info = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Instance": os.path.basename(TARGET_FILE),
            "N_Nodes": len(Q),
            "Spatial_Fitness": round(fitness, 6),
            "Scale_Factor": round(scale, 4),
            "Classical_Time_s": round(classic_time, 2),
            "Juelich_Job_ID": job_id,
            "Fitness_Metric": "Improved_Topological_MAE_SA_GRASP"
        }
        
        # CSV creation and append logic
        if os.path.exists(OUTPUT_CSV):
            df_history = pd.read_csv(OUTPUT_CSV)
            df_new = pd.DataFrame([run_info])
            df_updated = pd.concat([df_history, df_new], ignore_index=True)
        else:
            df_updated = pd.DataFrame([run_info])
            
        df_updated.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[INFO] Run data successfully saved in '{OUTPUT_CSV}'")