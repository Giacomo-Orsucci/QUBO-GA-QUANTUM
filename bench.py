import os
import time
import numpy as np
import pandas as pd
import pygad
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt 
import dataclasses

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

# ==========================================
# 2. NATIVE PARSER FOR HAMBURG MATRICES (.npz)
# ==========================================
def load_hamburg_matrix(file_path):
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
# 3. GENETIC ENGINE (Embedding Optimization)
# ==========================================
def optimize_embedding(Q, num_restarts=10):
    N_ATOMS = len(Q)
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)

    # Physical Scale Factor Calculation
    V_max_allowed = device.interaction_coeff / (MIN_DIST**6)
    Q_max_off_diag = np.max(Q_off_diag)
    scale_space = V_max_allowed / Q_max_off_diag if Q_max_off_diag > 0 else float('inf')

    channel = device.channels["rydberg_global"]
    max_detuning = channel.max_abs_detuning if channel.max_abs_detuning is not None else 40.0
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scale_laser = max_detuning / avg_linear_weight if avg_linear_weight > 0 else float('inf')

    scale_factor = min(scale_space, scale_laser)
    Q_target = Q_off_diag * scale_factor

    # --- FITNESS: PURE MAE (Unweighted) ---
    def pure_mae_fitness_func(ga_instance, solution, solution_idx):
        coords = np.reshape(solution, (N_ATOMS, 2))
        distances = pdist(coords)
        
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        # Pure Mean Absolute Error (no bond_importance multiplier)
        base_error = np.mean(np.abs(V_triu - Q_triu))
        penalty = 0.0

        if np.any(distances < MIN_DIST):
            penalty += np.sum(np.clip(MIN_DIST - distances, 0, None)) * 100000.0 
            
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            penalty += np.sum(np.clip(radii - MAX_RADIUS, 0, None)) * 100000.0
            
        total_error = base_error + penalty
        return 1.0 / (total_error + 1e-6)
    
    # --- FITNESS: TOPOLOGICAL MAE (The Best One) ---
    def topological_mae_fitness_func(ga_instance, solution, solution_idx):
        coords = np.reshape(solution, (N_ATOMS, 2))
        distances = pdist(coords)
        
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        bond_importance = np.abs(Q_triu)
        if np.max(bond_importance) > 0:
            bond_importance = bond_importance / np.max(bond_importance)
        
        base_error = np.sum(bond_importance * np.abs(V_triu - Q_triu))
        penalty = 0.0

        if np.any(distances < MIN_DIST):
            penalty += np.sum(np.clip(MIN_DIST - distances, 0, None)) * 100000.0 
            
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            penalty += np.sum(np.clip(radii - MAX_RADIUS, 0, None)) * 100000.0
            
        return 1.0 / (base_error + penalty + 1e-6)
    

    # --- FITNESS: STANDARD MSE ---
    def pure_mse_fitness_func(ga_instance, solution, solution_idx):
        coords = np.reshape(solution, (N_ATOMS, 2))
        distances = pdist(coords)
        
        # 1. Calculate physical potential
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        
        # 2. Extract upper triangles
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        # 3. Calculate Mean Squared Error (MSE)
        # Replacing weighted MAE with standard MSE
        base_error = np.mean((V_triu - Q_triu) ** 2)
        
        # 4. Hardware limit penalties (Unchanged)
        penalty = 0.0
        if np.any(distances < MIN_DIST):
            penalty += np.sum(np.clip(MIN_DIST - distances, 0, None)) * 100000.0 
            
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            penalty += np.sum(np.clip(radii - MAX_RADIUS, 0, None)) * 100000.0
            
        total_error = base_error + penalty
        return 1.0 / (total_error + 1e-6)
    
    # --- FITNESS: TOPOLOGICAL MSE ---
    def topological_mse_fitness_func(ga_instance, solution, solution_idx):
        coords = np.reshape(solution, (N_ATOMS, 2))
        distances = pdist(coords)
        
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        bond_importance = np.abs(Q_triu)
        if np.max(bond_importance) > 0:
            bond_importance = bond_importance / np.max(bond_importance)
        
        # Topological MSE: Weighted Squared Error
        base_error = np.sum(bond_importance * ((V_triu - Q_triu) ** 2))
        penalty = 0.0

        if np.any(distances < MIN_DIST):
            penalty += np.sum(np.clip(MIN_DIST - distances, 0, None)) * 100000.0 
            
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            penalty += np.sum(np.clip(radii - MAX_RADIUS, 0, None)) * 100000.0
            
        return 1.0 / (base_error + penalty + 1e-6)

    gene_space = [{'low': -MAX_RADIUS, 'high': MAX_RADIUS} for _ in range(N_ATOMS * 2)]
    best_overall_fitness = -float('inf')
    best_overall_coords = None

    print(f"  -> Starting Multi-Start ({num_restarts} runs)...")
    for run_idx in range(num_restarts):
        ga = pygad.GA(
            num_generations=800,
            num_parents_mating=20,
            fitness_func=topological_mae_fitness_func, # Switch to mse_fitness_func to test MSE
            sol_per_pop=100,
            num_genes=N_ATOMS * 2,
            gene_space=gene_space,
            parent_selection_type="tournament",
            K_tournament=3,
            keep_elitism=5,
            crossover_type="uniform",
            mutation_type="adaptive",
            mutation_probability=[0.4, 0.05],
            random_mutation_min_val=-3.0, 
            random_mutation_max_val=3.0,
            allow_duplicate_genes=False,
            suppress_warnings=True
        )
        ga.run()
        run_best_solution, run_best_fitness, _ = ga.best_solution()
        
        if run_best_fitness > best_overall_fitness:
            best_overall_fitness = run_best_fitness
            best_overall_coords = np.reshape(run_best_solution, (N_ATOMS, 2))
            print(f"    [Run {run_idx+1}] New Record! Fitness: {run_best_fitness:.4f}")
            
    return best_overall_coords, best_overall_fitness, scale_factor

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
    except ImportError:
        try:
            from qat.qlmaas.qpus import QLMaaSQPU
            qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
        except ImportError:
            print("[WARNING] Remote server not found, using local emulator...")
            from pulser_myqlm import IsingAQPU
            qpu_emulator = IsingAQPU()

    TARGET_FILE ="./my_QUBO_instances/scaling_tests/friendly/global_friendly_5x5_d30_s100.npz"

    print(f"\n--- COMPILING BASE CASE: {TARGET_FILE} ---")
    
    Q = load_hamburg_matrix(TARGET_FILE)
    if Q is not None:
        # Calculate classical phase time for CSV
        start_classic = time.time()
        coords, fitness, scale = optimize_embedding(Q, num_restarts=5)
        t_classic = time.time() - start_classic
        
        # Asynchronous Execution
        job_id = run_quantum_job(Q, coords, scale, qpu_emulator)
        
        print("\n" + "="*50)
        print(f" OPERATION COMPLETED")
        print(f" Your Job ID is: {job_id}")
        print(f" Use the 'retrieve.py' script to extract the data.")
        print("="*50)
        
        # --- CSV APPEND SAVING LOGIC ---
        OUTPUT_CSV = "./new_csv/experiment_registry.csv"
        
        # Build the dictionary with current metadata
        info_run = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Instance": os.path.basename(TARGET_FILE),
            "N_Nodes": len(Q),
            "Spatial_Fitness": round(fitness, 6),
            "Scale_Factor": round(scale, 4),
            "Classical_Time_s": round(t_classic, 2),
            "Juelich_Job_ID": job_id,
            "Fitness_Metric": "Topological_MAE"  # Change to "MSE" when testing the other metric
        }
        
        # If the file already exists, load the history and append the new row.
        # If it doesn't exist, Pandas will create it from scratch with headers.
        if os.path.exists(OUTPUT_CSV):
            df_history = pd.read_csv(OUTPUT_CSV)
            df_new = pd.DataFrame([info_run])
            df_updated = pd.concat([df_history, df_new], ignore_index=True)
        else:
            df_updated = pd.DataFrame([info_run])
            
        df_updated.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[INFO] Run data successfully saved in '{OUTPUT_CSV}'")