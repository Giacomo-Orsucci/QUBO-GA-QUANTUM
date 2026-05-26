import os
import glob
import time
import numpy as np
import pandas as pd
import pygad
import networkx as nx
import scipy.sparse as sp 
from scipy.spatial.distance import pdist, squareform
import dataclasses

from pulser import InterpolatedWaveform, Pulse, Sequence, Register
from pulser_myqlm import IsingAQPU

#This is the new script to explore the direction without discretization of state space.
#HERE the GA tries to find the best configuration on a continuos space with auxiliary ancillas.

# ==========================================
# SETUP HARDWARE (FakeJade)
# ==========================================
try:
    from pulser.devices import Jade as target_device
except ImportError:
    from pulser.devices import AnalogDevice
    try:
        target_device = dataclasses.replace(AnalogDevice, name="FakeJade", max_radial_distance=50)
    except TypeError:
        target_device = dataclasses.replace(AnalogDevice, name="FakeJade", maximum_radial_distance=50)
    print("[AVVISO] Profilo Jade iniettato artificialmente (Raggio esteso a 50 µm).")

# ==========================================
# PARSER DEI .CSV e .NPZ 
# ==========================================
def load_hamburg_matrix(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.npz':
            with np.load(file_path, allow_pickle=True) as data:
                i_indices, j_indices, weights = data['i'], data['j'], data['Jij']
                n_nodes = int(max(np.max(i_indices), np.max(j_indices))) + 1
                Q = np.zeros((n_nodes, n_nodes))
                for r, c, w in zip(i_indices, j_indices, weights):
                    r, c = int(r), int(c)
                    Q[r, c] = w
                    if r != c: Q[c, r] = w
                return Q
                
        elif ext == '.csv':
            df = pd.read_csv(file_path, header=None, names=['i', 'j', 'weight'])
            n_nodes = int(max(df['i'].max(), df['j'].max())) + 1
            Q = np.zeros((n_nodes, n_nodes))
            for _, row in df.iterrows():
                r, c, w = int(row['i']), int(row['j']), row['weight']
                Q[r, c] = w
                if r != c: Q[c, r] = w
            return Q
        return None
    except Exception as e:
        print(f"  [ERRORE LETTURA] {file_path}: {e}")
        return None

# ==========================================
# MOTORE GENETICO (CONTINUO + ANCILLE)
# ==========================================
def optimize_embedding(Q, num_restarts=1):
    N_ATOMS = len(Q)
    MAX_ANCILLAS = 4  # Quante ancille fluttuanti concediamo al max al GA
    device = target_device
    MIN_DIST = device.min_atom_distance
    MAX_RADIUS = device.max_radial_distance if hasattr(device, 'max_radial_distance') else 50.0

    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)
    
    # --- CALCOLO FATTORE DI SCALA ---
    V_max_allowed = device.interaction_coeff / (MIN_DIST**6)
    Q_max_off_diag = np.max(np.abs(Q_off_diag))    
    scale_space = V_max_allowed / Q_max_off_diag if Q_max_off_diag > 0 else float('inf')

    channel = device.channels["rydberg_global"]
    max_detuning = channel.max_abs_detuning if channel.max_abs_detuning is not None else 40.0
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scale_laser = max_detuning / avg_linear_weight if avg_linear_weight > 0 else float('inf')

    scale_factor = min(scale_space, scale_laser)
    Q_target = Q_off_diag * scale_factor

    # --- 1. IL CROMOSOMA IBRIDO ---
    # Geni = [x1,y1, x2,y2 ... xN,yN] + [ax1,ay1,act1 ... axM,ayM,actM]
    gene_space = []
    gene_type = []
    
    # Nodi Logici (sempre attivi, coordinate continue)
    for _ in range(N_ATOMS):
        gene_space.extend([{'low': -MAX_RADIUS, 'high': MAX_RADIUS}, {'low': -MAX_RADIUS, 'high': MAX_RADIUS}])
        gene_type.extend([float, float])
        
    # Nodi Ancilla (coordinate continue + flag binario di attivazione)
    for _ in range(MAX_ANCILLAS):
        gene_space.extend([{'low': -MAX_RADIUS, 'high': MAX_RADIUS}, {'low': -MAX_RADIUS, 'high': MAX_RADIUS}, [0, 1]])
        gene_type.extend([float, float, int])

    num_genes = (N_ATOMS * 2) + (MAX_ANCILLAS * 3)
    sol_per_pop = 150

    # --- 2. WARM-START (Initial Population con Molle) ---
    print("  -> Generazione Popolazione Iniziale (Warm-start Force-Directed)...")
    G = nx.from_numpy_array(np.abs(Q_off_diag))
    spring_pos = nx.spring_layout(G, scale=MAX_RADIUS * 0.6) 
    
    initial_population = []
    for _ in range(sol_per_pop):
        ind = []
        # Inseriamo i nodi logici basandoci sul layout a molla (con un po' di rumore)
        for i in range(N_ATOMS):
            x = np.clip(spring_pos[i][0] + np.random.normal(0, 3.0), -MAX_RADIUS, MAX_RADIUS)
            y = np.clip(spring_pos[i][1] + np.random.normal(0, 3.0), -MAX_RADIUS, MAX_RADIUS)
            ind.extend([x, y])
            
        # Inseriamo le ancille con posizioni e attivazioni random
        for _ in range(MAX_ANCILLAS):
            ind.extend([np.random.uniform(-MAX_RADIUS, MAX_RADIUS), 
                        np.random.uniform(-MAX_RADIUS, MAX_RADIUS), 
                        np.random.choice([0, 1])])
        initial_population.append(ind)

    # --- 3. LA FUNZIONE DI FITNESS ---
    def fitness_func(ga_instance, solution, solution_idx):
        # Estrazione nodi logici
        logical_coords = np.array(solution[:N_ATOMS*2]).reshape((N_ATOMS, 2))
        
        # Estrazione ancille attive
        ancilla_data = solution[N_ATOMS*2:]
        active_ancillas = []
        for i in range(MAX_ANCILLAS):
            x, y, active = ancilla_data[i*3 : i*3+3]
            if active == 1:
                active_ancillas.append([x, y])
                
       # Forza il cast a float64 per evitare l'errore "Unsupported dtype object"
        if active_ancillas:
            all_coords = np.vstack([logical_coords, active_ancillas]).astype(np.float64)
        else:
            all_coords = logical_coords.astype(np.float64)
            
        distances_all = pdist(all_coords)
        
        # Anche per le distanze logiche, assicuriamoci che siano float
        distances_logical = pdist(logical_coords.astype(np.float64))
        
        # Termine 1: Errore MSE (solo sui nodi logici rispetto al QUBO)
        V_logical = squareform(device.interaction_coeff / ((distances_logical + 1e-9) ** 6))
        V_triu = V_logical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        W_errore = 1.0
        mse_error = np.mean((V_triu - Q_triu) ** 2)
        
        # Termine 2: Hard Constraint (Penalità)
        W_penalita = 100000.0
        penalty = 0.0
        if np.any(distances_all < MIN_DIST):
            violation = np.sum(np.clip(MIN_DIST - distances_all, 0, None))
            penalty += violation * W_penalita
            
        radii = np.linalg.norm(all_coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            violation = np.sum(np.clip(radii - MAX_RADIUS, 0, None))
            penalty += violation * W_penalita
            
        # (Opzionale: aggiungere qui la logica di penalità per ancille inutilizzate)

        total_error = (W_errore * mse_error) + penalty
        return 1.0 / (total_error + 1e-6)

    # --- 4. CROSSOVER SPAZIALE GEOMETRICO ---
    def spatial_crossover(parents, offspring_size, ga_instance):
        offspring = []
        idx = 0
        while len(offspring) < offspring_size[0]:
            p1 = parents[idx % parents.shape[0], :].copy()
            p2 = parents[(idx + 1) % parents.shape[0], :].copy()
            
            p1_log_coords = p1[:N_ATOMS*2].reshape(N_ATOMS, 2)
            p2_log_coords = p2[:N_ATOMS*2].reshape(N_ATOMS, 2)
            
            child = np.zeros_like(p1)
            # Taglio geometrico: gli atomi con X > 0 ereditano da p1, X <= 0 da p2
            for i in range(N_ATOMS):
                if p1_log_coords[i, 0] > 0:
                    child[i*2 : i*2+2] = p1_log_coords[i]
                else:
                    child[i*2 : i*2+2] = p2_log_coords[i]
            
            # Crossover a punto singolo per la sezione ancille
            split_pt = len(p1) - (MAX_ANCILLAS*3) + (np.random.randint(MAX_ANCILLAS) * 3)
            child[N_ATOMS*2 : split_pt] = p1[N_ATOMS*2 : split_pt]
            child[split_pt:] = p2[split_pt:]
            
            offspring.append(child)
            idx += 1
        return np.array(offspring)

    # --- ESECUZIONE GA MULTI-START ---
    best_overall_fitness = -float('inf')
    best_overall_coords = None
    best_ancillas = []

    print(f"  -> Avvio Multi-Start: {num_restarts} run (Continuo + Ancille)...")

    for run_idx in range(num_restarts):
        
        # Callback per far decrescere la mutazione nel tempo (Simulated Annealing style)
        def on_generation(ga_instance):
            gen = ga_instance.generations_completed
            # Riduciamo la finestra di mutazione dell'1% a ogni generazione
            if gen > 0:
                ga_instance.random_mutation_min_val *= 0.99
                ga_instance.random_mutation_max_val *= 0.99
                
            if gen % 200 == 0:
                current_best = ga_instance.best_solution()[1]
                print(f"      [Run {run_idx+1}] Gen {gen}/{ga_instance.num_generations} | Fitness: {current_best:.6f}")

        ga = pygad.GA(
            num_generations=1500,
            num_parents_mating=20,
            initial_population=initial_population,
            fitness_func=fitness_func,
            num_genes=num_genes,
            gene_type=gene_type,
            gene_space=gene_space,
            parent_selection_type="tournament",
            crossover_type=spatial_crossover, # Usa il nostro operatore 2D!
            mutation_type="random",
            mutation_probability=0.1,
            random_mutation_min_val=-5.0, # Inizio con escursioni larghe (5 µm)
            random_mutation_max_val=5.0,
            keep_elitism=5,
            on_generation=on_generation,
            suppress_warnings=True
        )

        ga.run()
        run_best_solution, run_best_fitness, _ = ga.best_solution()
        
        if run_best_fitness > best_overall_fitness:
            best_overall_fitness = run_best_fitness
            
            best_overall_coords = np.array(run_best_solution[:N_ATOMS*2]).reshape(N_ATOMS, 2)
            ancilla_data = run_best_solution[N_ATOMS*2:]
            best_ancillas = [[ancilla_data[i*3], ancilla_data[i*3+1]] 
                             for i in range(MAX_ANCILLAS) if ancilla_data[i*3+2] == 1]
            
            print(f"  -> [Run {run_idx+1}] RECORD! Fit: {run_best_fitness:.6f} | Ancille attive: {len(best_ancillas)}")

    # Uniamo nodi logici e ancille attive per la QPU
    final_coords = best_overall_coords.tolist() + best_ancillas
    return final_coords, best_overall_fitness, scale_factor, Q_off_diag

# ==========================================
# ESECUZIONE QUANTISTICA (ASINCRONA)
# ==========================================
def run_quantum_job(Q, coords, scale_factor, qpu_emulator):
    device = target_device
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)
    Q_target = Q_off_diag * scale_factor
    
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scaled_delta = avg_linear_weight * scale_factor
    
    # Nomi per la QPU: Logici = q0..q15, Ancille = a0..aM
    qubits = {}
    for i, c in enumerate(coords):
        if i < len(Q): qubits[f"q{i}"] = c
        else: qubits[f"a{i - len(Q)}"] = c
        
    reg = Register(qubits)
    reg.draw(blockade_radius=device.min_atom_distance, draw_half_radius=True, draw_graph=False)
    
    ideal_omega = np.median(Q_target[Q_target > 0]) if np.any(Q_target > 0) else 1.0
    channel_max_amp = device.channels["rydberg_global"].max_amp
    Omega = min(ideal_omega, channel_max_amp / 1.2) if channel_max_amp else ideal_omega

    T = 4 * 1000 
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
    print(f"  -> Invio pacchetto (HW: {device.name})...")

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            async_job = qpu_emulator.submit(job)
            try: job_id = async_job.batch_id
            except AttributeError: job_id = str(async_job) 
            print(f"  -> [SUCCESSO] Job accettato! ID: {job_id}")
            return job_id, reg
        except Exception as e:
            if attempt < MAX_RETRIES - 1: time.sleep(5)
            else: return "ERRORE_INVIO", reg

# ==========================================
# ORCHESTRATORE
# ==========================================
def run_benchmark(dataset_folder, output_csv="benchmark_results.csv"):
    try:
        from qlmaas.qpus import AnalogQPU
        qpu_emulator = AnalogQPU()
    except ImportError:
        from qat.qlmaas.qpus import QLMaaSQPU
        qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
   
    files = glob.glob(os.path.join(dataset_folder, "*.npz")) + glob.glob(os.path.join(dataset_folder, "*.csv"))
    files.sort()
    
    results_list = []
    
    for idx, file_path in enumerate(files[:1]): 
        filename = os.path.basename(file_path)
        print(f"\nAnalisi di: {filename}")
        Q = load_hamburg_matrix(file_path)
        
        try:
            start_classic = time.time()
            coords, fitness, scale, Q_expanded = optimize_embedding(Q, num_restarts=1)
            t_classic = time.time() - start_classic
            print(f"  -> Spazio ottimizzato. Fit: {fitness:.4f}, Scale: {scale:.4f}")
            
            job_id, reg = run_quantum_job(Q_expanded, coords, scale, qpu_emulator)      
            
            results_list.append({
                "Istanza": filename, "Fit_Spaziale": round(fitness, 4),
                "Tempo_s": round(t_classic, 2), "Job_ID": job_id
            })
        except Exception as e:
            results_list.append({"Istanza": filename, "Job_ID": str(e)})
            
        pd.DataFrame(results_list).to_csv(output_csv, index=False)

if __name__ == "__main__":
    TARGET_FOLDER = "./qubo-bench/qubo-benchmark-main/generate/compsup/instances/2d_(4, 4)_precision256"
    OUTPUT_FILE = "risultati_benchmark.csv"
    run_benchmark(TARGET_FOLDER, output_csv=OUTPUT_FILE)