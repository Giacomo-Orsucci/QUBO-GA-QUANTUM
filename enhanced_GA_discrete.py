import os
import glob
import time
import numpy as np
import pandas as pd
import pygad
import scipy.sparse as sp 
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt 
import dataclasses

from pulser import InterpolatedWaveform, Pulse, Sequence, Register
from pulser_myqlm import IsingAQPU

#This is the new script to explore the direction with discretization of state space.
#HERE the GA tries to find the best configuration on a fixed grid with auxiliary ancillas.

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
        else:
            print(f"  [AVVISO] Formato {ext} non supportato.")
            return None
    except Exception as e:
        print(f"  [ERRORE LETTURA] {file_path}: {e}")
        return None

def apply_minor_embedding(Q_original, max_degree=5, J_chain=-3.0):
    """
    Scansiona la matrice QUBO e applica dinamicamente il Minor Embedding.
    Splitta i nodi logici che superano 'max_degree' aggiungendo atomi ancilla (qubit fisici).
    """
    Q_new = Q_original.copy()
    N_original = len(Q_new)
    
    # Mappatura: serve per ricordarci quali atomi fisici compongono un singolo nodo logico
    # Es. logico 4 -> fisici [4, 16]
    logical_to_physical = {i: [i] for i in range(N_original)}
    
    # Contatore degli split effettuati
    split_count = 0 
    
    for i in range(N_original):
        # Trova tutti i vicini del nodo logico i (ignora la diagonale)
        neighbors = np.where(Q_new[i] != 0)[0]
        # Rimuoviamo se stesso dai vicini in caso di termini lineari
        neighbors = neighbors[neighbors != i] 
        
        if len(neighbors) > max_degree:
            print(f"      [Minor Embedding] Nodo logico {i} è un Hub (grado {len(neighbors)}). Splittamento in corso...")
            split_count += 1
            
            # Calcola quanti vicini spostare sull'ancilla (la metà)
            split_idx = len(neighbors) // 2
            neighbors_to_move = neighbors[split_idx:]
            
            # Crea un nuovo indice per l'atomo ancilla in coda alla matrice
            new_node_idx = len(Q_new)
            logical_to_physical[i].append(new_node_idx)
            
            # Espandi la matrice di 1 riga e 1 colonna (riempiendo di zeri)
            Q_new = np.pad(Q_new, ((0, 1), (0, 1)), mode='constant')
            
            # Sposta i legami dal vecchio nodo all'ancilla
            for neighbor in neighbors_to_move:
                weight = Q_new[i, neighbor]
                # Cancella il legame dal nodo originale
                Q_new[i, neighbor] = 0.0
                Q_new[neighbor, i] = 0.0
                # Ricrea il legame sull'atomo ancilla
                Q_new[new_node_idx, neighbor] = weight
                Q_new[neighbor, new_node_idx] = weight
            
            # --- IL CUORE DEL MINOR EMBEDDING ---
            # Crea la Catena Ferromagnetica tra il nodo originale e l'ancilla
            Q_new[i, new_node_idx] = J_chain
            Q_new[new_node_idx, i] = J_chain
            
            print(f"        -> Creato atomo ancilla (Fisico: {new_node_idx}). Legame di catena: {J_chain}")

    if split_count > 0:
        print(f"      [Minor Embedding] Espansione completata. Matrice passata da {N_original} a {len(Q_new)} atomi fisici.")
    else:
        print("      [Minor Embedding] Nessun Hub rilevato. La matrice è già sufficientemente sparsa.")
        
    return Q_new, logical_to_physical


# ==========================================
# MOTORE GENETICO (GRID-BASED DISCRETO)
# ==========================================
def optimize_embedding(Q, num_restarts=10):
    N_ATOMS = len(Q)
    device = target_device
    MIN_DIST = device.min_atom_distance
    
    # --- CREAZIONE RETICOLO OTTICO (GRID) ---
    GRID_DIM = 6          # 6x6 = 36 slot
    STEP_UM = 8.0         # Distanza tra slot (maggiore di MIN_DIST)
    grid_coords = []
    for i in range(GRID_DIM):
        for j in range(GRID_DIM):
            x = (i - GRID_DIM/2) * STEP_UM
            y = (j - GRID_DIM/2) * STEP_UM
            grid_coords.append((x, y))
    grid_coords = np.array(grid_coords)
    NUM_SLOTS = len(grid_coords)
    gene_space = list(range(NUM_SLOTS)) # Indici da 0 a 35

    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)

        # --- PRUNING E PREPARAZIONE MATRICE ---

    #PRUNING_PERCENTAGE = 30
    #pesi_esistenti = np.abs(Q_off_diag[Q_off_diag != 0])
    #if len(pesi_esistenti) > 0:
     #   soglia = np.percentile(pesi_esistenti, PRUNING_PERCENTAGE)
      #  print(f"  Applicazione pruning a valori sotto soglia < {soglia:.4f}")
      #  Q_off_diag[np.abs(Q_off_diag) < soglia] = 0.0

    # --------  MINOR EMBEDDING --------
    print("\n  -> Analisi della topologia per eventuale Minor Embedding...")
    
    max_degree = 3 #(forza lo split molto spesso)
    J_chain = -3.0 #(forza catena. I pesi normali arrivano a -1.0)
    Q_off_diag, logical_map = apply_minor_embedding(Q_off_diag, max_degree=3, J_chain=-3.0)
    
    # AGGIORNAMENTO CRITICO: N_ATOMS ora potrebbe essere > 16!
    N_ATOMS = len(Q_off_diag)

    # --- CONTROLLO DI SICUREZZA GRIGLIA ---
    if N_ATOMS > NUM_SLOTS:
        print(f"  [AVVISO] Atomi ({N_ATOMS}) superano gli slot ({NUM_SLOTS}). Allargo la griglia a 7x7...")
        GRID_DIM = 7
        grid_coords = []
        for i in range(GRID_DIM):
            for j in range(GRID_DIM):
                x = (i - GRID_DIM/2) * STEP_UM
                y = (j - GRID_DIM/2) * STEP_UM
                grid_coords.append((x, y))
        grid_coords = np.array(grid_coords)
        NUM_SLOTS = len(grid_coords)
        gene_space = list(range(NUM_SLOTS))
    # --------------------------------------------------

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

    # --- FITNESS TOPOLOGICA DISCRETA ---
    def topological_fitness_func(ga_instance, solution, solution_idx):
        # 1. Mappatura dagli indici discreti alle coordinate fisiche
        coords = grid_coords[solution]
        distances = pdist(coords)
        
        # 2. Calcolo potenziale fisico
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        # 3. Ponderazione Topologica 
        bond_importance = np.abs(Q_triu)
        if np.max(bond_importance) > 0:
            bond_importance = bond_importance / np.max(bond_importance)
        
        base_error = np.sum(bond_importance * np.abs(V_triu - np.abs(Q_triu)))        
        # 4. Le Death Penalty per sovrapposizioni o uscite dal raggio
        # (Con la griglia non dovrebbero verificarsi, ma le manteniamo come scudo di sicurezza)
        penalty = 0.0
        if np.any(distances < MIN_DIST):
            violation = np.sum(np.clip(MIN_DIST - distances, 0, None))
            penalty += violation * 100000.0 
            
        MAX_RADIUS = device.max_radial_distance if hasattr(device, 'max_radial_distance') else 50
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            violation = np.sum(np.clip(radii - MAX_RADIUS, 0, None))
            penalty += violation * 100000.0
            
        total_error = base_error + penalty
        return 1.0 / (total_error + 1e-6)

    # --- ESECUZIONE MULTI-START ---
    STAGNATION_LIMIT = 40
    best_overall_fitness = -float('inf')
    best_overall_coords = None

    print(f"  -> Avvio Multi-Start (Grid-Based): {num_restarts} run indipendenti per esplorare le permutazioni...")

    for run_idx in range(num_restarts):
        ga_state = {'last_best': 0.0, 'stagnation': 0}

        def on_generation(ga_instance):
            current_best = ga_instance.best_solution()[1]
            gen = ga_instance.generations_completed
            if gen % 200 == 0:
                print(f"      [Run {run_idx+1}] Gen {gen}/{ga_instance.num_generations} | Best fitness temporanea: {current_best:.4f}")

            if current_best > ga_state['last_best'] + 1e-6:
                ga_state['last_best'] = current_best
                ga_state['stagnation'] = 0
            else:
                ga_state['stagnation'] += 1
                
            if ga_state['stagnation'] >= STAGNATION_LIMIT:
                # Sostituiamo il 30% della popolazione con nuove permutazioni
                num_replacements = int(ga_instance.sol_per_pop * 0.3)
                # Creiamo nuove soluzioni valide pescando indici univoci
                new_genes = [np.random.choice(gene_space, size=N_ATOMS, replace=False) for _ in range(num_replacements)]
                ga_instance.population[-num_replacements:] = new_genes
                ga_state['stagnation'] = 0

        # GA con parametri ottimizzati per la permutazione
        ga = pygad.GA(
            num_generations=1500,
            num_parents_mating=20,
            fitness_func=topological_fitness_func,
            sol_per_pop=150,
            
            # --- PARAMETRI DISCRETI ---
            num_genes=N_ATOMS,
            gene_type=int,
            gene_space=gene_space,
            allow_duplicate_genes=False,
            mutation_type="swap", 
            # --------------------------
            
            parent_selection_type="tournament",
            K_tournament=3,
            keep_elitism=5,
            crossover_type="single_point",
            on_generation=on_generation,
            suppress_warnings=True
        )

        ga.run()
        run_best_solution, run_best_fitness, _ = ga.best_solution()
        
        if run_best_fitness > best_overall_fitness:
            best_overall_fitness = run_best_fitness
            # Traduzione finale in coordinate
            best_overall_coords = grid_coords[run_best_solution]
            print(f"  -> [Run {run_idx+1}/{num_restarts}] NUOVO RECORD! Fitness finale: {run_best_fitness:.4f}")
        else:
            print(f"  -> [Run {run_idx+1}/{num_restarts}] Nessun miglioramento globale (Fitness: {run_best_fitness:.4f})")

  
    return best_overall_coords, best_overall_fitness, scale_factor, Q_off_diag

# ==========================================
# ESECUZIONE QUANTISTICA (ASINCRONA)
# ==========================================
def run_quantum_job(Q, coords, scale_factor, qpu_emulator):
    N_ATOMS = len(Q)
    device = target_device
    
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)
    Q_target = Q_off_diag * scale_factor
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scaled_delta = avg_linear_weight * scale_factor
    
    qubits = {f"q{i}": c for i, c in enumerate(coords)}
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
    print(f"  -> Invio pacchetto alla coda remota (HW: {device.name})...")

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            async_job = qpu_emulator.submit(job)
            try:
                job_id = async_job.batch_id
            except AttributeError:
                job_id = str(async_job) 
            print(f"  -> [SUCCESSO] Job accettato da Jülich! ID assegnato: {job_id}")
            return job_id, reg
        except Exception as e:
            print(f"  -> [AVVISO] Connessione caduta (Tentativo {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                return "ERRORE_INVIO", reg

# ==========================================
# ORCHESTRATORE
# ==========================================
def run_benchmark(dataset_folder, output_csv="benchmark_results.csv"):
    print("Inizializzazione dell'emulatore remoto AnalogQPU...")
    try:
        from qlmaas.qpus import AnalogQPU
        qpu_emulator = AnalogQPU()
    except ImportError:
        try:
            from qat.qlmaas.qpus import QLMaaSQPU
            qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
        except ImportError:
            print("[ERRORE FATALE] Emulatore non disponibile. Interruzione.")
            return
   
    files = glob.glob(os.path.join(dataset_folder, "*.npz")) + glob.glob(os.path.join(dataset_folder, "*.csv"))
    files.sort()
    
    print(f"\n--- INIZIO BENCHMARK: Trovati {len(files)} file in {dataset_folder} ---")
    results_list = []
    
    for idx, file_path in enumerate(files[:1]): 
        filename = os.path.basename(file_path)
        print(f"\n[{idx+1}/{len(files)}] Analisi di: {filename}")
        
        Q = load_hamburg_matrix(file_path)
        if Q is None: continue
        n_nodes = len(Q)
        
        try:
            start_classic = time.time()
            # Ricevi la matrice espansa
            coords, fitness, scale, Q_expanded = optimize_embedding(Q, num_restarts=1)
            t_classic = time.time() - start_classic
            print(f"  -> Spazio ottimizzato. Fitness Topologica: {fitness:.4f}, Scala: {scale:.4f}")
            
            # Passa la matrice espansa a Pulser!
            job_id, reg = run_quantum_job(Q_expanded, coords, scale, qpu_emulator)      
            
            print(f"  [OK] Fase classica completata in {round(t_classic,1)}s.")
            print(f"  [AVVISO] Job {job_id} in esecuzione.")
            
            results_list.append({
                "Istanza": filename,
                "N_Nodi": n_nodes,
                "Fitness_Spaziale": round(fitness, 4),
                "Fattore_Scala": round(scale, 4),
                "Tempo_Classico_s": round(t_classic, 2),
                "Jülich_Job_ID": job_id,
                "Stato_Invio": "Inviato" if job_id != "ERRORE_INVIO" else "Fallito"
            })
            
        except Exception as e:
            print(f"  [FALLITO] Errore critico su {filename}: {e}")
            results_list.append({"Istanza": filename, "N_Nodi": n_nodes, "Stato_Invio": str(e)})
            
        pd.DataFrame(results_list).to_csv(output_csv, index=False)
    print(f"\n--- INVIO BENCHMARK COMPLETATO! Controlla il file: {output_csv} ---")

if __name__ == "__main__":
    TARGET_FOLDER = "./qubo-bench/qubo-benchmark-main/generate/compsup/instances/2d_(4, 4)_precision256"
    OUTPUT_FILE = "risultati_benchmark_2d_4x4.csv"
    run_benchmark(TARGET_FOLDER, output_csv=OUTPUT_FILE)