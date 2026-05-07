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


#WORK IN PROGRESS to build a classic/quantum hybrid script complete pipeline to test our GA
#on ground truth benchmarks.

#Jade non è utilizzabile, quindi l'idea era di emularlo con mockdevice, ma
#a quanto pare non è più ammesso usare un mockdevice (che aveva lo scopo di replicare i vincoli di Jade).
#Quindi, il tentativo è quello di usare AnalogDevice ma "forzarlo" con dei vincoli analoghi a Jade.

try:
    from pulser.devices import Jade as target_device
except ImportError:
    from pulser.devices import AnalogDevice
    # cloniamo AnalogDevice
    # e forziamo il suo raggio massimo a 50 micrometri (le specifiche di Jade)
    try:
        target_device = dataclasses.replace(AnalogDevice, name="FakeJade", max_radial_distance=50)
    except TypeError:
        # Per compatibilità con versioni di Pulser ancora più vecchie
        target_device = dataclasses.replace(AnalogDevice, name="FakeJade", maximum_radial_distance=50)
    print("[AVVISO] Profilo Jade iniettato artificialmente (Raggio esteso a 50 µm).")


#--- PARSER DEI .CSV e .NPZ CONTENENTI I PROBLEMI DEL BENCHMARK ---
def load_hamburg_matrix(file_path):
    """
    Legge i file del benchmark.
    Decodifica nativamente le liste di coordinate salvate negli .npz (i, j, Jij)
    e i file testuali .csv (i, j, weight).
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        # CASO A: File compresso NumPy (.npz) con chiavi 'i', 'j', 'Jij'
        if ext == '.npz':
            with np.load(file_path, allow_pickle=True) as data:
                # Estraiamo gli array esatti trovati con l'ispezione
                i_indices = data['i']
                j_indices = data['j']
                weights = data['Jij']
                
                # Calcoliamo la dimensione della matrice
                n_nodes = int(max(np.max(i_indices), np.max(j_indices))) + 1
                Q = np.zeros((n_nodes, n_nodes))
                
                # Popoliamo la matrice forzando la simmetria fisica
                for r, c, w in zip(i_indices, j_indices, weights):
                    r, c = int(r), int(c)
                    Q[r, c] = w
                    if r != c:
                        Q[c, r] = w
                        
                return Q
                
        # CASO B: File testuale edge-list (.csv)
        elif ext == '.csv':
            df = pd.read_csv(file_path, header=None, names=['i', 'j', 'weight'])
            n_nodes = int(max(df['i'].max(), df['j'].max())) + 1
            Q = np.zeros((n_nodes, n_nodes))
            
            for _, row in df.iterrows():
                r, c, w = int(row['i']), int(row['j']), row['weight']
                Q[r, c] = w
                if r != c:
                    Q[c, r] = w
            return Q
            
        else:
            print(f"  [AVVISO] Formato {ext} non supportato.")
            return None
            
    except Exception as e:
        print(f"  [ERRORE LETTURA] Impossibile caricare {file_path}: {e}")
        return None


#--- MOTORE GENETICO ---
def optimize_embedding(Q, num_restarts=10):
    N_ATOMS = len(Q)
    device = target_device
    MIN_DIST = device.min_atom_distance
    SAFE_MIN_DIST = MIN_DIST + 0.1 # Margine di sicurezza interno
    
    # Raggio massimo dal centro (ora sarà 50 grazie a FakeJade)
    MAX_RADIUS = device.max_radial_distance if hasattr(device, 'max_radial_distance') else 50
    
    # Il gene space è un quadrato largo 100x100 (da -50 a +50)
    gene_space = [{'low': -MAX_RADIUS, 'high': MAX_RADIUS} for _ in range(N_ATOMS * 2)]

    # --- Calcolo Fattore di Scala ---
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)
    V_max_allowed = device.interaction_coeff / (MIN_DIST**6)
    Q_max_off_diag = np.max(Q_off_diag)
    scale_space = V_max_allowed / Q_max_off_diag if Q_max_off_diag > 0 else float('inf')

    channel = device.channels["rydberg_global"]
    max_detuning = channel.max_abs_detuning if channel.max_abs_detuning is not None else 40.0
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scale_laser = max_detuning / avg_linear_weight if avg_linear_weight > 0 else float('inf')

    scale_factor = min(scale_space, scale_laser)
    Q_target = Q_off_diag * scale_factor

    # --- Funzione di Fitness ---
    def fitness_func(ga_instance, solution, solution_idx):
        coords = np.reshape(solution, (N_ATOMS, 2))
        distances = pdist(coords)
        
        # 1. Calcolo del potenziale
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        base_error = np.sum(np.abs(V_triu - Q_triu)) # Metrica MAE
        
        penalty = 0.0
        
        # 2. Death Penalty: Collisione tra atomi
        if np.any(distances < SAFE_MIN_DIST):
            violation = np.sum(np.clip(SAFE_MIN_DIST - distances, 0, None))
            penalty += violation * 100000.0 
            
        # 3. Death Penalty: Fuori dal raggio del laser
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            violation = np.sum(np.clip(radii - MAX_RADIUS, 0, None))
            penalty += violation * 100000.0
            
        return 1.0 / (base_error + penalty + 1e-6)
    
    # --- STRATEGY B: TOPOLOGICAL FITNESS ---
    def topological_fitness_func(ga_instance, solution, solution_idx):
        coords = np.reshape(solution, (N_ATOMS, 2))
        distances = pdist(coords)
        
        # 1. Calculate physical potential (adding 1e-9 to prevent division by zero)
        V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
        
        # Extract upper triangles for comparison
        V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
        Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
        
        # 2. Topological Weighting
        # Calculate how "important" each bond is in the original QUBO.
        bond_importance = np.abs(Q_triu)
        
        # Normalize importance between 0 and 1 for numerical stability
        if np.max(bond_importance) > 0:
            bond_importance = bond_importance / np.max(bond_importance)
        
        # The error is no longer flat. We MULTIPLY the absolute error by the bond importance.
        # The GA will now focus on preserving strong bonds and sacrificing weak ones.
        base_error = np.sum(bond_importance * np.abs(V_triu - Q_triu))
        
        # 3. Soft Penalties
        penalty = 0.0

        # Death Penalty 1: Minimum Distance (Collisions)
        if np.any(distances < MIN_DIST):
            violation = np.sum(np.clip(MIN_DIST - distances, 0, None))
            penalty += violation * 100000.0 
            
        # Death Penalty 2: Maximum Radius (Outside laser FOV)
        radii = np.linalg.norm(coords, axis=1)
        if np.any(radii > MAX_RADIUS):
            violation = np.sum(np.clip(radii - MAX_RADIUS, 0, None))
            penalty += violation * 100000.0
            
        total_error = base_error + penalty
        return 1.0 / (total_error + 1e-6)

    # --- Esecuzione Multi-Start ---
    STAGNATION_LIMIT = 40
    best_overall_fitness = -float('inf')
    best_overall_coords = None


    print(f"  -> Avvio Multi-Start: {num_restarts} run indipendenti per esplorare lo spazio...")

    for run_idx in range(num_restarts):
        ga_state = {'last_best': 0.0, 'stagnation': 0}

        def on_generation(ga_instance):
            current_best = ga_instance.best_solution()[1]
            gen = ga_instance.generations_completed
            
            # Stampa periodica ogni 200 generazioni per mostrare che il processo è vivo
            if gen % 200 == 0:
                print(f"      [Run {run_idx+1}] Gen {gen}/800 | Best fitness temporanea: {current_best:.4f}")

            if current_best > ga_state['last_best'] + 1e-6:
                ga_state['last_best'] = current_best
                ga_state['stagnation'] = 0
            else:
                ga_state['stagnation'] += 1
                
            if ga_state['stagnation'] >= STAGNATION_LIMIT:
                # Decommenta la riga sotto se vuoi vedere quante volte avviene un'estinzione
                # print(f"      [Run {run_idx+1}] Estinzione per stagnazione a gen {gen}. Inserimento nuovi geni...")
                num_replacements = int(ga_instance.sol_per_pop * 0.3)
                new_genes = np.random.uniform(low=-MAX_RADIUS, high=MAX_RADIUS, size=(num_replacements, N_ATOMS * 2))
                ga_instance.population[-num_replacements:] = new_genes
                ga_state['stagnation'] = 0

        ga = pygad.GA(
            num_generations=800,
            num_parents_mating=20,
            fitness_func=topological_fitness_func,
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
            on_generation=on_generation,
            suppress_warnings=True
        )

        ga.run()
        run_best_solution, run_best_fitness, _ = ga.best_solution()
        
        if run_best_fitness > best_overall_fitness:
            best_overall_fitness = run_best_fitness
            best_overall_coords = np.reshape(run_best_solution, (N_ATOMS, 2))
            print(f"  -> [Run {run_idx+1}/{num_restarts}] NUOVO RECORD! Fitness finale: {run_best_fitness:.4f}")
        else:
            print(f"  -> [Run {run_idx+1}/{num_restarts}] Nessun miglioramento globale (Fitness: {run_best_fitness:.4f})")

    return best_overall_coords, best_overall_fitness, scale_factor

# --- ESECUZIONE QUANTISTICA
def run_quantum_job(Q, coords, scale_factor, qpu_emulator):
    """Costruisce la sequenza e sottompone il job all'emulatore."""
    N_ATOMS = len(Q)
    device = target_device
    
    Q_off_diag = Q.copy()
    np.fill_diagonal(Q_off_diag, 0)
    Q_target = Q_off_diag * scale_factor
    avg_linear_weight = np.mean(np.abs(np.diag(Q)))
    scaled_delta = avg_linear_weight * scale_factor
    
    qubits = {f"q{i}": c for i, c in enumerate(coords)}
    reg = Register(qubits)
    print("Embedding su registro")
    reg.draw(blockade_radius=device.min_atom_distance, draw_half_radius=True, draw_graph=False)
    
    ideal_omega = np.median(Q_target[Q_target > 0]) if np.any(Q_target > 0) else 1.0
    channel_max_amp = device.channels["rydberg_global"].max_amp
    Omega = min(ideal_omega, channel_max_amp / 1.2) if channel_max_amp else ideal_omega

    T = 4 * 1000 # 4 microsecondi
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

    # --- 5. ESECUZIONE SU JÜLICH HPC ---

    # --- CON SISTEMA DI RETRY ---

    MAX_RETRIES = 3
    results = None
    
    for attempt in range(MAX_RETRIES):
        try:
            # Invio asincrono
            async_job = qpu_emulator.submit(job)
            print(f"  -> [Tentativo {attempt+1}] Job accettato! In attesa della risoluzione quantistica...")
            
            # Attesa dei risultati
            results = async_job.join()
            print("  -> Risultati ricevuti da Jülich con successo!")
            break 
            
        except Exception as e:
            print(f"  -> [AVVISO] Connessione caduta durante il tentativo {attempt+1}: {e}")
            if attempt < MAX_RETRIES - 1:
                print("  -> Attendo 10 secondi per far respirare la rete e riprovo...")
                time.sleep(10)
            else:
                raise Exception("Jülich irraggiungibile o in timeout cronico dopo 3 tentativi.")

    if results is None:
        raise Exception("Nessun risultato ottenuto dall'emulatore.")
    # ---------------------------------------------
    
    samples = {}
    for sample in results.raw_data:
        bitstring = sample.state.bitstring.zfill(N_ATOMS)
        samples[bitstring] = sample.probability

    samples = dict(sorted(samples.items(), key=lambda item: item[1], reverse=True))
    top_bitstring = list(samples.keys())[0]
    top_probability = list(samples.values())[0]
    
    return top_bitstring, top_probability, samples, reg

# --- ORCHESTRATORE

def run_benchmark(dataset_folder, output_csv="benchmark_results.csv"):


    print("Inizializzazione dell'emulatore remoto AnalogQPU...")
    # --- L'EMULATORE DEVE ESSERE INIZIALIZZATO QUI, UNA SOLA VOLTA ---
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
   
   
    # Cerchiamo sia file .npz che .csv
    files = glob.glob(os.path.join(dataset_folder, "*.npz")) + glob.glob(os.path.join(dataset_folder, "*.csv"))
    files.sort()
    
    print(f"\n--- INIZIO BENCHMARK: Trovati {len(files)} file in {dataset_folder} ---")
    results_list = []
    
    for idx, file_path in enumerate(files[:1]):  #files[1] così per ora lavoro solo sulla prima istanza.
        filename = os.path.basename(file_path)
        print(f"\n[{idx+1}/{len(files)}] Analisi di: {filename}")
        
        Q = load_hamburg_matrix(file_path)
        if Q is None: continue
        
        n_nodes = len(Q)
        print(f"  -> Nodi identificati: {n_nodes}")
        
        try:
            # FASE 1: CLASSICA
            start_classica = time.time()
            coords, fitness, scale = optimize_embedding(Q, num_restarts=10)
            tempo_classico = time.time() - start_classica
            print(f"  -> Spazio ottimizzato. Fitness: {fitness:.4f}, Scala: {scale:.4f}")
            
            # FASE 2: QUANTISTICA
            start_quantum = time.time()
            top_bits, top_prob, samples, reg = run_quantum_job(Q, coords, scale, qpu_emulator)            
            tempo_quantum = time.time() - start_quantum
            print(f"  [OK] Soluzione QPU: {top_bits} (Prob: {top_prob:.2f})")
            print(f"  [OK] Tempi -> GA: {round(tempo_classico,1)}s | QPU: {round(tempo_quantum,1)}s")
            
            results_list.append({
                "Istanza": filename,
                "N_Nodi": n_nodes,
                "Fitness_Spaziale": round(fitness, 4),
                "Fattore_Scala": round(scale, 4),
                "Best_Bitstring": top_bits,
                "Probabilita": round(top_prob, 4),
                "Tempo_Classico_s": round(tempo_classico, 2),
                "Tempo_Quantum_s": round(tempo_quantum, 2),
                "Errore": "Nessuno"
            })

# ==========================================
# ESECUZIONE
# ==========================================
if __name__ == "__main__":
    # Sostituisci questo percorso con la cartella corretta della tua repository clonata.
    # partire dalla cartella 2d_(4, 4) per fare un test rapido su 16 atomi.
    TARGET_FOLDER = "./qubo-bench/qubo-benchmark-main/generate/compsup/instances/2d_(4, 4)_precision256"
    
    # Nome del file in cui verranno salvati i risultati
    OUTPUT_FILE = "risultati_benchmark_2d_4x4.csv"
    
    # Avvio dello script
    run_benchmark(TARGET_FOLDER, output_csv=OUTPUT_FILE)