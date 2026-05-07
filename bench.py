import pygad
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from pulser import InterpolatedWaveform, Pulse, Sequence, Register
from pulser_myqlm import IsingAQPU
import dataclasses


#WORK IN PROGRESS to build a classic/quantum hybrid script complete pipeline to test our GA
#on ground truth benchmarks. This is the "playground" where to test and improve the GA to use it 
#in hamburg_bench.py

# --- 1. PROFILO FISICO (FAKE JADE) ---
try:
    from pulser.devices import Jade as target_device
except ImportError:
    from pulser.devices import AnalogDevice
    try:
        target_device = dataclasses.replace(AnalogDevice, name="FakeJade", max_radial_distance=50)
    except TypeError:
        target_device = dataclasses.replace(AnalogDevice, name="FakeJade", maximum_radial_distance=50)
    print("[AVVISO] Profilo Jade iniettato artificialmente (Raggio esteso a 50 µm).")

device = target_device
MIN_DIST = device.min_atom_distance
MAX_RADIUS = device.max_radial_distance if hasattr(device, 'max_radial_distance') else 50

# --- 2. DEFINIZIONE DEL PROBLEMA QUBO ---
# Hardcoded per test, ma sostituibile con il parser per il benchmark ml-uhh
Q = np.array([
    [-10.0, 19.74, 19.74, 5.42, 5.42],
    [19.74, -10.0, 20.68, 0.18, 0.86],
    [19.74, 20.68, -10.0, 0.86, 0.18],
    [5.42,  0.18,  0.86, -10.0, 0.32],
    [5.42,  0.86,  0.18,  0.32, -10.0],
])

N_ATOMS = len(Q)

# --- 3. MOTORE GENETICO: PyGAD PER L'EMBEDDING ---
print("\nAvvio Algoritmo Genetico (PyGAD) per l'embedding spaziale...")

# Parametri fisici vincolanti
device = target_device
MIN_DIST = device.min_atom_distance # micrometri (distanza limite tipica dei tweezer)

# Field of view (+/- 40 micrometri). Conservativo, Jade dovrebbe permettere fino a +/- 50

# Estrazione dei termini quadratici (interazioni spaziali)
Q_off_diag = Q.copy()
np.fill_diagonal(Q_off_diag, 0)

# --- IL FATTORE DI SCALA (NORMALIZZAZIONE FISICA) ---
#si vanno a riscalare i valori fuori diagonale (che regolano le interazioni tra coppie di atomi)
#così da non permettere la disposizione "alla deriva" (angoli del register) di tali atomi derivante da un'eccessiva penalizzazione
#di valori altrimenti erroneamente considerati troppo grandi.


# 1. Limite Spaziale
V_max_allowed = device.interaction_coeff / (MIN_DIST**6)
Q_max_off_diag = np.max(Q_off_diag)
#fattore di scala per non penalizzare troppo le connessioni deboli
scale_space = V_max_allowed / Q_max_off_diag if Q_max_off_diag > 0 else float('inf')

# 2. Limite del Laser (Usiamo un limite di sicurezza, es. 40 rad/µs se il device non lo ha)
channel = device.channels["rydberg_global"]
max_detuning = channel.max_abs_detuning if channel.max_abs_detuning is not None else 40.0
avg_linear_weight = np.mean(np.abs(np.diag(Q)))
scale_laser = max_detuning / avg_linear_weight if avg_linear_weight > 0 else float('inf')

# 3. Il vero Fattore di Scala è il collo di bottiglia tra i due.
scale_factor = min(scale_space, scale_laser)

# 4. Creiamo la Matrice Target: i pesi QUBO ora sono tradotti in veri potenziali fisici
# che possono essere usati nella valutazione della fitness function
Q_target = Q_off_diag * scale_factor
print(f"Fattore di scala (Spazio: {scale_space:.2f}, Laser: {scale_laser:.2f}) -> Scelto: {scale_factor:.4f}")


print(f"Fattore di scala applicato: {scale_factor:.4f}")
print(f"Max QUBO originale: {Q_max_off_diag} -> Max target fisico: {V_max_allowed:.2f}")

#Il GA si occupa di trovare la disposizione degli atomi sulla base dei valori fuori diagonale
#mentre i valori sulla diagonale sono regolati dal detuning del raggio laser.

def fitness_func(ga_instance, solution, solution_idx):
    coords = np.reshape(solution, (N_ATOMS, 2))
    distances = pdist(coords)
    
    # 1. Calcolo del potenziale fisico
    # Aggiungiamo un piccolissimo epsilon (1e-9) per evitare divisioni per zero assolute
    V_physical = squareform(device.interaction_coeff / ((distances + 1e-9) ** 6))
    
    # Estraiamo i triangoli superiori per il confronto
    V_triu = V_physical[np.triu_indices(N_ATOMS, k=1)]
    Q_triu = Q_target[np.triu_indices(N_ATOMS, k=1)]
    
    # 2. Errore Assoluto (MAE) al posto del Quadratico (MSE)
    # Questo impedisce al decadimento 1/r^6 di "accecare" l'algoritmo
    # facendogli ignorare le connessioni deboli (atomi lontani)
    base_error = np.sum(np.abs(V_triu - Q_triu))
    
    # 3. Penalità Fluida (Soft Penalty)
    penalty = 0.0

    # 1. Death Penalty: Distanza Minima (Collisioni)
    if np.any(distances < MIN_DIST):
        violation = np.sum(np.clip(MIN_DIST - distances, 0, None))
        penalty += violation * 100000.0 
        
    # 2. Death Penalty: Raggio Massimo (Fuori dal laser)
    radii = np.linalg.norm(coords, axis=1)
    if np.any(radii > MAX_RADIUS):
        violation = np.sum(np.clip(radii - MAX_RADIUS, 0, None))
        penalty += violation * 100000.0
        
    total_error = base_error + penalty
    return 1.0 / (total_error + 1e-6)


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

# --- PARAMETRI ESTINZIONE ---
STAGNATION_LIMIT = 40
last_best_fitness = 0.0
stagnation_counter = 0

# Variabili per tenere traccia del "Campione Assoluto" tra tutte le run
NUM_RESTARTS = 10 #restart casuali per prendere la migliore dellle n run
best_overall_fitness = -float('inf')
best_overall_coords = None

print(f"\nAvvio di {NUM_RESTARTS} run indipendenti di PyGAD (Multi-Start Strategy)...")

gene_space = [{'low': -MAX_RADIUS, 'high': MAX_RADIUS} for _ in range(N_ATOMS * 2)]

for run_idx in range(NUM_RESTARTS):
    # Reset per ogni nuova run
    last_best_fitness = 0.0
    stagnation_counter = 0

    def on_generation(ga_instance):
        global last_best_fitness, stagnation_counter
        current_best = ga_instance.best_solution()[1]
        if current_best > last_best_fitness + 1e-6:
            last_best_fitness = current_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        if stagnation_counter >= STAGNATION_LIMIT:
            num_replacements = int(ga_instance.sol_per_pop * 0.3)
            new_genes = np.random.uniform(low=-MAX_RADIUS, high=MAX_RADIUS, size=(num_replacements, N_ATOMS * 2))
            ga_instance.population[-num_replacements:] = new_genes
            stagnation_counter = 0

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
        print(f"  -> [Run {run_idx+1}/{NUM_RESTARTS}] Nuovo record! Fitness: {run_best_fitness:.4f}")
    else:
        print(f"  -> [Run {run_idx+1}/{NUM_RESTARTS}] Nessun miglioramento (Fitness: {run_best_fitness:.4f})")

print(f"\nEmbedding Multi-Start completato! Useremo il Campione (Fitness: {best_overall_fitness:.4f})")

# --- 3.5 CREAZIONE REGISTRO SPAZIALE---
# USARE best_overall_coords
qubits = {f"q{i}": coord for i, coord in enumerate(best_overall_coords)}
reg = Register(qubits)

print("Generazione del plot spaziale dei qubit...")
# draw_half_radius=True disegna i cerchi col raggio dimezzato: 
# se i cerchi si toccano, gli atomi sono esattamente a MIN_DIST.
reg.draw(
    blockade_radius=MIN_DIST, 
    draw_half_radius=True, 
    draw_graph=False
)

# --- 4. COSTRUZIONE SEQUENZA ADIABATICA ---
print("Costruzione della sequenza adiabatica...")
# La mediana deve essere calcolata sui pesi SCALATI
ideal_omega = np.median(Q_target[Q_target > 0])
channel_max_amp = device.channels["rydberg_global"].max_amp
Omega = min(ideal_omega, channel_max_amp / 1.2) if channel_max_amp else ideal_omega

scaled_delta = avg_linear_weight * scale_factor

duration_us = 4
T = duration_us * 1000 
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

# --- 5. ESECUZIONE SU JÜLICH HPC ---
# Inizializza l'emulatore (se non era stato fatto)
try:
    from qlmaas.qpus import AnalogQPU
    qpu_emulator = AnalogQPU()
except ImportError:
    from qat.qlmaas.qpus import QLMaaSQPU
    qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")

job = IsingAQPU.convert_sequence_to_job(seq, nbshots=0)

print(f"  -> Invio pacchetto alla coda remota (HW: {device.name})...")
# UN SINGOLO INVIO ASINCRONO
results = qpu_emulator.submit(job).join()
print("Esecuzione completata!")

# --- 6. PARSING E PLOTTING ---
samples = {}
for sample in results.raw_data:
    bitstring = sample.state.bitstring.zfill(N_ATOMS)
    samples[bitstring] = sample.probability

samples = dict(sorted(samples.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(10, 5))
plt.bar(list(samples.keys())[:30], list(samples.values())[:30], color="blue", alpha=0.7)
plt.xlabel("Bitstrings")
plt.ylabel("Probabilità")
plt.title(f"Top 30 Soluzioni Trovate (Best Fitness: {best_overall_fitness:.4f})")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()