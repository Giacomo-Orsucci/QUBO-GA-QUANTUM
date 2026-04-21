import pygad
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from pulser import InterpolatedWaveform, Pulse, Sequence, Register
from pulser.devices import MockDevice
from pulser_myqlm import IsingAQPU

#WORK IN PROGRESS to build a classic/quantum hybrid script complete pipeline to test our GA
#on ground truth benchmarks.


# --- 1. SETUP DELL'EMULATORE REMOTO (JÜLICH via myQLM) ---
print("Inizializzazione dell'emulatore remoto AnalogQPU...")
try:
    from qlmaas.qpus import AnalogQPU
    qpu_emulator = AnalogQPU()
    print("Emulatore caricato tramite qlmaas.qpus!")
except ImportError:
    from qat.qlmaas.qpus import QLMaaSQPU
    qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
    print("Emulatore caricato tramite classe base QLMaaSQPU!")

device = MockDevice

# --- 2. DEFINIZIONE DEL PROBLEMA QUBO ---
# Hardcoded per test, ma qui andrà il parser per il benchmark ml-uhh
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

# Normalizziamo Q_target (ignorando la diagonale) per il calcolo dell'errore. Tramite GA ci interessa
# mappare le interezioni tra coppie di elementi, non ci interessa quindi la diagonale (che sarà invece protagonista del raggio di detuning).
Q_off_diag = Q.copy()
np.fill_diagonal(Q_off_diag, 0)
max_Q = np.max(Q_off_diag)

# Parametri fisici vincolanti
MIN_DIST = 4.0 # micrometri (distanza limite tipica dei tweezer)
FOV = 40.0 # Field of view (+/- 40 micrometri). Conservativo, Jade dovrebbe permettere fino a +/- 50

def fitness_func(ga_instance, solution, solution_idx):
    coords = np.reshape(solution, (N_ATOMS, 2)) #solution è lineare di 10 atomi, lo portiamo ad essere matrice 5x2.
    
    # per calcolare velocemente tutte le distanze reciproche
    distances = pdist(coords)
    
    # Penalità molto alta se gli atomi sono troppo vicini
    if np.any(distances < MIN_DIST):
        # Più violano il limite, più la penalità è alta
        violation = np.sum(np.clip(MIN_DIST - distances, 0, None))
        return 1.0 / (1e6 * violation) #si penalizza restituendo una fitness function molto bassa
    
    # Calcolo della matrice delle interazioni fisiche generate da queste coordinate
    V_physical = squareform(device.interaction_coeff / (distances ** 6))
    
    # Errore quadratico rispetto al QUBO target (consideriamo solo le upper-triangular per efficienza)
    error = np.sum((V_physical[np.triu_indices(N_ATOMS, k=1)] - Q_off_diag[np.triu_indices(N_ATOMS, k=1)])**2)
    
    # PyGAD massimizza, quindi restituiamo l'inverso dell'errore
    return 1.0 / (error + 1e-6) #per evitare divisioni per 0 in caso di geometria perfetta (e quindi error=0)

# Setup spazio dei geni (vincolati nel Field of View)
gene_space = [{'low': -FOV, 'high': FOV} for _ in range(N_ATOMS * 2)]

ga = pygad.GA(
    num_generations=500,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=N_ATOMS * 2,
    gene_space=gene_space,
    mutation_type="random",
    mutation_probability=0.3,
    random_mutation_min_val=-2.0, # Jitter spaziale per fine-tuning
    random_mutation_max_val=2.0,
    suppress_warnings=True
)

ga.run()
best_solution, best_fitness, _ = ga.best_solution()
coords = np.reshape(best_solution, (N_ATOMS, 2))
print(f"Embedding completato. Fitness score: {best_fitness:.4f}")

# Creazione del Registro Pulser
qubits = {f"q{i}": coord for i, coord in enumerate(coords)}
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
ideal_omega = np.median(Q_off_diag[Q_off_diag > 0])
channel_max_amp = device.channels["rydberg_global"].max_amp
Omega = min(ideal_omega, channel_max_amp / 1.2) if channel_max_amp else ideal_omega

duration_us = 4
T = duration_us * 1000 
delta_0 = -5.0
delta_f = -delta_0

adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
    InterpolatedWaveform(T, [delta_0, 0, delta_f]),
    0,
)

seq = Sequence(reg, device)
seq.declare_channel("ising", "rydberg_global")
seq.add(adiabatic_pulse, "ising")

# --- 5. ESECUZIONE SU JÜLICH HPC ---

NBSHOTS = 0 
print(f"Preparazione Job ({NBSHOTS} shots)...")
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS)

print("Job inviato. In attesa dei risultati asincroni...")
results = qpu_emulator.submit(job).join()
print("Esecuzione completata!")

# --- 6. PARSING E PLOTTING ---
samples = {}
for sample in results.raw_data:
    bitstring = sample.state.bitstring.zfill(N_ATOMS)
    samples[bitstring] = sample.probability

# Ordina in base alla probabilità
samples = dict(sorted(samples.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(10, 5))
plt.bar(samples.keys(), samples.values(), color="blue", alpha=0.7)
plt.xlabel("Bitstrings")
plt.ylabel("Probabilità")
plt.title(f"Soluzioni Trovare - PyGAD + Pulser (Fitness GA: {best_fitness:.4f})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()