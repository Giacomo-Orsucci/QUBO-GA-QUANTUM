import numpy as np
import pygad
import itertools
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register
from pulser.devices import Device
from pulser.waveforms import InterpolatedWaveform
from qat.core import Result

from pulser_myqlm import FresnelQPU, IsingAQPU

"""
ANALISI DI SCALABILITÀ: DA 5x5 A 10x10 (Stress Test)
Questo script estende il risolutore di embedding tramite Algoritmo Genetico (GA) 
a un problema QUBO 10x10. L'obiettivo è osservare come il GA si comporta di fronte 
a un sistema matematicamente sovradeterminato (45 interazioni da mappare con sole 20 coordinate) 
e a un problema fisico di impacchettamento spaziale (Circle Packing) molto più denso.

Le differenze principali rispetto alla versione 5x5:
1. Matrice Q: Generata casualmente in modo simmetrico (10x10).
2. SPACE_BOUND: Aumentato da 15.0 a 25.0 µm per permettere a 10 atomi di non collidere.
3. Risolutore Classico: Aggiunto per calcolare a priori lo stato fondamentale (soluzione ottima)
   della nuova matrice casuale, necessario per valutare la correttezza della QAA.


   Prima implementazione giusto per saggiare e vedere come i G.A si comportano scalando a 10 atomi. Sicuramente da ricontrollare,
   pulire e migliorare.
"""

# Creiamo una rappresentazione virtuale della QPU Fresnel di Pasqal.
FRESNEL_QPU = FresnelQPU(None)  
FRESNEL_DEVICE = Device.from_abstract_repr(FRESNEL_QPU.get_specs().description)

LOCAL_SIMULATIONS = True
NBSHOTS = 0  
MODULATION = False

if not LOCAL_SIMULATIONS:
    try:
        from qlmaas.qpus import AnalogQPU
    except ImportError as e:
        raise ImportError("Can't import AnalogQPU: simulations can only be performed locally.") from e
    
if not LOCAL_SIMULATIONS and NBSHOTS > 0:
    raise ValueError("Simulation with AnalogQPU: number of shots must be 0.")

# ==========================================
# 1. SETUP DEL PROBLEMA (Matrice Q 10x10)
# ==========================================
np.random.seed(42)  # Seed per rendere il problema riproducibile tra diverse run

# Generiamo interazioni casuali positive tra 0.1 e 20.0 (repulsioni logiche)
Q_random = np.random.uniform(0.1, 20.0, size=(10, 10))
Q = (Q_random + Q_random.T) / 2  # Rendiamo la matrice simmetrica
np.fill_diagonal(Q, -10.0)       # Pesi lineari (detuning locale) sulla diagonale

N_ATOMS = len(Q)

# SPAZIO AMPLIATO: Passiamo a 50.0 µm. Con 10 atomi e 5 µm di raggio di blocco, 
# l'area di 15x15 (quella usata con Q 5x5) era troppo angusta e avrebbe generato troppe soluzioni "morte".
SPACE_BOUND = 50.0      
MIN_DISTANCE = 5.0      
C6_COEFF = FRESNEL_DEVICE.interaction_coeff  
NUM_GENERATIONS = 400  

# ==========================================
# 2. FUNZIONI GENETICHE E FISICHE
# ==========================================
def atom_aware_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        for atom_idx in range(N_ATOMS): 
            x_idx = atom_idx * 2
            y_idx = x_idx + 1
            if np.random.rand() > 0.5:
                offspring[k, x_idx:y_idx+1] = parents[parent1_idx, x_idx:y_idx+1]
            else:
                offspring[k, x_idx:y_idx+1] = parents[parent2_idx, x_idx:y_idx+1]
    return offspring

def fitness_func(ga_instance, solution, solution_idx):
    coords_temp = np.reshape(solution, (N_ATOMS, 2))
    distances = pdist(coords_temp)
    
    if np.min(distances) < MIN_DISTANCE:
        return -999999.0  
        
    Q_fisica = squareform(C6_COEFF / (distances ** 6))
    np.fill_diagonal(Q_fisica, -10.0)
    
    errore = np.linalg.norm(Q_fisica - Q)
    return -errore 

# ==========================================
# 3. GRIGLIA DI TEST (SENSITIVITY ANALYSIS)
# ==========================================
# NOTA: Per un problema a 20 variabili, popolazioni piccole potrebbero faticare. 
# per sicurezza aggiungiamo [400, 800] ai test.
populations = [50, 100, 200, 400, 800]      
mutations_percentage = [1, 2, 3, 5, 15]
REPETITIONS = 10  

num_genes = N_ATOMS * 2 
gene_space = [{'low': 0.0, 'high': SPACE_BOUND} for _ in range(num_genes)]

raw_results = {f"Pop: {pop} | Mut: {mut}%": [] for pop in populations for mut in mutations_percentage}

best_overall_error = float('inf')
best_overall_coords = None
best_overall_name = ""

print(f"--- INIZIO GRID SEARCH (Problema {N_ATOMS}x{N_ATOMS}) CON G.A ---")
tot_runs = len(populations) * len(mutations_percentage) * REPETITIONS
print(f"Esecuzioni totali previste: {tot_runs}\n")

test_counter = 1
for pop in populations:
    for mut in mutations_percentage:
        test_name = f"Pop: {pop} | Mut: {mut}%"
        print(f"Test in corso: {test_name} (attendere {REPETITIONS} run)...")
        
        num_parents = max(2, int(pop * 0.2)) 

        for rep in range(REPETITIONS):
            ga_instance = pygad.GA(
                num_generations=NUM_GENERATIONS,        
                num_parents_mating=num_parents,      
                fitness_func=fitness_func,
                sol_per_pop=pop,            
                num_genes=num_genes,
                gene_space=gene_space,
                mutation_percent_genes=mut,  
                crossover_type=atom_aware_crossover, 
                mutation_type="random",
                suppress_warnings=True
            )
            
            ga_instance.run()
            
            error_history = [abs(fitness) for fitness in ga_instance.best_solutions_fitness]
            raw_results[test_name].append(error_history)
            
            best_sol, best_fit, _ = ga_instance.best_solution()
            current_error = abs(best_fit)
            
            if current_error < best_overall_error:
                best_overall_error = current_error
                best_overall_coords = np.reshape(best_sol, (N_ATOMS, 2))
                best_overall_name = test_name
            
            test_counter += 1

print(f"\n--- RICERCA COMPLETATA ---")
print(f"IL VINCITORE ASSOLUTO È: {best_overall_name}")
print(f"Errore minimo raggiunto (Atteso più alto del 5x5): {best_overall_error:.4f}")

# ==========================================
# 4. VISUALIZZAZIONE DEI RISULTATI (Grafico delle Medie)
# ==========================================
plt.figure(figsize=(16, 10))
for test_name, history_list in raw_results.items():
    avg_history = np.mean(history_list, axis=0)
    # Alziamo il filtro visuale a 100 perché all'inizio gli errori del 10x10 sono enormi
    filtered_avg_history = [min(err, 150) for err in avg_history] 
    linewidth = 3 if test_name == best_overall_name else 1.5
    plt.plot(filtered_avg_history, label=test_name, linewidth=linewidth)

plt.title(f"Analisi di Sensibilità GA ({N_ATOMS}x{N_ATOMS} - Media su {REPETITIONS} esecuzioni)", fontsize=16)
plt.xlabel("Generazione", fontsize=14)
plt.ylabel("Errore Medio", fontsize=14)
plt.legend(fontsize=10, loc="upper right", ncol=2)
plt.grid(True, linestyle="--", alpha=0.7)
plt.ylim(0, 80) # Scala Y adattata
plt.tight_layout()
plt.show()

# ==========================================
# 4.5 RISOLUTORE CLASSICO (Ground State Verità)
# ==========================================
print("\nCalcolo della soluzione esatta del QUBO tramite forza bruta classica...")
def solve_qubo_classically(Q_mat):
    """Esplora le 1024 combinazioni per trovare il minimo reale dell'energia"""
    best_cost = float('inf')
    best_states = []
    # Genera tutti i possibili stati binari per N atomi (es. (0,0,0,0,0,0,0,0,0,0) fino a (1,...,1))
    for state in itertools.product([0, 1], repeat=len(Q_mat)):
        x = np.array(state)
        cost = x.T @ Q_mat @ x
        # Raccoglie i degenerati se ce ne sono
        if cost < best_cost - 1e-9: # Tolleranza per float
            best_cost = cost
            best_states = ["".join(map(str, state))]
        elif abs(cost - best_cost) < 1e-9:
            best_states.append("".join(map(str, state)))
    return best_states, best_cost

# Troviamo le vere stringhe ottimali
true_optimal_states, true_minimum_energy = solve_qubo_classically(Q)
print(f"Stati ottimali teorici trovati: {true_optimal_states} con energia {true_minimum_energy:.4f}")

# ==========================================
# 5. PASSAGGIO ALLA FASE QUANTISTICA (Pulser)
# ==========================================
print(f"\n--- AVVIO SIMULAZIONE QUANTISTICA (QAA) ---")
qubits = {f"q{i}": coord for i, coord in enumerate(best_overall_coords)}
reg = Register(qubits)

print("Visualizzazione del layout spaziale degli atomi...")
reg.draw(
    blockade_radius=FRESNEL_DEVICE.rydberg_blockade_radius(1.0),
    draw_graph=False,
    draw_half_radius=True,
)

# ==========================================
# 6. DEFINIZIONE DELL'IMPULSO ADIABATICO. (stesso del caso 5x5, ma con T più lungo)
# ==========================================
max_amp = FRESNEL_DEVICE.channels["rydberg_global"].max_amp
Omega = min(np.median(Q[Q > 0].flatten()), max_amp)

delta_0 = -5  
delta_f = -delta_0  
T = 10000  # Aumentato da 4000 a 10000 microsecondi perché il sistema 10x10 richiede evoluzioni più lente

adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]), 
    InterpolatedWaveform(T, [delta_0, 0, delta_f]), 
    0, 
)

seq = Sequence(reg, FRESNEL_DEVICE)
seq.declare_channel("ising", "rydberg_global")
seq.add(adiabatic_pulse, "ising")

# ==========================================
# 7. ESECUZIONE SIMULAZIONE (MyQLM)
# ==========================================
print("Esecuzione della simulazione adiabatica in corso (calcolo funzioni d'onda)...")
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS, modulation=MODULATION)

MyQLMPulserSimBackend = IsingAQPU.from_sequence(seq, qpu=None)
MYQLM_BACKEND = MyQLMPulserSimBackend if LOCAL_SIMULATIONS else AnalogQPU()

results = MYQLM_BACKEND.submit(job)

# ==========================================
# 8. VISUALIZZAZIONE RISULTATI QUANTISTICI
# ==========================================
def get_samples_from_result(result: Result):
    samples = {}
    n_qubits = len(qubits)
    for sample in result.raw_data:
        if len(sample.state.bitstring) > n_qubits:
            raise ValueError(f"State {sample.state} is incompatible.")
        counts = sample.probability
        samples[sample.state.bitstring.zfill(n_qubits)] = counts
    return samples

def plot_distribution(result: Result):
    C = get_samples_from_result(result)
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    
    # Prendiamo solo i top 20 risultati per non affollare troppo l'asse X (che ne avrebbe 1024)
    top_n = 20
    C_top = dict(list(C.items())[:top_n])
    
    # Usiamo le soluzioni ottimali trovate dal risolutore classico!
    color_dict = {key: "r" if key in true_optimal_states else "g" for key in C_top}
    
    plt.figure(figsize=(15, 6))
    plt.xlabel("Top 20 Bitstrings", fontsize=12)
    plt.ylabel("Probabilità di Misurazione", fontsize=12)
    plt.bar(C_top.keys(), C_top.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation=90)
    
    plt.title(f"QAA (10x10) - Embedding Genetico ({best_overall_name} | Err: {best_overall_error:.2f})", fontsize=14)
    plt.tight_layout()
    plt.show()

plot_distribution(results)