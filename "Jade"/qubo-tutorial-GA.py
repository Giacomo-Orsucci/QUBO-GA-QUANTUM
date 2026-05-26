import numpy as np
import pygad
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register
from pulser.devices import Device
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from qat.core import Result

from pulser_myqlm import FresnelQPU, IsingAQPU

"""
Questo file contiene la variante del problema QUBO del tutorial originale, ma risolto con un algoritmo genetico GA.
L'idea è di, modificando il problema stesso, familiarizzare con la struttura di un algoritmo genetico, e poi fare un'analisi di sensibilità sui parametri più importanti (popolazione e mutazione) per capire come influenzano la convergenza.
Una volta stressato il problema e familiarizzato con esso mediante GA, l'idea è di vedere come si comporta su di un problema
10x10 e superiore, che sono dimensioni per cui Nelder-Mead e altri algoritmi classici diventano impraticabili, mentre GA e altri meta-heuristici possono ancora dare risultati interessanti.
Questa prima versione fa uso di coordinate continue, sicuramente dobbiamo passare ad una griglia (lattice) soprattutto al crescere del numero di atomi e quindi, dello spazio di ricerca.
Si stampa l'errore medio su 10 esecuzioni, ma si prende la migliore per continuare con la parte quantistica 
(ancora da implementare).
Simuliamo senza rumore, quello che vogliamo quantificare direttamente su Jade.
"""


# Creiamo una rappresentazione virtuale della QPU Fresnel di Pasqal.
# Passando "None", evitiamo la connessione ai server cloud.
FRESNEL_QPU = FresnelQPU(None)  

# Estraiamo le specifiche fisiche (potenza massima laser, raggio di Rydberg, ecc.).
# Il simulatore userà questo oggetto per impedirci di programmare sequenze impossibili in natura.
FRESNEL_DEVICE = Device.from_abstract_repr(FRESNEL_QPU.get_specs().description)

# Assicuriamo che i calcoli in locale.
LOCAL_SIMULATIONS = True

# NBSHOTS = 0 indica che in locale il simulatore eseguirà internamente 2000 misurazioni
# per costruire la distribuzione statistica finale dei risultati. MODULATION=False mantiene le onde del laser perfette e ideali.
NBSHOTS = 0  
MODULATION = False


# Checking AnalogQPU can be imported
if not LOCAL_SIMULATIONS:
    try:
        from qlmaas.qpus import AnalogQPU
    except ImportError as e:
        raise ImportError(
            "Can't import AnalogQPU: simulations can only be performed locally using IsingAQPU (uses pulser-simulation)."
        ) from e
    

if not LOCAL_SIMULATIONS and NBSHOTS > 0:
    raise ValueError("Simulation with AnalogQPU: number of shots must be 0.")

# ==========================================
# 1. SETUP DEL PROBLEMA (Matrice Q 5x5)
# ==========================================
Q = np.array([
    [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
    [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
    [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
    [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
    [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0]
])

N_ATOMS = len(Q)
SPACE_BOUND = 15.0      
MIN_DISTANCE = 5.0 #vincolo legato alla macchina reale (non possiamo mettere due atomi troppo vicini, altrimenti si fondono in un super-atomo e il modello fisico non regge più)      
C6_COEFF = FRESNEL_DEVICE.interaction_coeff  # Parametro di FRESNEL_DEVICE
NUM_GENERATIONS = 400  # Fissato per avere un confronto equo sul tempo

# ==========================================
# 2. FUNZIONI GENETICHE E FISICHE
# ==========================================

#crossover custom che assicura che i geni di un atomo (x e y) rimangano sempre insieme, evitando combinazioni non fisiche
#i parametri sono i 3 classici per una funzione custom di crossover in PyGAD:
#- parents: matrice degli individui genitori selezionati per la riproduzione
#- offspring_size: tupla che indica la dimensione della matrice degli individui figli da generare (numero figli, numero geni)
#- ga_instance: l'istanza dell'algoritmo genetico.
def atom_aware_crossover(parents, offspring_size, ga_instance):

    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):#in posizione 0 abbiamo il numero di figli da generare

        #si scelgono i genitori da far accoppiare per creare i figli in modo circolare
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0] #come secondo genitore si prende il successivo al precedente

        for atom_idx in range(N_ATOMS): #si itera sugli atomi e si trovano le coordinate x e y corrispondenti a quell'atomo (ricordiamo che ogni atomo è rappresentato da 2 geni: x e y)

            x_idx = atom_idx * 2
            y_idx = x_idx + 1

            #ispirandosi ad un lancio di moneta Mendeliano, non avendo particolari necessità di sbilanciamento
            #nella probabilità di scelta dei geni dei genitori, si decide casualmente se prendere le coordinate x e y dell'atomo dal primo o dal secondo genitore, ma sempre insieme per mantenere la coerenza fisica
            # x_idx:y_idx+1 serve a prendere sia la coordinata x che quella y dell'atomo, assicurando che non vengano mischiate tra genitori diversi. +1 perchè l'ultimo è sempre escluso.
            if np.random.rand() > 0.5:
                offspring[k, x_idx:y_idx+1] = parents[parent1_idx, x_idx:y_idx+1]
            else:
                offspring[k, x_idx:y_idx+1] = parents[parent2_idx, x_idx:y_idx+1]
    return offspring


#L'idea è di avere una fitness function che misuri l'aderenza della mappatura trovata con GA con la matrice Q
#del problema, ma che allo stesso tempo penalizzi fortemente le soluzioni che violano il vincolo di distanza minima tra atomi (MIN_DISTANCE), che è un vincolo fisico reale della macchina su cui poi vorremmo implementare la soluzione trovata.
def fitness_func(ga_instance, solution, solution_idx):

    coords_temp = np.reshape(solution, (N_ATOMS, 2))
    distances = pdist(coords_temp)
    
    if np.min(distances) < MIN_DISTANCE:
        return -999999.0  
        
    Q_fisica = squareform(C6_COEFF / (distances ** 6))
    np.fill_diagonal(Q_fisica, -10.0)
    
    errore = np.linalg.norm(Q_fisica - Q)
    return -errore #ritornato con il meno perché PyGAD massimizza la fitness, mentre noi vogliamo minimizzare l'errore. In questo modo, più la soluzione è vicina alla matrice Q desiderata, più la fitness sarà alta (meno negativa).

# ==========================================
# 3. GRIGLIA DI TEST (SENSITIVITY ANALYSIS)
# ==========================================
# Definiamo i parametri da far scontrare
populations = [50, 100, 200, 400, 800, 1000]      
mutations_percentage = [1, 2, 3, 5, 15, 30, 50]       # 30% - 50% caos, si va a distruggere il processo di ereditarietà delle caratteristiche buone. 
 
REPETITIONS = 10  # Numero di esecuzioni per ogni combinazione di parametri, per avere una stima più robusta della performance media e della variabilità del processo evolutivo.
# I pretests sopra riportati hanno mostrato che, per questo specifico problema, a meno della stocasticità del processo
# la popolazione migliore è sui 200 individui con percetuale di mutazione del 3%. Conto poi di fare altri test
# con i valori sopra e mediando su 10 esecuzioni ciascuno, ma intanto sovrascrivo con i dati migliori ad ora trovati
# per sperimentare con la parte quantistica.

populations = [50, 100, 200]      
mutations_percentage = [1, 2, 3, 5]

num_genes = N_ATOMS * 2 #La singola coordinata di un atomo è rappresentata da 2 geni (x e y).

#così definiamo il dizionario del range di valori per ogni gene. Viene usato sia in creazione che 
#durante l'evoluzione.
gene_space = [{'low': 0.0, 'high': SPACE_BOUND} for _ in range(num_genes)]

# Dizionario per accumulare TUTTE le 10 storie di ogni test
raw_results = {f"Pop: {pop} | Mut: {mut}%": [] for pop in populations for mut in mutations_percentage}

# --- TRACKER DEL CAMPIONE ASSOLUTO ---
# Per la parte quantistica (Pulser)
best_overall_error = float('inf')
best_overall_coords = None
best_overall_name = ""

print(f"--- INIZIO GRID SEARCH (Medie su {REPETITIONS} esecuzioni) ---")
tot_runs = len(populations) * len(mutations_percentage) * REPETITIONS
print(f"Esecuzioni totali previste: {tot_runs}\n")

test_counter = 1
for pop in populations:
    for mut in mutations_percentage:
        test_name = f"Pop: {pop} | Mut: {mut}%"
        print(f"Test in corso: {test_name} (attendere {REPETITIONS} run)...")
        
       
            
        # Pressione selettiva: facciamo riprodurre sempre il 20% della popolazione
        # In letteratura si indica una percetuale tra il 10% e il 30% come buona, ma dipende molto dal problema specifico. 
        # Qui scegliamo il 20% come compromesso per mantenere una buona diversità genetica senza diluire troppo la qualità dei genitori selezionati.
        num_parents = max(2, int(pop * 0.2)) 

        for rep in range(REPETITIONS):
            
            # Inizializziamo una nuova istanza PyGAD "pulita" per ogni test
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
            
            # Estraiamo l'errore e salviamo la storia di questa specifica run
            error_history = [abs(fitness) for fitness in ga_instance.best_solutions_fitness]
            raw_results[test_name].append(error_history)
            
            # Estraiamo il miglior risultato della run
            best_sol, best_fit, _ = ga_instance.best_solution()
            current_error = abs(best_fit)
            
            # --- AGGIORNAMENTO CAMPIONE ASSOLUTO ---
            if current_error < best_overall_error:
                best_overall_error = current_error
                best_overall_coords = np.reshape(best_sol, (N_ATOMS, 2))
                best_overall_name = test_name
            
            test_counter += 1

print(f"\n--- RICERCA COMPLETATA ---")
print(f"IL VINCITORE ASSOLUTO È: {best_overall_name}")
print(f"Errore minimo raggiunto: {best_overall_error:.4f}")

# ==========================================
# 4. VISUALIZZAZIONE DEI RISULTATI (Grafico delle Medie)
# ==========================================
plt.figure(figsize=(16, 10))

# Calcoliamo la media delle 10 esecuzioni per ogni configurazione
for test_name, history_list in raw_results.items():
    # history_list contiene 10 liste. Con np.mean(axis=0) facciamo la media colonna per colonna (generazione per generazione)
    avg_history = np.mean(history_list, axis=0)
    
    # Filtriamo i valori iniziali troppo alti per non rovinare la scala del grafico
    filtered_avg_history = [min(err, 50) for err in avg_history] 
    
    # Evidenziamo il vincitore
    linewidth = 3 if test_name == best_overall_name else 1.5
    
    plt.plot(filtered_avg_history, label=test_name, linewidth=linewidth)

plt.title(f"Analisi di Sensibilità GA (Media su {REPETITIONS} esecuzioni)", fontsize=16)
plt.xlabel("Generazione", fontsize=14)
plt.ylabel("Errore Medio", fontsize=14)
plt.legend(fontsize=10, loc="upper right", ncol=2)
plt.grid(True, linestyle="--", alpha=0.7)
plt.ylim(0, 30) 
plt.tight_layout()
plt.show()

print("--- RICERCA COMPLETATA ---")

# ==========================================
# 5. PASSAGGIO ALLA FASE QUANTISTICA (Pulser)
# ==========================================
print(f"\n--- AVVIO SIMULAZIONE QUANTISTICA (QAA) ---")
print(f"Utilizzo la migliore configurazione trovata ({best_overall_name}) con errore {best_overall_error:.4f}")

# Creiamo il dizionario dei qubit usando stringhe come chiavi ("q0", "q1", ...) per Pulser
# best_overall_coords contiene le (X,Y) perfette calcolate dal GA
qubits = {f"q{i}": coord for i, coord in enumerate(best_overall_coords)}
reg = Register(qubits)

# Disegniamo il registro spaziale trovato dal GA
print("Visualizzazione del layout spaziale degli atomi...")
reg.draw(
    blockade_radius=FRESNEL_DEVICE.rydberg_blockade_radius(1.0),
    draw_graph=False,
    draw_half_radius=True,
)

# ==========================================
# 6. DEFINIZIONE DELL'IMPULSO ADIABATICO
# ==========================================
# Calcoliamo l'intensità del laser (Omega) basandoci sulla matrice Q,
# assicurandoci di non superare il limite fisico dell'hardware FRESNEL.
max_amp = FRESNEL_DEVICE.channels["rydberg_global"].max_amp
Omega = min(np.median(Q[Q > 0].flatten()), max_amp)

delta_0 = -5  
delta_f = -delta_0  
T = 4000  # 4 microsecondi (tempo di evoluzione adiabatico)

# Creazione della forma d'onda a campana per evitare shock energetici
adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]), 
    InterpolatedWaveform(T, [delta_0, 0, delta_f]), 
    0, 
)

# Inizializziamo la sequenza e assegniamo il laser globale
seq = Sequence(reg, FRESNEL_DEVICE)
seq.declare_channel("ising", "rydberg_global")
seq.add(adiabatic_pulse, "ising")

# ==========================================
# 7. ESECUZIONE SIMULAZIONE (MyQLM)
# ==========================================
print("Esecuzione della simulazione adiabatica in corso (calcolo funzioni d'onda)...")
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS, modulation=MODULATION)

# Configurazione del backend locale
MyQLMPulserSimBackend = IsingAQPU.from_sequence(seq, qpu=None)
MYQLM_BACKEND = MyQLMPulserSimBackend if LOCAL_SIMULATIONS else AnalogQPU()

results = MYQLM_BACKEND.submit(job)

# ==========================================
# 8. VISUALIZZAZIONE RISULTATI QUANTISTICI
# ==========================================
def get_samples_from_result(result: Result):
    """Estrae le probabilità degli stati quantistici dal risultato MyQLM"""
    samples = {}
    n_qubits = len(qubits)
    for sample in result.raw_data:
        if len(sample.state.bitstring) > n_qubits:
            raise ValueError(f"State {sample.state} is incompatible.")
        counts = sample.probability
        samples[sample.state.bitstring.zfill(n_qubits)] = counts
    return samples

def plot_distribution(result: Result):
    """Plotta l'istogramma colorando di rosso le soluzioni corrette"""
    C = get_samples_from_result(result)
    
    # Ordiniamo i risultati per probabilità decrescente
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    
    # Le due soluzioni ottimali note del nostro problema QUBO. Specifiche per la nostra matrice Q.
    indexes = ["01011", "00111"]  
    color_dict = {key: "r" if key in indexes else "g" for key in C}
    
    plt.figure(figsize=(12, 6))
    plt.xlabel("Bitstrings (Stati Quantistici Finali)", fontsize=12)
    plt.ylabel("Probabilità di Misurazione", fontsize=12)
    plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    
    # Inseriamo nel titolo l'info sull'embedding genetico usato
    plt.title(f"Distribuzione QAA - Basata su Embedding Genetico ({best_overall_name} | Err: {best_overall_error:.2f})", fontsize=14)
    plt.tight_layout()
    plt.show()

plot_distribution(results)
