# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register
from pulser.devices import Device
from pulser.waveforms import InterpolatedWaveform
from scipy.spatial.distance import pdist, squareform
from qat.core import Result

from pulser_myqlm import FresnelQPU, IsingAQPU

#Variante del tutorial con Q 10x10 (10 qubits) ma che non fa uso dell'ottimizzatore classico (Nelder-Mead) per trovare le coordinate degli atomi, ma che invece genera direttamente delle coordinate "sicure" (rispettando i vincoli della QPU) e da queste ricava la matrice Q.
#E' un approccio inverso rispetto al tutorial, ma secondo me è più interessante per mostrare i limiti di QAA su scale più grandi. Parte della complessità è dovuta al fatto che soddisfare vincoli
#geometrici con 10 atomi è già molto complesso (proprio per come è costruito fisicamente l'hw non si possono mettere)
#atomi a distanza < 5um. Quello che mira ad evidenziare questa versione è che anche solo con 10 atomi la situazione si complica
#moltissimo e trovare la soluzione ottima diventa molto complesso a causa del numero di diverse soluzioni così vicine tra loro,
#soprattutto considerando che abbiamo atomici fisici molto vicini che rischiano di deragliare dal calcolo molto velocemente a causa dei livelli energetici tra loto molto vicini.
#A tal proposito il T max per il raggio laser (10000 ns) risulta troppo veloce.
#Infatti notiamo in maniera molto interessante come non si riesca a trovare la soluzione ottima classicamente calcolata.
#Questo mostra bene i limiti di QAA (algoritmo adiabatico) su scale più grandi.

# --- 1. SETUP HARDWARE E SIMULATORE ---
FRESNEL_QPU = FresnelQPU(None)  
FRESNEL_DEVICE = Device.from_abstract_repr(FRESNEL_QPU.get_specs().description)

LOCAL_SIMULATIONS = True
NBSHOTS = 0  
MODULATION = False

if not LOCAL_SIMULATIONS:
    try:
        from qlmaas.qpus import AnalogQPU
    except ImportError as e:
        raise ImportError(
            "Can't import AnalogQPU: simulations can only be performed locally using IsingAQPU."
        ) from e
    
if not LOCAL_SIMULATIONS and NBSHOTS > 0:
    raise ValueError("Simulation with AnalogQPU: number of shots must be 0.")

# --- 2. GENERAZIONE SICURA DEL QUBO 10x10 E EMBEDDING DIRETTO ---
# Bypassiamo l'ottimizzatore classico (Nelder-Mead) che su 10 atomi fallisce. 
# Creiamo posizioni sicure fin dall'inizio e ne deriviamo la matrice Q.

print("\n--- GENERAZIONE SICURA E EMBEDDING DIRETTO (10 ATOMI) ---")
np.random.seed(42)
coords_10 = []

# Generiamo 10 coordinate in uno spazio 25x25 garantendo la distanza minima di 5.5 µm
while len(coords_10) < 10:
    punto = np.random.uniform(0, 25, 2)
    if len(coords_10) == 0 or np.all(np.linalg.norm(np.array(coords_10) - punto, axis=1) >= 5.5):
        coords_10.append(punto)
        
coords_10 = np.array(coords_10)

# Costruiamo la matrice Q (100% fisicamente legale)
interazioni = squareform(FRESNEL_DEVICE.interaction_coeff / pdist(coords_10) ** 6)
Q = interazioni.copy()
np.fill_diagonal(Q, -5.0)

print("Matrice Q 10x10 generata con successo!")

# --- 3. RISOLUZIONE CLASSICA (FORZA BRUTA) ---
# Generiamo tutte le 1024 combinazioni possibili (2^10).
bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))] 
costs = []

for b in bitstrings:
    z = np.array(list(b), dtype=int)
    cost = z.T @ Q @ z 
    costs.append(cost) 

sort_zipped = sorted(zip(bitstrings, costs), key=lambda x: x[1])
print("Le 3 migliori soluzioni classiche (su 1024):")
for res in sort_zipped[:3]:
    print(res)

# Salviamo dinamicamente le 2 migliori soluzioni per colorarle di rosso nel grafico
best_solutions = [sort_zipped[0][0], sort_zipped[1][0]]

# --- 4. CREAZIONE DEL REGISTRO QUANTISTICO ---
# Usiamo direttamente le coordinate, chiamando i qubit con le stringhe 'q0', 'q1'...
qubits = {f"q{i}": c for i, c in enumerate(coords_10)}
reg = Register(qubits)

# Disegniamo la costellazione a 10 atomi!
reg.draw(
    blockade_radius=FRESNEL_DEVICE.rydberg_blockade_radius(1.0),
    draw_graph=False,
    draw_half_radius=True,
)

# --- 5. DEFINIZIONE IMPULSO ADIABATICO E ESECUZIONE ---
max_amp = FRESNEL_DEVICE.channels["rydberg_global"].max_amp
Omega = min(np.median(Q[Q > 0].flatten()), max_amp)  
delta_0 = -5  
delta_f = -delta_0  
T = 10000  

adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]), 
    InterpolatedWaveform(T, [delta_0, 0, delta_f]), 
    0, 
)

seq = Sequence(reg, FRESNEL_DEVICE)
seq.declare_channel("ising", "rydberg_global")
seq.add(adiabatic_pulse, "ising")

job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS, modulation=MODULATION)

MyQLMPulserSimBackend = IsingAQPU.from_sequence(seq, qpu=None)
MYQLM_BACKEND = MyQLMPulserSimBackend if LOCAL_SIMULATIONS else AnalogQPU()
results = MYQLM_BACKEND.submit(job)

# --- 6. LETTURA E VISUALIZZAZIONE RISULTATI ---
def get_samples_from_result(result: Result):
    samples = {}
    n_qubits = len(qubits)
    for sample in result.raw_data:
        counts = sample.probability
        samples[sample.state.bitstring.zfill(n_qubits)] = counts
    return samples

def plot_distribution(result: Result):
    C = get_samples_from_result(result)
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    
    # Usiamo le migliori soluzioni trovate dalla forza bruta invece di "101"
    color_dict = {key: "r" if key in best_solutions else "g" for key in C}
    
    plt.figure(figsize=(12, 6))
    plt.xlabel("bitstrings")
    plt.ylabel("rates")
    
    # Mostriamo solo le prime 30 stringhe per non affollare troppo il grafico (1024 sono tante!)
    top_30_keys = list(C.keys())[:30]
    top_30_values = list(C.values())[:30]
    top_30_colors = [color_dict[k] for k in top_30_keys]
    
    plt.bar(top_30_keys, top_30_values, width=0.5, color=top_30_colors)
    plt.xticks(rotation="vertical")
    plt.title("Distribuzione delle probabilità (Top 30 risultati)")
    plt.show()

plot_distribution(results)

# --- 7. VERIFICA TEMPO DI EVOLUZIONE VS COSTO ---
def get_cost_colouring(bitstring, Q_mat):
    z = np.array(list(bitstring), dtype=int) 
    cost = z.T @ Q_mat @ z
    return cost

def get_cost(result, Q_mat):
    counter = get_samples_from_result(result)
    cost = sum(counter[key] * get_cost_colouring(key, Q_mat) for key in counter)
    return cost / sum(counter.values())  

cost = []

print("Simulazione scansione tempi in corso (potrebbe richiedere qualche secondo)...")
for time_T in 1000 * np.linspace(1, 10, 10):
    seq_temp = Sequence(reg, FRESNEL_DEVICE)
    seq_temp.declare_channel("ising", "rydberg_global")
    temp_pulse = Pulse(
        InterpolatedWaveform(time_T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(time_T, [delta_0, 0, delta_f]),
        0,
    )
    seq_temp.add(temp_pulse, "ising")
    temp_job = IsingAQPU.convert_sequence_to_job(seq_temp, nbshots=NBSHOTS, modulation=MODULATION)
    temp_results = MYQLM_BACKEND.submit(temp_job)
    cost.append(get_cost(temp_results, Q) / 3)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), np.array(cost), "--o")
plt.xlabel("Tempo di evoluzione totale (µs)", fontsize=14)
plt.ylabel("Costo scalato", fontsize=14)
plt.title("Costo medio vs Tempo di Evoluzione (QAA)")
plt.show()


from scipy.optimize import minimize
import numpy as np

print("\n--- INIZIO OTTIMIZZAZIONE QAOA ---")

# 1. Definiamo l'architettura del QAOA
# Scegliamo p=2 (2 layer, quindi 4 colpi di laser in totale)
p_layers = 2  

def create_qaoa_sequence(t_params):
    """
    Costruisce la sequenza laser usando i tempi passati dall'ottimizzatore classico.
    t_params conterrà 4 valori (2 per ogni layer: t_cost e t_mixer).
    """
    seq = Sequence(reg, FRESNEL_DEVICE)
    seq.declare_channel("ising", "rydberg_global")
    
    # Vincolo Hardware: i tempi in Pulser devono essere multipli di 4ns e >= 16ns
    t_params = np.clip(np.round(t_params / 4) * 4, 16, 5000)
    
    for layer in range(p_layers):
        t_cost = int(t_params[2 * layer])
        t_mixer = int(t_params[2 * layer + 1])
        
        # Livello Costo: applichiamo il detuning (premio) per un tempo t_cost
        pulse_cost = Pulse.ConstantPulse(t_cost, 0.0, delta_f, 0.0)
        seq.add(pulse_cost, "ising")
        
        # Livello Mixer: applichiamo l'ampiezza (frullatore) per un tempo t_mixer
        pulse_mixer = Pulse.ConstantPulse(t_mixer, Omega, 0.0, 0.0)
        seq.add(pulse_mixer, "ising")
        
    return seq

# 2. Il "Cervello" Classico (Funzione Obiettivo)
iterazione = 0
def qaoa_objective(params):
    global iterazione
    iterazione += 1
    
    # Crea la sequenza con i parametri correnti
    seq_qaoa = create_qaoa_sequence(params)
    job_qaoa = IsingAQPU.convert_sequence_to_job(seq_qaoa, nbshots=NBSHOTS, modulation=MODULATION)
    
    # Esegue sulla QPU (o simulatore)
    results_qaoa = MYQLM_BACKEND.submit(job_qaoa)
    
    # Calcola il costo medio (valore atteso). Questo è ciò che vogliamo minimizzare!
    costo_medio = get_cost(results_qaoa, Q)
    
    if iterazione % 5 == 0:
        print(f"Iterazione {iterazione} | Costo medio: {costo_medio:.3f}")
        
    return costo_medio

# 3. Il Loop di Ottimizzazione
# Partiamo con durate casuali (es. tra 100 e 500 ns) per i nostri 4 impulsi
np.random.seed(42)
tempi_iniziali = np.random.uniform(100, 500, 2 * p_layers)

print("Avvio del ciclo ibrido quantistico-classico (COBYLA)...")
res_qaoa = minimize(
    qaoa_objective,
    tempi_iniziali,
    method="COBYLA", # COBYLA è ottimo per funzioni costo rumorose o quantistiche
    options={"maxiter": 60} # Facciamo massimo 60 tentativi per non aspettare troppo
)

print("\nOttimizzazione conclusa!")
print("I tempi ottimali trovati per i laser (in ns) sono:", np.round(res_qaoa.x))

# 4. Esecuzione finale con i parametri vincenti e Plot
seq_finale = create_qaoa_sequence(res_qaoa.x)
job_finale = IsingAQPU.convert_sequence_to_job(seq_finale, nbshots=NBSHOTS, modulation=MODULATION)
results_finali = MYQLM_BACKEND.submit(job_finale)

print("\n--- RISULTATO FINALE QAOA ---")
plot_distribution(results_finali)

#Anche qui questa variante mira soprattutto a mostrare i limiti di ottimizzazioni ibride classiche-quantistiche e come anche in questo caso
#i risultati non siano soddisfacenti.