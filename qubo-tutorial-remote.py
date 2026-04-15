import numpy as np
from pulser import InterpolatedWaveform, Pulse, Sequence, Register
from pulser.devices import MockDevice
from pulser_myqlm import IsingAQPU
from qat.qlmaas.connection import QLMaaSConnection
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#(Tutorial: https://github.com/pasqal-io/Pulser-myQLM/blob/main/tutorials/QAOA%20and%20QAA%20to%20solve%20a%20QUBO%20problem.ipynb)
#MODIFICHE/AGGIUNTE: Variante per l'esecuzione su infrastruttura remota HPC JULIC. Al momento settata per eseguire su AnalogQPU, Jade in trasferimento in altra struttura.

#NOTA BENE: In questa fase lo script, a scopo di test, è scritto in maniera tale da aspettare asincronicamente i risultati,
#ma nulla vieta di essere interrotto dopo l'invio e recuperarli in un secondo momento tramite id.

# 1. & 2. CONNESSIONE E SELEZIONE EMULATORE (Metodo Robusto)
print("Inizializzazione dell'emulatore remoto AnalogQPU...")

try:
    # Tenta il caricamento diretto tramite il wrapper di Jülich (come facevi per Jade)
    from qlmaas.qpus import AnalogQPU
    qpu_emulator = AnalogQPU()
    print("Emulatore caricato tramite qlmaas.qpus!")
except ImportError:
    # Se fallisce, usa la classe base standard di myQLM che sappiamo essere presente
    from qat.qlmaas.qpus import QLMaaSQPU
    qpu_emulator = QLMaaSQPU("qat.qpus:AnalogQPU")
    print("Emulatore caricato tramite classe base QLMaaSQPU!")

# 3. DEFINIZIONE DEVICE ASTRATTO
# Usiamo MockDevice che è un template generico per atomi neutri
device = MockDevice

# 2. DEFINIZIONE DEL PROBLEMA QUBO
Q = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)


# 3. MAPPATURA SPAZIALE 
def evaluate_mapping(new_coords, *args):
    Q, shape = args
    new_coords = np.reshape(new_coords, shape)
    # Calcolo usando il VERO coefficiente del simulatore/macchina
    new_Q = squareform(device.interaction_coeff / pdist(new_coords) ** 6)
    return np.linalg.norm(new_Q - Q)


shape = (len(Q), 2)
np.random.seed(0)
x0 = np.random.random(shape).flatten()

print("Ottimizzazione classica delle coordinate degli atomi...")
res = minimize(
    evaluate_mapping,
    x0,
    args=(Q, shape),
    method="Nelder-Mead",
    tol=1e-6,
    options={"maxiter": 200000},
)

coords = np.reshape(res.x, (len(Q), 2))
qubits = {f"q{i}": coord for i, coord in enumerate(coords)}

# Definizione del Registro.
# NOTA: Pulser verificherà in automatico se le distanze rispettano device.min_atom_distance
reg = Register(qubits)

# 4. CREAZIONE DEGLI IMPULSI LASER ADIABATICI
# Vogliamo che l'Omega sia proporzionale alla matrice Q, MA non deve superare il limite fisico quando applicato a macchina reale e non simulata
ideal_omega = np.median(Q[Q > 0].flatten())

# Controlliamo se il device ha un limite fisico (il MockDevice restituisce None)
channel_max_amp = device.channels["rydberg_global"].max_amp
if channel_max_amp is not None:
    max_omega = channel_max_amp / 1.2
    Omega = min(ideal_omega, max_omega)
else:
    # Se non c'è limite, usiamo l'Omega ideale senza restrizioni
    Omega = ideal_omega


delta_0 = -5.0
delta_f = -delta_0
duration_us = 4  # microsecondi
T = duration_us * 1000  # conversione in nanosecondi (Pulser ragiona in ns)

adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
    InterpolatedWaveform(T, [delta_0, 0, delta_f]),
    0,
)

# Costruiamo la sequenza 
seq = Sequence(reg, device)
seq.declare_channel("ising", "rydberg_global")
seq.add(adiabatic_pulse, "ising")

# 5. CONVERSIONE E SOTTOMISSIONE DEL JOB
NBSHOTS = 0  # Eseguiamo 10 misurazioni fisiche. Intanto 10, per provare e per questioni di budget.
print(f"Preparazione del Job quantistico con {NBSHOTS} shots...")
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS)

print("Invio all'emulatore remoto (AnalogQPU) tramite QLMaaS...")
async_results = qpu_emulator.submit(job)

# Usiamo .join() perché la chiamata è tornata asincrona
print("Job in coda sul cluster. In attesa dei risultati...")
results = async_results.join() 
print("Esecuzione completata!")


# 6. RECUPERO E PLOT DEI RISULTATI
# Usiamo la funzione originale del tutorial per estrarre le bitstringhe dal formato myQLM Result
def get_samples_from_result(result, n_qubits):
    samples = {}
    for sample in result.raw_data:
        # Assicuriamoci che la stringa abbia la lunghezza corretta (5 atomi = 5 bit)
        bitstring = sample.state.bitstring.zfill(n_qubits)
        samples[bitstring] = sample.probability
    return samples


# Estrazione e ordinamento
C = get_samples_from_result(results, len(Q))
C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))

# Plotting
indexes = ["01011", "00111"]  # Le soluzioni teoriche ottimali del QUBO
color_dict = {key: "red" if key in indexes else "green" for key in C}

plt.figure(figsize=(12, 6))
plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
plt.xlabel("Configurazioni Misurate (Bitstrings)", fontsize=12)
plt.ylabel("Probabilità / Frequenza", fontsize=12)
plt.title("Risultati dell'Emulazione Remota (Device: MockDevice, Solver: AnalogQPU)", fontsize=14)
plt.xticks(rotation="vertical")
plt.tight_layout()
plt.show()

# Possiamo anche ispezionare i metadati restituiti dalla QPU (es. tempi esatti di esecuzione, code)
print("\nMetadati del Job:")
print(results.meta_data)
