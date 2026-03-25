import numpy as np
from pulser import InterpolatedWaveform, Pulse, Sequence
from pulser.devices import Device
from pulser_myqlm import IsingAQPU


from os import getenv
from qat.qlmaas.connection import QLMaaSConnection

connection = QLMaaSConnection()

from qlmaas.qpus import JadeQPU as QPU
qpu = connection.get_qpu("qat.qpus:JadeQPU")()


# Build device and register from QPU specs
specs = qpu.get_specs()
device = Device.from_abstract_repr(specs.description)

# 2. DEFINIZIONE DEL PROBLEMA QUBO
Q = np.array([
    [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
    [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
    [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
    [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
    [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
])

# 3. MAPPATURA SPAZIALE (Usando i parametri di Jade)
def evaluate_mapping(new_coords, *args):
    Q, shape = args
    new_coords = np.reshape(new_coords, shape)
    # Calcolo usando il VERO coefficiente di Van der Waals di Jade
    new_Q = squareform(jade_device.interaction_coeff / pdist(new_coords) ** 6)
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
qubits = dict(enumerate(coords))

# Definizione del Registro. 
# NOTA: Pulser verificherà in automatico se le distanze rispettano jade_device.min_atom_distance
reg = Register(qubits)

# 4. CREAZIONE DEGLI IMPULSI LASER ADIABATICI
# Vogliamo che l'Omega sia proporzionale alla matrice Q, MA non deve superare il limite fisico di Jade
ideal_omega = np.median(Q[Q > 0].flatten())
max_jade_omega = jade_device.max_amp / 1.2  # Ci teniamo un margine di sicurezza
Omega = min(ideal_omega, max_jade_omega)

delta_0 = -5.0
delta_f = -delta_0
duration_us = 4  # microsecondi
T = duration_us * 1000  # conversione in nanosecondi (Pulser ragiona in ns)

adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
    InterpolatedWaveform(T, [delta_0, 0, delta_f]),
    0,
)

# Costruiamo la sequenza per Jade
seq = Sequence(reg, jade_device)
seq.declare_channel("ising", "rydberg_global")
seq.add(adiabatic_pulse, "ising")

# 5. CONVERSIONE E SOTTOMISSIONE DEL JOB
NBSHOTS = 10 # Eseguiamo 10 misurazioni fisiche. Intanto 10, per provare e per questioni di budget.
print(f"Preparazione del Job quantistico con {NBSHOTS} shots...")
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS)

print("Invio alla coda di Jade...")
async_results = qpu.submit(job)

# A questo punto il job è nel sistema. L'infrastruttura di JUNIQ ci fornisce 
# un oggetto asincrono. Usando .join(), il nostro script si mette in pausa
# e aspetta pazientemente che Jade faccia il suo lavoro.
print("Job sottomesso! In attesa dei risultati dalla QPU (potrebbe volerci del tempo)...")
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
plt.title("Risultati del Calcolo Ibrido su Jade (Pasqal QPU)", fontsize=14)
plt.xticks(rotation="vertical")
plt.tight_layout()
plt.show()

# Possiamo anche ispezionare i metadati restituiti dalla QPU (es. tempi esatti di esecuzione, code)
print("\nMetadati del Job:")
print(results.meta_data)