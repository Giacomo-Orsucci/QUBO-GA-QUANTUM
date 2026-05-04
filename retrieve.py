import numpy as np
import matplotlib.pyplot as plt
from os import getenv
from qat.qlmaas.connection import QLMaaSConnection

# Questo script è progettato per recuperare i risultati di un job eseguito su Jülich, utilizzando il QLMaaSConnection.


# --- 1. JOB ID ---
JOB_ID = "SJob207598"

print(f"Connessione a Jülich in corso per cercare il Job: {JOB_ID}")
connection = QLMaaSConnection()

# --- 2. RECUPERO DEL JOB ---
try:

    # rincreando una connessione al server
    connection = QLMaaSConnection()

    if JOB_ID == "":
        raise RuntimeError("specify the environment variable JOB_ID")

    # check the status of the job
    status = connection.get_status(JOB_ID)

    # get the result of the job
    results = connection.get_result(JOB_ID)

    print(f"Stato attuale del Job: {status}")
    
except Exception as e:
    print(f"\nErrore durante il recupero. Assicurati che l'ID sia corretto. Dettagli: {e}")
    exit()

# --- 3. ESTRAZIONE E PLOT DEI RISULTATI ---
# Inserire la dimensione del problema (ad esempio 5 atomi)
n_qubits = 5

def get_samples_from_result(result, n_qubits):
    samples = {}
    for sample in result.raw_data:
        bitstring = sample.state.bitstring.zfill(n_qubits)
        samples[bitstring] = sample.probability
    return samples

C = get_samples_from_result(results, n_qubits)
C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))

# Inserire le soluzioni ottimali attese
indexes =  ["01011", "00111"]
color_dict = {key: "red" if key in indexes else "green" for key in C}

plt.figure(figsize=(12, 6))
plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
plt.xlabel("Configurazioni Misurate (Bitstrings)", fontsize=12)
plt.ylabel("Probabilità / Frequenza", fontsize=12)
plt.title(f"Risultati dell'Emulazione Remota (Job ID: {JOB_ID}...)", fontsize=14)
plt.xticks(rotation="vertical")
plt.tight_layout()
plt.show()

print("\nMetadati del Job:")
print(results.meta_data)