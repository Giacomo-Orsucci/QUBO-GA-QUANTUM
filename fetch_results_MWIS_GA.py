import os
import numpy as np
from qat.qlmaas.connection import QLMaaSConnection


def fetch_raw_results(job_ids, expected_qubits):
    print("\n--- RECUPERO RISULTATI QUANTISTICI GREZZI ---")
    try:
        connection = QLMaaSConnection()

    except ImportError:
        connection = QLMaaSConnection()

    
    for i, j_id in enumerate(job_ids):
        qubit_count = expected_qubits[i]
        print(f"\nAnalisi del Job: {j_id} (Qubit attesi: {qubit_count})...")
        try:
            res = connection.get_result(j_id)
            best_state = res[0].state
            
            # Estrazione testuale pura
            s = str(best_state).replace('|', '').replace('>', '').strip()
            
            # FIX INFORMATICO: Padding degli zeri iniziali (zfill)
            # Se myQLM restituisce "1", ma i qubit erano 5, diventa "00001"
            s_padded = s.zfill(qubit_count)
            bits = [int(b) for b in s_padded]
                
            print(f"  -> Bitstring Migliore : {bits}")
            print(f"  -> Nodi eccitati (1)  : {sum(bits)} su {qubit_count}")
            print(f"  -> Probabilità Misura : {res[0].probability:.4f}")
            
        except Exception as e:
            print(f"  [ERRORE] Impossibile recuperare {j_id}: {e}")

if __name__ == "__main__":
    # Quando rilanci MWIS_GA.py, segnati quanti atomi ci sono per ogni cella
    # (Lo leggi dal terminale durante la FASE 2)
    my_job_ids = [
        "SJob210108", 
        "SJob210109", 
        "SJob210110"
        
    ]
    # Sostituisci questi numeri con la dimensione reale di ogni cella
    qubits_per_job = [5,5,2] 
    
    fetch_raw_results(my_job_ids, qubits_per_job)