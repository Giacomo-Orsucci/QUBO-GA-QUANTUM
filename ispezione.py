import numpy as np

# Metti il percorso corretto al primo file
file_path = "./qubo-bench/qubo-benchmark-main/generate/compsup/instances/2d_(4, 4)_precision256/seed00.npz"

try:
    with np.load(file_path, allow_pickle=True) as data:
        print(f"File caricato con successo!")
        print(f"Chiavi presenti nell'archivio: {data.files}\n")
        
        for key in data.files:
            array_data = data[key]
            print(f"--- Chiave: '{key}' ---")
            print(f"Tipo: {type(array_data)}")
            print(f"Dimensioni (Shape): {array_data.shape}")
            # Stampiamo i primi 5 elementi per capire cosa sono
            if array_data.size > 0:
                print(f"Primi elementi: {array_data.flatten()[:5]}\n")
            else:
                print("Array vuoto\n")
except Exception as e:
    print(f"Errore: {e}")