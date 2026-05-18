import numpy as np

#Very little script to save as .npz a precise matrix

# La tua matrice hardcodata
Q_5x5 = np.array([
    [-10.0, 19.7365809, 19.7365809,  5.42015853,  5.42015853],
    [19.7365809, -10.0, 20.67626392,  0.17675796,  0.85604541],
    [19.7365809, 20.67626392, -10.0,  0.85604541,  0.17675796],
    [ 5.42015853,  0.17675796,  0.85604541, -10.0,  0.32306662],
    [ 5.42015853,  0.85604541,  0.17675796,  0.32306662, -10.0]
])

# Estraiamo gli indici e i valori solo per il triangolo superiore (inclusa la diagonale)
# Questo alleggerisce il file ed è il formato standard per i QUBO simmetrici
i_idx, j_idx = np.where(np.triu(Q_5x5) != 0)
weights = Q_5x5[i_idx, j_idx]

# Salviamo il file usando le chiavi esatte che il tuo parser si aspetta: 'i', 'j', 'Jij'
nome_file = "tutorial_5x5.npz"
np.savez(nome_file, i=i_idx, j=j_idx, Jij=weights)

print(f"File '{nome_file}' generato con successo!")