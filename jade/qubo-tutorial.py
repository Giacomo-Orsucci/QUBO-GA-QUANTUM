import numpy as np
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register
from pulser.devices import Device
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from qat.core import Result

from pulser_myqlm import FresnelQPU, IsingAQPU

#(Tutorial: https://github.com/pasqal-io/Pulser-myQLM/blob/main/tutorials/QAOA%20and%20QAA%20to%20solve%20a%20QUBO%20problem.ipynb)
#MODIFICHE/AGGIUNTE: ottimizzazione eseguita 10 volte per mitigare leggermente la sensibilità di Nelder-Mead ai minimi locali
#e confrontare l'errore (e quindi la soluzione migliore) con quello ottenuto grazie a GA.


# Creiamo una rappresentazione virtuale della QPU Fresnel di Pasqal.
# Passando "None", evitiamo la connessione ai server cloud.
FRESNEL_QPU = FresnelQPU(None)  

# Estraiamo le specifiche fisiche (potenza massima laser, raggio di Rydberg, ecc.).
# Il simulatore userà questo oggetto per impedirci di programmare sequenze impossibili in natura.
FRESNEL_DEVICE = Device.from_abstract_repr(FRESNEL_QPU.get_specs().description)

# Assicuriamo che i calcoli siano eseguiti in locale o meno.
LOCAL_SIMULATIONS = False

# NBSHOTS = 0 indica che in locale il simulatore eseguirà internamente 2000 misurazioni
# per costruire la distribuzione statistica finale. MODULATION=False mantiene le onde del laser perfette e ideali.
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

# La matrice Q rappresenta i nodi (sulla diagonale) e gli archi/penalità (fuori dalla diagonale).
# I valori sulla diagonale sono tutti uguali (-10.0), il che ci permetterà di usare un singolo laser globale.

Q = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)

#And we can check optimal solutions (exponential cost in Q dimension; brute force):

# Generiamo tutte le 32 combinazioni possibili (2^5). 
# Questa riga evidenzia il limite classico: per N atomi, il tempo cresce esponenzialmente.
bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))] #generate all possible bitstrings of length equal to the dimension of Q
costs = []


# this takes exponential time with the dimension of the QUBO
for b in bitstrings:
    z = np.array(list(b), dtype=int)
    # Calcoliamo l'energia della configurazione moltiplicando lo stato per la matrice delle regole.
    cost = z.T @ Q @ z #then compute the cost for each bitstring using the QUBO formula: z^T Q z
    costs.append(cost) #costs represent the value of the objective function, bute also the minimum energy of the system for that configuration of bits



# Ordiniamo e stampiamo. I risultati ci diranno che 01011 e 00111 sono gli stati di minima energia.
zipped = zip(bitstrings, costs)
sort_zipped = sorted(zipped, key=lambda x: x[1])
print(sort_zipped[:3])

def evaluate_mapping(new_coords, *args):
    """Cost function to minimize. Ideally, the pairwise distances are conserved."""
    Q, shape = args
    new_coords = np.reshape(new_coords, shape)

    # Questa è la legge fisica fondamentale: l'interazione di van der Waals decresce 
    # con la sesta potenza della distanza (pdist(new_coords) ** 6).
    new_Q = squareform(FRESNEL_DEVICE.interaction_coeff / pdist(new_coords) ** 6)
    # Restituisce la differenza tra la fisica (new_Q) e la matematica (Q). 
    # Vogliamo che questo valore tenda a zero.
    return np.linalg.norm(new_Q - Q) #to test if the "quantistic" atoms disposure reflect the original matrix.
# high value returned -> bad disposure

shape = (len(Q), 2)
costs = []
np.random.seed(0)

best_error = float('inf')
best_coords = None
best_res = None

for i in range(10):  # Run the optimization multiple times to mitigate local minima
    # Partiamo da una disposizione di coordinate 2D casuale ad ogni iterazione
    x0 = np.random.random(shape).flatten()
    # L'algoritmo classico Nelder-Mead sposta gli atomi nello spazio virtuale fino a minimizzare
    # l'errore calcolato dalla funzione evaluate_mapping.
    res = minimize(
        evaluate_mapping,
        x0,
        args=(Q, shape),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None},
    )

    # res.fun contiene il valore restituito da evaluate_mapping (il nostro errore)
    current_error = res.fun
    print(f"Iterazione {i+1} - Errore: {current_error:.4f}")

    # Se troviamo un errore minore, aggiorniamo la nostra migliore soluzione
    if current_error < best_error:
        best_error = current_error
        best_res = res
        best_coords = np.reshape(res.x, (len(Q), 2))

print(f"\nOttimizzazione completata. Miglior errore trovato: {best_error:.4f}")

# Registriamo le coordinate VINCENTI nel Registro fisico della macchina
coords = best_coords
qubits = dict(enumerate(coords))
reg = Register(qubits)


# Disegniamo il registro. draw_half_radius=True mostra visivamente il Blocco di Rydberg:
# se due aloni si sovrappongono, gli atomi non potranno eccitarsi simultaneamente.
reg.draw(
    blockade_radius=FRESNEL_DEVICE.rydberg_blockade_radius(1.0),
    draw_graph=False,
    draw_half_radius=True,
)

#THE ATOMS ARE NOW SET, WE CAN NOW DEFINE PARAMETERS FOR THE ADIABATIC PULSE
#(Vedere equazione sezione 2 del tutorial).

# Omega è l'intensità del laser (frequenza di Rabi, ampiezza). Usiamo la mediana dei vincoli
# per assicurarci che il laser abbia una forza paragonabile a quella del blocco di Rydberg.
Omega = np.median(Q[Q > 0].flatten())
# Detuning (frequenza relativa del laser). 
# Parte da un valore negativo (penalizza l'eccitazione degli atomi)...
delta_0 = -5  # just has to be negative
# ...e arriva a un valore positivo (premia l'eccitazione degli atomi). Questo perchè all'inizio vogliamo che il sistema sia nel suo stato fondamentale (tutti gli atomi a terra), e alla fine vogliamo che il sistema sia in uno stato eccitato che corrisponde alla soluzione del nostro problema di ottimizzazione.
delta_f = -delta_0  # just has to be positive
# Durata del processo (4 microsecondi). Abbastanza lungo da permettere 
# al sistema di adattarsi adiabaticamente senza "rompere" lo stato di minima energia.
T = 4000  # time in ns, we choose a time long enough to ensure the propagation of information in the system


# Costruiamo la forma d'onda del laser nel tempo.
adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]), #Partiamo da 1e-9, si sale fino a Omega e si riscende a 1e-9. Questa forma a campana permette di avere di non dare uno schock energetico agli atomi.
    InterpolatedWaveform(T, [delta_0, 0, delta_f]), #frequenza del laser, che parte negativa e sale fino ad essere positiva. Questo perchè all'inizio vogliamo penalizzare l'accesione degli atomi che in seguito vogliamo invece premiare.
    0, #Fase, in questo caso non è importante, quindi la lasciamo a zero.
)


# Inseriamo l'impulso nel canale "rydberg_global", il che significa che il laser 
# colpirà tutti gli atomi contemporaneamente, sfruttando l'omogeneità della diagonale di Q.
seq = Sequence(reg, FRESNEL_DEVICE)
seq.declare_channel("ising", "rydberg_global")
seq.add(adiabatic_pulse, "ising")
seq.draw()

# Convertiamo la partitura in un job per l'infrastruttura QLM.
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS, modulation=MODULATION)

# Scegliamo il backend. In questo caso userà MyQLMPulserSimBackend sul PC locale.
MyQLMPulserSimBackend = IsingAQPU.from_sequence(seq, qpu=None)
MYQLM_BACKEND = MyQLMPulserSimBackend if LOCAL_SIMULATIONS else AnalogQPU()


results = MYQLM_BACKEND.submit(job)


def get_samples_from_result(result: Result):
    """Converting the MyQLM Results into Pulser Samples"""
    samples = {}
    n_qubits = len(qubits)
    for sample in result.raw_data:
        if len(sample.state.bitstring) > n_qubits:
            raise ValueError(
                f"State {sample.state} is incompatible with number of qubits"
                f" declared {n_qubits}."
            )
        counts = sample.probability
        samples[sample.state.bitstring.zfill(n_qubits)] = counts
    return samples


def plot_distribution(result: Result):
    """
    Genera l'istogramma dei risultati. Conosce già quali sono le soluzioni corrette 
    ("01011", "00111") e istruisce il grafico a colorarle di rosso ("r"), 
    lasciando in verde ("g") le configurazioni sub-ottimali.
    """
    C = get_samples_from_result(result)
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    indexes = ["01011", "00111"]  # QUBO solutions
    color_dict = {key: "r" if key in indexes else "g" for key in C}
    plt.figure(figsize=(12, 6))
    plt.xlabel("bitstrings")
    plt.ylabel("rates")
    plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    plt.show()


plot_distribution(results)

#Per garantire che il sistema rimanga nello stato di minima energia durante tutta l'evoluzione, è fondamentale scegliere una durata T sufficientemente lunga. Se T è troppo breve, il sistema potrebbe non avere il tempo di adattarsi adiabaticamente, rischiando di "rompere" lo stato di minima energia e finire in uno stato eccitato indesiderato. In questo caso, abbiamo scelto T = 4000 ns, che è abbastanza lungo per permettere al sistema di evolversi senza subire transizioni non desiderate.
#L'unico problema è che nel mondo reale il T scelto è troppo lungo a causa del rumore ambientale e termico.
#Si passa dunque al QAOA, Quantum Approximate Optimization Algorithm, che è una versione "digitale" dell'approccio adiabatico, e permette di eseguire l'ottimizzazione in tempi più brevi, riducendo così l'impatto del rumore.
#L'idea è di non usare un unico impulso lungo e lento, ma impulsi più brevi e parametrizzati. Questo modo di agire difficilmente però ci porta direttamente al primo colpo alla soluzione perfetta.
# Quindi, si parametrizza casualmente la QPU, la quale esegue ed effettura la misurazione mediante la quale il computer classico va a misurare la qualità del risultato
# (z^T Q z) e calcola il costo medio delle strinche calcolate, se tale costo è alto la parametrizzazione non era corretta. Si usa dunque un classico algoritmo di ottimizzazione per aggiustare il tiro sui parametri,
# e si riesegue sulla QPU. 
# 
# IMPORTANTE:
# 
# Tutto giusto quello scritto sopra ed è quello che anche nella pratica viene fatto, ma in realtà di seguito secondo me il tutorial semplicemente vuole testare come cambiare T va ad influenzare i risultati in uscita!!!

def get_cost_colouring(bitstring, Q):
    z = np.array(list(bitstring), dtype=int) #prende quanto ottenuto dalla QPU, lo converte in un array di 0 e 1, e calcola il costo associato a quella stringa di bit usando la matrice Q.
    cost = z.T @ Q @ z
    return cost

#Calcola il valore medio dell'intera esecuzione quantistica, costituisce il feedback per il computer classico e la riparametrizzazione.
def get_cost(result, Q):
    counter = get_samples_from_result(result)
    cost = sum(counter[key] * get_cost_colouring(key, Q) for key in counter)
    return cost / sum(counter.values())  # Divide by total samples

cost = []
#Da 1 a 10 microsecondi, eseguiamo la sequenza quantistica, misuriamo il costo e lo memorizziamo. In pratica i parametri rimangono fissi e quelli di prima,
#ma si testa l'aumento del tempo T. In questo caso non cambiamo la parametrizzazione, ma il tempo si così da poter valutare per quale tempo T si ottiene lo stato a minore energia.
for T in 1000 * np.linspace(1, 10, 10):
    seq = Sequence(reg, FRESNEL_DEVICE)
    seq.declare_channel("ising", "rydberg_global")
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )
    seq.add(adiabatic_pulse, "ising")
    job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS, modulation=MODULATION)
    results = MYQLM_BACKEND.submit(job)
    cost.append(get_cost(results, Q) / 3)



# RESULTS PLOTTING
plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), np.array(cost), "--o", color="blue", label="Costo Quantistico Medio")

# Aggiungiamo il best_error nel titolo del grafico in modo che sia subito visibile
plt.title(f"Impatto del tempo T sulle performance quantistiche\n(Errore di embedding fisico: {best_error:.6f})", fontsize=15)

plt.xlabel("Total time evolution T (µs)", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.show()