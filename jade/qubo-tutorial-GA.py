# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

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

#One of the very first script with early experiments and considerations

"""
This file contains the variant of the QUBO problem from the original tutorial, but solved with a genetic algorithm (GA).
The idea is to, by modifying the problem itself, familiarize with the structure of a genetic algorithm, and then do a sensitivity analysis on the most important parameters (population and mutation) to understand how they influence convergence.

We simulate without noise, which is what we want to quantify directly on Jade.
"""


# We create a virtual representation of Pasqal's Fresnel QPU.
# By passing "None", we avoid connecting to the cloud servers.
FRESNEL_QPU = FresnelQPU(None)  

# We extract the physical specifications (maximum laser power, Rydberg radius, etc.).
# The simulator will use this object to prevent us from programming sequences impossible in nature.
FRESNEL_DEVICE = Device.from_abstract_repr(FRESNEL_QPU.get_specs().description)

# Ensure local calculations.
LOCAL_SIMULATIONS = True

# NBSHOTS = 0 indicates that locally the simulator will internally perform 2000 measurements
# to build the final statistical distribution of the results. MODULATION=False keeps the laser waves perfect and ideal.
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
# 1. PROBLEM SETUP (5x5 Q Matrix)
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
MIN_DISTANCE = 5.0 # constraint tied to the real machine (we cannot place two atoms too close, otherwise they merge into a super-atom and the physical model no longer holds)      
C6_COEFF = FRESNEL_DEVICE.interaction_coeff  # FRESNEL_DEVICE parameter
NUM_GENERATIONS = 400  # Fixed to have a fair time comparison

# ==========================================
# 2. GENETIC AND PHYSICAL FUNCTIONS
# ==========================================

# custom crossover ensuring that an atom's genes (x and y) always stay together, avoiding non-physical combinations
# the parameters are the 3 classic ones for a custom crossover function in PyGAD:
# - parents: matrix of the parent individuals selected for reproduction
# - offspring_size: tuple indicating the size of the offspring matrix to generate (number of children, number of genes)
# - ga_instance: the genetic algorithm instance.
def atom_aware_crossover(parents, offspring_size, ga_instance):

    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]): # at index 0 we have the number of children to generate

        # we choose the parents to mate in order to create the children in a circular way
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0] # as the second parent we take the one next to the previous one

        for atom_idx in range(N_ATOMS): # we iterate over the atoms and find the x and y coordinates corresponding to that atom (remember that each atom is represented by 2 genes: x and y)

            x_idx = atom_idx * 2
            y_idx = x_idx + 1

            # inspired by a Mendelian coin toss, having no particular need to unbalance 
            # the probability of choosing the parents' genes, we randomly decide whether to take the x and y coordinates of the atom from the first or second parent, but always together to maintain physical coherence
            # x_idx:y_idx+1 is used to take both the x and y coordinates of the atom, ensuring they are not mixed between different parents. +1 because the last one is always excluded.
            if np.random.rand() > 0.5:
                offspring[k, x_idx:y_idx+1] = parents[parent1_idx, x_idx:y_idx+1]
            else:
                offspring[k, x_idx:y_idx+1] = parents[parent2_idx, x_idx:y_idx+1]
    return offspring


# The idea is to have a fitness function that measures the adherence of the mapping found by GA with the Q matrix
# of the problem, but at the same time strongly penalizes solutions that violate the minimum distance constraint between atoms (MIN_DISTANCE), which is a real physical constraint of the machine on which we then want to implement the found solution.
def fitness_func(ga_instance, solution, solution_idx):

    coords_temp = np.reshape(solution, (N_ATOMS, 2))
    distances = pdist(coords_temp)
    
    if np.min(distances) < MIN_DISTANCE:
        return -999999.0  
        
    Q_fisica = squareform(C6_COEFF / (distances ** 6))
    np.fill_diagonal(Q_fisica, -10.0)
    
    errore = np.linalg.norm(Q_fisica - Q)
    return -errore # returned with a minus because PyGAD maximizes fitness, while we want to minimize the error. This way, the closer the solution is to the desired Q matrix, the higher (less negative) the fitness will be.

# ==========================================
# 3. TEST GRID (SENSITIVITY ANALYSIS)
# ==========================================
# We define the parameters to clash
populations = [50, 100, 200, 400, 800, 1000]      
mutations_percentage = [1, 2, 3, 5, 15, 30, 50]       # 30% - 50% chaos, this destroys the inheritance process of good traits. 
 
REPETITIONS = 10  # Number of runs for each parameter combination, to have a more robust estimate of the average performance and the evolutionary process variability.
# The pretests reported above showed that, for this specific problem, aside from the stochasticity of the process
# the best population is around 200 individuals with a 3% mutation percentage. I plan to do other tests
# with the values above and averaging over 10 runs each, but meanwhile I overwrite with the best data found so far
# to experiment with the quantum part.

populations = [50, 100, 200]      
mutations_percentage = [1, 2, 3, 5]

num_genes = N_ATOMS * 2 # The single coordinate of an atom is represented by 2 genes (x and y).

# this is how we define the value range dictionary for each gene. It is used both in creation and 
# during evolution.
gene_space = [{'low': 0.0, 'high': SPACE_BOUND} for _ in range(num_genes)]

# Dictionary to accumulate ALL 10 histories of each test
raw_results = {f"Pop: {pop} | Mut: {mut}%": [] for pop in populations for mut in mutations_percentage}

# --- ABSOLUTE CHAMPION TRACKER ---
# For the quantum part (Pulser)
best_overall_error = float('inf')
best_overall_coords = None
best_overall_name = ""

print(f"--- STARTING GRID SEARCH (Averages over {REPETITIONS} runs) ---")
tot_runs = len(populations) * len(mutations_percentage) * REPETITIONS
print(f"Total expected runs: {tot_runs}\n")

test_counter = 1
for pop in populations:
    for mut in mutations_percentage:
        test_name = f"Pop: {pop} | Mut: {mut}%"
        print(f"Test in progress: {test_name} (wait {REPETITIONS} runs)...")
        
       
            
        # Selective pressure: we always let 20% of the population reproduce
        # In the literature, a percentage between 10% and 30% is indicated as good, but it highly depends on the specific problem. 
        # Here we choose 20% as a compromise to maintain good genetic diversity without diluting the quality of the selected parents too much.
        num_parents = max(2, int(pop * 0.2)) 

        for rep in range(REPETITIONS):
            
            # We initialize a new "clean" PyGAD instance for each test
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
            
            # We extract the error and save the history of this specific run
            error_history = [abs(fitness) for fitness in ga_instance.best_solutions_fitness]
            raw_results[test_name].append(error_history)
            
            # We extract the best result of the run
            best_sol, best_fit, _ = ga_instance.best_solution()
            current_error = abs(best_fit)
            
            # --- UPDATING ABSOLUTE CHAMPION ---
            if current_error < best_overall_error:
                best_overall_error = current_error
                best_overall_coords = np.reshape(best_sol, (N_ATOMS, 2))
                best_overall_name = test_name
            
            test_counter += 1

print(f"\n--- SEARCH COMPLETED ---")
print(f"THE ABSOLUTE WINNER IS: {best_overall_name}")
print(f"Minimum error reached: {best_overall_error:.4f}")

# ==========================================
# 4. RESULTS VISUALIZATION (Averages Plot)
# ==========================================
plt.figure(figsize=(16, 10))

# We calculate the average of the 10 runs for each configuration
for test_name, history_list in raw_results.items():
    # history_list contains 10 lists. With np.mean(axis=0) we average column by column (generation by generation)
    avg_history = np.mean(history_list, axis=0)
    
    # We filter out the excessively high initial values to avoid ruining the plot scale
    filtered_avg_history = [min(err, 50) for err in avg_history] 
    
    # We highlight the winner
    linewidth = 3 if test_name == best_overall_name else 1.5
    
    plt.plot(filtered_avg_history, label=test_name, linewidth=linewidth)

plt.title(f"GA Sensitivity Analysis (Average over {REPETITIONS} runs)", fontsize=16)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Average Error", fontsize=14)
plt.legend(fontsize=10, loc="upper right", ncol=2)
plt.grid(True, linestyle="--", alpha=0.7)
plt.ylim(0, 30) 
plt.tight_layout()
plt.show()

print("--- SEARCH COMPLETED ---")

# ==========================================
# 5. TRANSITION TO THE QUANTUM PHASE (Pulser)
# ==========================================
print(f"\n--- STARTING QUANTUM SIMULATION (QAA) ---")
print(f"Using the best configuration found ({best_overall_name}) with error {best_overall_error:.4f}")

# We create the qubits dictionary using strings as keys ("q0", "q1", ...) for Pulser
# best_overall_coords contains the perfect (X,Y) calculated by the GA
qubits = {f"q{i}": coord for i, coord in enumerate(best_overall_coords)}
reg = Register(qubits)

# We draw the spatial register found by the GA
print("Visualizing the spatial layout of the atoms...")
reg.draw(
    blockade_radius=FRESNEL_DEVICE.rydberg_blockade_radius(1.0),
    draw_graph=False,
    draw_half_radius=True,
)

# ==========================================
# 6. DEFINITION OF THE ADIABATIC PULSE
# ==========================================
# We calculate the laser intensity (Omega) based on the Q matrix,
# ensuring we do not exceed the physical limit of the FRESNEL hardware.
max_amp = FRESNEL_DEVICE.channels["rydberg_global"].max_amp
Omega = min(np.median(Q[Q > 0].flatten()), max_amp)

delta_0 = -5  
delta_f = -delta_0  
T = 4000  # 4 microseconds (adiabatic evolution time)

# Creation of the bell waveform to avoid energy shocks
adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]), 
    InterpolatedWaveform(T, [delta_0, 0, delta_f]), 
    0, 
)

# We initialize the sequence and assign the global laser
seq = Sequence(reg, FRESNEL_DEVICE)
seq.declare_channel("ising", "rydberg_global")
seq.add(adiabatic_pulse, "ising")

# ==========================================
# 7. SIMULATION EXECUTION (MyQLM)
# ==========================================
print("Executing the adiabatic simulation (calculating wave functions)...")
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS, modulation=MODULATION)

# Local backend configuration
MyQLMPulserSimBackend = IsingAQPU.from_sequence(seq, qpu=None)
MYQLM_BACKEND = MyQLMPulserSimBackend if LOCAL_SIMULATIONS else AnalogQPU()

results = MYQLM_BACKEND.submit(job)

# ==========================================
# 8. VISUALIZATION OF QUANTUM RESULTS
# ==========================================
def get_samples_from_result(result: Result):
    """Extracts the probabilities of the quantum states from the MyQLM result"""
    samples = {}
    n_qubits = len(qubits)
    for sample in result.raw_data:
        if len(sample.state.bitstring) > n_qubits:
            raise ValueError(f"State {sample.state} is incompatible.")
        counts = sample.probability
        samples[sample.state.bitstring.zfill(n_qubits)] = counts
    return samples

def plot_distribution(result: Result):
    """Plots the histogram coloring the correct solutions red"""
    C = get_samples_from_result(result)
    
    # We sort the results in descending probability
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    
    # The two known optimal solutions of our QUBO problem. Specific to our Q matrix.
    indexes = ["01011", "00111"]  
    color_dict = {key: "r" if key in indexes else "g" for key in C}
    
    plt.figure(figsize=(12, 6))
    plt.xlabel("Bitstrings (Final Quantum States)", fontsize=12)
    plt.ylabel("Measurement Probability", fontsize=12)
    plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    
    # We insert the info about the genetic embedding used into the title
    plt.title(f"QAA Distribution - Based on Genetic Embedding ({best_overall_name} | Err: {best_overall_error:.2f})", fontsize=14)
    plt.tight_layout()
    plt.show()

plot_distribution(results)