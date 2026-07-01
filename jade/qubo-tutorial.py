# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

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
# MODIFICATIONS/ADDITIONS: optimization executed 10 times to slightly mitigate Nelder-Mead's sensitivity to local minima
# and compare the error (and thus the best solution) with the one obtained using GA.


# Create a virtual representation of Pasqal's Fresnel QPU.
# By passing "None", we avoid connecting to cloud servers.
FRESNEL_QPU = FresnelQPU(None)  

# Extract physical specifications (maximum laser power, Rydberg radius, etc.).
# The simulator will use this object to prevent us from programming sequences that are physically impossible.
FRESNEL_DEVICE = Device.from_abstract_repr(FRESNEL_QPU.get_specs().description)

# Ensure whether calculations are executed locally or not.
LOCAL_SIMULATIONS = False

# NBSHOTS = 0 indicates that locally the simulator will internally perform 2000 measurements
# to build the final statistical distribution. MODULATION=False keeps the laser waves perfect and ideal.
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

# The Q matrix represents the nodes (on the diagonal) and the edges/penalties (off-diagonal).
# The diagonal values are all equal (-10.0), which will allow us to use a single global laser.

Q = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)

# And we can check optimal solutions (exponential cost in Q dimension; brute force):

# We generate all 32 possible combinations (2^5). 
# This line highlights the classical limit: for N atoms, the time grows exponentially.
bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))] # generate all possible bitstrings of length equal to the dimension of Q
costs = []


# this takes exponential time with the dimension of the QUBO
for b in bitstrings:
    z = np.array(list(b), dtype=int)
    # We calculate the configuration's energy by multiplying the state by the rules matrix.
    cost = z.T @ Q @ z # then compute the cost for each bitstring using the QUBO formula: z^T Q z
    costs.append(cost) # costs represent the value of the objective function, but also the minimum energy of the system for that configuration of bits



# Sort and print. The results will tell us that 01011 and 00111 are the minimum energy states.
zipped = zip(bitstrings, costs)
sort_zipped = sorted(zipped, key=lambda x: x[1])
print(sort_zipped[:3])

def evaluate_mapping(new_coords, *args):
    """Cost function to minimize. Ideally, the pairwise distances are conserved."""
    Q, shape = args
    new_coords = np.reshape(new_coords, shape)

    # This is the fundamental physical law: the van der Waals interaction decreases 
    # with the sixth power of the distance (pdist(new_coords) ** 6).
    new_Q = squareform(FRESNEL_DEVICE.interaction_coeff / pdist(new_coords) ** 6)
    # Returns the difference between the physics (new_Q) and the math (Q). 
    # We want this value to tend to zero.
    return np.linalg.norm(new_Q - Q) # to test if the "quantistic" atoms disposure reflect the original matrix.
# high value returned -> bad disposure

shape = (len(Q), 2)
costs = []
np.random.seed(0)

best_error = float('inf')
best_coords = None
best_res = None

for i in range(10):  # Run the optimization multiple times to mitigate local minima
    # We start from a random 2D coordinate layout at each iteration
    x0 = np.random.random(shape).flatten()
    # The classical Nelder-Mead algorithm moves the atoms in the virtual space to minimize
    # the error calculated by the evaluate_mapping function.
    res = minimize(
        evaluate_mapping,
        x0,
        args=(Q, shape),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None},
    )

    # res.fun contains the value returned by evaluate_mapping (our error)
    current_error = res.fun
    print(f"Iteration {i+1} - Error: {current_error:.4f}")

    # If we find a lower error, we update our best solution
    if current_error < best_error:
        best_error = current_error
        best_res = res
        best_coords = np.reshape(res.x, (len(Q), 2))

print(f"\nOptimization completed. Best error found: {best_error:.4f}")

# We register the WINNING coordinates in the machine's physical Register
coords = best_coords
qubits = dict(enumerate(coords))
reg = Register(qubits)


# Draw the register. draw_half_radius=True visually shows the Rydberg Blockade:
# if two halos overlap, the atoms cannot be excited simultaneously.
reg.draw(
    blockade_radius=FRESNEL_DEVICE.rydberg_blockade_radius(1.0),
    draw_graph=False,
    draw_half_radius=True,
)

# THE ATOMS ARE NOW SET, WE CAN NOW DEFINE PARAMETERS FOR THE ADIABATIC PULSE
# (See equation in section 2 of the tutorial).

# Omega is the laser intensity (Rabi frequency, amplitude). We use the median of the constraints
# to ensure the laser has a strength comparable to the Rydberg blockade.
Omega = np.median(Q[Q > 0].flatten())
# Detuning (relative laser frequency). 
# It starts from a negative value (penalizes atom excitation)...
delta_0 = -5  # just has to be negative
# ...and reaches a positive value (rewards atom excitation). This is because initially we want the system in its ground state (all atoms in ground state), and eventually we want it in an excited state corresponding to the solution of our optimization problem.
delta_f = -delta_0  # just has to be positive
# Duration of the process (4 microseconds). Long enough to allow 
# the system to adapt adiabatically without "breaking" the minimum energy state.
T = 4000  # time in ns, we choose a time long enough to ensure the propagation of information in the system


# Build the laser waveform over time.
adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]), # Starting from 1e-9, it rises to Omega and drops back down to 1e-9. This bell shape avoids giving the atoms an energy shock.
    InterpolatedWaveform(T, [delta_0, 0, delta_f]), # laser frequency, starting negative and rising to positive. This is because initially we want to penalize atom excitation, which we later want to reward.
    0, # Phase, not important in this case, so we leave it at zero.
)


# Insert the pulse in the "rydberg_global" channel, meaning the laser 
# will hit all atoms simultaneously, exploiting the homogeneity of Q's diagonal.
seq = Sequence(reg, FRESNEL_DEVICE)
seq.declare_channel("ising", "rydberg_global")
seq.add(adiabatic_pulse, "ising")
seq.draw()

# Convert the sequence into a job for the QLM infrastructure.
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS, modulation=MODULATION)

# Choose the backend. In this case, it will use MyQLMPulserSimBackend on the local PC.
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
    Generates the histogram of the results. It already knows the correct solutions 
    ("01011", "00111") and instructs the graph to color them red ("r"), 
    leaving sub-optimal configurations green ("g").
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

# To ensure the system remains in the minimum energy state throughout its evolution, it is crucial to choose a sufficiently long duration T. If T is too short, the system might not have enough time to adapt adiabatically, risking a "break" in the minimum energy state and ending up in an unwanted excited state. In this case, we chose T = 4000 ns, which is long enough to let the system evolve without unwanted transitions.
# The only problem is that in the real world, the chosen T is too long due to environmental and thermal noise.
# Therefore, we move to QAOA (Quantum Approximate Optimization Algorithm), which is a "digital" version of the adiabatic approach and allows optimizations to be executed in shorter times, reducing the impact of noise.
# The idea is not to use a single long and slow pulse, but shorter and parameterized pulses. However, this approach rarely leads directly to the perfect solution on the first try.
# Therefore, the QPU is randomly parameterized, executes, and performs the measurement through which the classical computer measures the result quality (z^T Q z) and calculates the average cost of the computed strings. If this cost is high, the parameterization was incorrect. A classical optimization algorithm is then used to adjust the parameters, and it is re-executed on the QPU.
# 
# IMPORTANT:
# 
# Everything written above is correct and is what is actually done in practice, but below, in my opinion, the tutorial simply wants to test how changing T affects the output results!!!

def get_cost_colouring(bitstring, Q):
    z = np.array(list(bitstring), dtype=int) # takes what is obtained from the QPU, converts it into an array of 0s and 1s, and computes the cost associated with that bitstring using the Q matrix.
    cost = z.T @ Q @ z
    return cost

# Calculates the average value of the entire quantum execution; this constitutes the feedback for the classical computer and re-parameterization.
def get_cost(result, Q):
    counter = get_samples_from_result(result)
    cost = sum(counter[key] * get_cost_colouring(key, Q) for key in counter)
    return cost / sum(counter.values())  # Divide by total samples

cost = []
# From 1 to 10 microseconds, we execute the quantum sequence, measure the cost, and store it. In practice, the parameters remain fixed as before,
# but we test the increase in time T. In this case, we do not change the parameterization, but we do change the time so we can evaluate for which time T the lowest energy state is achieved.
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
plt.plot(range(1, 11), np.array(cost), "--o", color="blue", label="Average Quantum Cost")

# Add the best_error to the plot title so it is immediately visible
plt.title(f"Impact of time T on quantum performance\n(Physical embedding error: {best_error:.6f})", fontsize=15)

plt.xlabel("Total time evolution T (µs)", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.show()