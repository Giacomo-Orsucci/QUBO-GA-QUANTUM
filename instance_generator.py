# SPDX-FileCopyrightText: © 2026 Giacomo Orsucci
# SPDX-License-Identifier: MIT

import numpy as np
import os
from scipy.spatial.distance import pdist, squareform


target_degree = 8.0

#The final density has some variance in it, but is due to random generation.
def generate_qubo(n=5, density=0.25, seed=42):
    # Fixed seed
    np.random.seed(seed)
    
    Q = np.zeros((n, n))
    # Costant diagonal to generate instances we can solve with global ray
    np.fill_diagonal(Q, -1.0)
    
    # positive random connection
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < density:
                val = np.random.uniform(0.5, 1.5)
                Q[i, j] = val
                Q[j, i] = val
                

    os.makedirs("./my_QUBO_instances/scaling_tests/friendly", exist_ok=True)
    
    file_name = f"./my_QUBO_instances/scaling_tests/global_friendly_{n}x{n}_d{int(density*100)}_s{seed}.npz"
    
    i_idx, j_idx = np.where(np.triu(Q) != 0)
    weights = Q[i_idx, j_idx]
    np.savez(file_name, i=i_idx, j=j_idx, Jij=weights)
    print(f"[OK] Generata istanza: {file_name}")


import numpy as np
import os
from scipy.spatial.distance import pdist, squareform

def generate_verisimilar_UDG_qubo(n=5, target_degree=2.5, min_dist=4.0, rydberg_radius=12.0, max_hw_radius=50.0, seed=42):
    """
    Generate QUBO UDG instances to test their embedding on neutral atoms architecture like Jade.
    The generation is performed with the goal of creating a real matrix population with physical 
    constraints guaranteed. These instances can be used to test how the GA embedding performs and scales
    on Jade-like neutral atoms architecture to study how the success rate changes scaling the problem
    and changing its topology.
    """
    np.random.seed(seed)
    
    # 1. Calculation of the Dynamic Operational Area
    # In a Random Geometric Graph, to achieve an average degree (target_degree),
    # the placement radius must scale with the square root of N.
    # Formula derived from: <k> = N * (Rydberg_Area / Generation_Area)
    effective_radius = rydberg_radius * np.sqrt(n / target_degree)
    
    ## Ensure the area NEVER exceeds the physical limits of the machine (50 µm)
    # and that it's not too small to fit the atoms at min_dist
    min_required_radius = np.sqrt(n) * (min_dist / 2.0) * 1.5 
    effective_radius = np.clip(effective_radius, min_required_radius, max_hw_radius)
    
    print(f"Generation {n}x{n} | Calculated Operational Radius: {effective_radius:.2f} µm (Max Hardware: {max_hw_radius} µm)")    
    
    # Coordinates Generation with Rejection Sampling
    positions = []
    attempts = 0
    max_attempts = 20000
    
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        
        # Place atoms within the Operational Area (not the entire hardware area)
        # Use polar coordinates to distribute them uniformly in a circle
        r = effective_radius * np.sqrt(np.random.rand())
        theta = np.random.rand() * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Hardware Minimum Distance Check (Jade: 4 µm)
        if len(positions) > 0:
            pos_array = np.array(positions)
            distances_to_others = np.linalg.norm(pos_array - np.array([x, y]), axis=1)
            if np.any(distances_to_others < min_dist):
                continue # Too close, discard the atom
                
        positions.append([x, y])
        
    if len(positions) < n:
        raise ValueError(f"Unable to place {n} atoms! Try lowering the target_degree or min_dist.")        
    
    positions = np.array(positions)
    
    # QUBO Matrix Construction
    dist_matrix = squareform(pdist(positions))
    Q = np.zeros((n, n))
    
    # Diagonal: -1.0 (Global tendency to turn on)
    np.fill_diagonal(Q, -1.0)
    
    # Edges (Penalty): 2.0 if the distance is less than the Rydberg Radius
    adjacency_mask = (dist_matrix < rydberg_radius) & (dist_matrix > 0)
    Q[adjacency_mask] = 2.0
    
    # Saving Compatible with your 'load_hamburg_matrix'
    output_dir = "./my_QUBO_instances/scaling_tests/jade_udg"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/jade_udg_{n}x{n}_R{int(rydberg_radius)}_s{seed}_d{target_degree}.npz"
    
    # Upper Triangle Extraction
    i_idx, j_idx = np.where(np.triu(Q) != 0)
    weights = Q[i_idx, j_idx]
    
    np.savez(file_name, i=i_idx, j=j_idx, Jij=weights, positions=positions)
    
    # Validation statistics
    num_edges = np.sum(adjacency_mask) / 2
    density = num_edges / (n * (n - 1) / 2) if n > 1 else 0
    actual_degree = (num_edges * 2) / n
    print(f"-> [OK] Salvato. Archi totali: {int(num_edges)} | Grado medio: {actual_degree:.1f} | Densità: {density:.2f}\n")

# --- TEST DI SCALING ---
if __name__ == "__main__":
    for size in range(20, 21):
        # target_degree=3.0 guarantees that each node, on average, clashes 
        # (violates the Rydberg blockade) with 3 other nodes. 
        # It is a perfect difficulty level to test the embedding methods.
        generate_verisimilar_UDG_qubo(
            n=size, 
            target_degree=target_degree, 
            min_dist=4.0, 
            rydberg_radius=12.0, 
            seed=42 + size
        )