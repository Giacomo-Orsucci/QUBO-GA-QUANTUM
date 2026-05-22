import numpy as np
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite


#Obvious suggestion: export the token on your environment, do not hardoce it here and do not push it on GitHub 

#very first implementation, but some pretests on the 2 matrices specified in file_path below, highlight a supremacy
#of D-wave on QUBO problems: its more userfriendly, mature and with better performances.

#But, in the near future I want to compare D-Wave on a set of problems that jade "likes" and maybe
#explore some big instances to see how D-Wave scales.

# ==========================================
# 2. NATIVE PARSER (.npz) - DWAVE OPTIMIZED
# ==========================================
def load_hamburg_matrix_for_dwave(file_path):
    """
    Loads an .npz matrix and returns a sparse dictionary 
    in upper-triangular format, ready for D-Wave's sample_qubo().
    """
    try:
        with np.load(file_path, allow_pickle=True) as data:
            i_indices = data['i']
            j_indices = data['j']
            weights = data['Jij']
            
            # Use a dictionary to store only non-zero interactions (Sparse format)
            Q_dict = {}
            
            for r, c, w in zip(i_indices, j_indices, weights):
                r, c = int(r), int(c)
                
                # D-Wave QUBO standard strictly prefers upper-triangular format (row <= col).
                # We enforce this by swapping indices if row > col.
                if r > c:
                    r, c = c, r
                
                # If the symmetric counterpart was already processed, we add the weight.
                # This prevents doubling the energy if the original file had both (r,c) and (c,r).
                if (r, c) in Q_dict:
                    Q_dict[(r, c)] += float(w)
                else:
                    Q_dict[(r, c)] = float(w)
                    
            return Q_dict
            
    except Exception as e:
        print(f"  [READ ERROR] Unable to load {file_path}: {e}")
        return None
    
def load_hamburg_matrix_for_dwave_aligned(file_path):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            i_indices = data['i']
            j_indices = data['j']
            weights = data['Jij']
            
            Q_dict = {}
            
            for r, c, w in zip(i_indices, j_indices, weights):
                r, c = int(r), int(c)
                
                if r == c:
                    # Linear terms (diagonal): DO NOT double
                    if (r, r) in Q_dict:
                        Q_dict[(r, r)] += float(w)
                    else:
                        Q_dict[(r, r)] = float(w)
                else:
                    # Quadratic terms (off-diagonal): Classical code mirrored them,
                    # so we must multiply by 2 to match the x^T Q x energy.
                    if r > c:
                        r, c = c, r
                        
                    if (r, c) in Q_dict:
                        Q_dict[(r, c)] += 2.0 * float(w)
                    else:
                        Q_dict[(r, c)] = 2.0 * float(w)
                        
            return Q_dict
            
    except Exception as e:
        print(f"  [READ ERROR] Unable to load {file_path}: {e}")
        return None
    
# ==========================================
# 3. D-WAVE EXECUTION MODULE
# ==========================================
def solve_with_dwave(file_path, use_hybrid=True):
    """
    Loads a QUBO from an .npz file and solves it using D-Wave.
    
    Args:
        file_path (str): Path to the .npz matrix.
        use_hybrid (bool): If True, uses LeapHybridSampler (best for large problems).
                           If False, uses pure QPU with EmbeddingComposite.
    """
    # 1. Load the parsed dictionary
    print(f"Loading matrix from {file_path}...")
    Q_dict = load_hamburg_matrix_for_dwave_aligned(file_path)
    
    if Q_dict is None:
        print("Execution aborted due to loading error.")
        return None
        
    print(f"Success! Loaded QUBO with {len(Q_dict)} non-zero interactions.")
    
    # 2. Select the solver and execute
    if use_hybrid:
        print("Submitting to LeapHybridSampler (Cloud + QPU)...")
        sampler = LeapHybridSampler()
        # The hybrid solver determines the optimal number of reads/time limit automatically
        sampleset = sampler.sample_qubo(Q_dict)
    else:
        print("Submitting to pure QPU via EmbeddingComposite...")
        sampler = EmbeddingComposite(DWaveSampler())
        # For pure QPU, we explicitly define how many times to sample the state
        sampleset = sampler.sample_qubo(Q_dict, num_reads=250, return_embedding=True)
        
    best_solution = sampleset.first
    
    # 3b. Smart extraction of execution time for both QPU and Hybrid
    timing_data = sampleset.info.get('timing', {})
    exec_time = timing_data.get('qpu_access_time', sampleset.info.get('run_time', 'N/A'))
    
    print("\n" + "="*30)
    print("        D-WAVE RESULTS")
    print("="*30)
    print(f"Minimum Energy Found : {best_solution.energy}")
    print(f"Execution Time       : {exec_time} microseconds")
    print(f"Number of variables  : {len(best_solution.sample)}")
    
    # Extract and display the nodes assigned to '1' by the quantum computer
    active_nodes = [node for node, bit in best_solution.sample.items() if bit == 1]
    print(f"Solution Nodes (x=1) : {active_nodes}")
    print(f"Full bitstring       : {best_solution.sample}")



    embedding = sampleset.info.get('embedding_context', {}).get('embedding', {})
    
    # ---------------------------------------------------------
    # 4. ROBUST EMBEDDING EXTRACTION
    # ---------------------------------------------------------
    print("\n" + "="*30)
    print("        EMBEDDING INFO")
    print("="*30)
    
    # Let's see what D-Wave actually returned in the info dictionary
    # print("Available keys in info:", sampleset.info.keys())
    
    # Try the standard path first
    embedding_context = sampleset.info.get('embedding_context', {})
    embedding = embedding_context.get('embedding', None)
    
    # Fallback: sometimes it's just under 'embedding'
    if embedding is None:
        embedding = sampleset.info.get('embedding', None)
        
    if embedding:
        total_physical_qubits = sum(len(chain) for chain in embedding.values())
        max_chain_length = max(len(chain) for chain in embedding.values())
        
        print(f"Logical variables (from .npz) : {len(embedding)}")
        print(f"Physical Qubits used on chip  : {total_physical_qubits}")
        print(f"Maximum chain length          : {max_chain_length}")
        
        # Print the exact mapping for each logical node
        print("\nExact mapping (Logical -> Physical Qubits):")
        for logical_node, physical_chain in embedding.items():
            print(f"  Node {logical_node} -> Qubits {physical_chain}")
    else:
        print("Embedding data not found directly in sampleset.info.")
        print("Raw sampleset.info output for debugging:")
        print(sampleset.info)
        
    return best_solution
    
    return best_solution




#file_path = "../my_QUBO_instances/tutorial_5x5.npz"
file_path = "../qubo-bench/qubo-benchmark-main/instances/compsup/2d/2d_(4, 4)_precision256_seed18.npz"
hybrid = False
# Example of how to call it:
result = solve_with_dwave(file_path, use_hybrid=hybrid)

