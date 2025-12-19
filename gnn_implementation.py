import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_random_graph(n_nodes=15, edge_prob=0.3):
    """
    Generates a random symmetric adjacency matrix and node features
    for demonstration purposes.
    """
    # 1. Generate random Adjacency Matrix (A)
    # Create random edges
    A = np.random.choice([0, 1], size=(n_nodes, n_nodes), p=[1-edge_prob, edge_prob])
    # Make it symmetric (undirected graph)
    A = (A + A.T > 0).astype(int)
    # Remove self-loops
    np.fill_diagonal(A, 0)
    
    # 2. Generate random Node Features (X)
    # Simulating 10 possible features/categories for each node
    n_features = 10
    X = np.random.randn(n_features, n_nodes)
    
    return A, X

def draw_graph(A, title="Graph Visualization"):
    """Visualizes the graph using NetworkX."""
    G = nx.from_numpy_array(A)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=600, font_weight='bold', edge_color='gray')
    plt.title(title)
    plt.show()

def calculate_walks(A, length, start_node, end_node):
    """Calculates number of walks of a specific length between two nodes."""
    # Compute Matrix Power: A^k
    A_power = np.linalg.matrix_power(A, length)
    
    # Extract specific count
    count = A_power[start_node, end_node]
    return count, A_power

def graph_neural_network(A, X, Omega0, beta0, Omega1, beta1, Omega2, beta2, omega3, beta3):
    """
    Implements a 3-layer Graph Neural Network forward pass.
    Uses ReLU for hidden layers and Sigmoid for the readout.
    """
    # Layer 0: Aggregation -> Linear -> ReLU
    agg0 = np.matmul(X, A)
    h0 = np.maximum(0, np.matmul(Omega0, agg0) + beta0)

    # Layer 1: Aggregation -> Linear -> ReLU
    agg1 = np.matmul(h0, A)
    h1 = np.maximum(0, np.matmul(Omega1, agg1) + beta1)

    # Layer 2: Aggregation -> Linear -> ReLU
    agg2 = np.matmul(h1, A)
    h2 = np.maximum(0, np.matmul(Omega2, agg2) + beta2)

    # Output Layer (Readout)
    # Sum pooling
    h_graph = np.sum(h2, axis=1, keepdims=True)
    # Linear
    out = np.matmul(omega3, h_graph) + beta3
    # Sigmoid Activation
    f = 1 / (1 + np.exp(-out))
    
    return f

def graph_attention(X, omega, beta, phi, A):
    """
    Implements a Self-Attention mechanism for Graphs.
    """
    # 1. Compute Value transform (X')
    X_prime = np.matmul(omega, X) + beta

    # 2. Compute Attention Scores (S)
    S = np.matmul(np.matmul(X.T, phi), X)

    # 3. Apply Mask (Only attend to neighbors)
    n_nodes = X.shape[1]
    mask_condition = (A + np.eye(n_nodes)) == 0
    S[mask_condition] = -1e20 # Set to neg infinity to zero out in softmax

    # 4. Softmax Normalization
    S_max = np.max(S, axis=0, keepdims=True)
    exp_S = np.exp(S - S_max)
    attention = exp_S / np.sum(exp_S, axis=0, keepdims=True)

    # 5. Aggregate
    aggregation = np.matmul(X_prime, attention)

    # 6. Activation
    output = np.maximum(0, aggregation)

    return output

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Initializing Random Graph...")
    # Generate a random graph with 15 nodes
    A, X = generate_random_graph(n_nodes=15)
    
    # 1. Visualize
    # draw_graph(A) # Uncomment to show plot

    # 2. Calculate Walks
    walks, _ = calculate_walks(A, length=4, start_node=5, end_node=6)
    print(f"Number of walks of length 4 between nodes 5 and 6: {walks}")

    # 3. Initialize GNN Weights (Randomized)
    n_in = X.shape[0]
    n_hidden = 5
    
    np.random.seed(42)
    Omega0 = np.random.randn(n_hidden, n_in) * 0.1
    beta0 = np.random.randn(n_hidden, 1) * 0.1
    Omega1 = np.random.randn(n_hidden, n_hidden) * 0.1
    beta1 = np.random.randn(n_hidden, 1) * 0.1
    Omega2 = np.random.randn(n_hidden, n_hidden) * 0.1
    beta2 = np.random.randn(n_hidden, 1) * 0.1
    omega3 = np.random.randn(1, n_hidden) * 0.1
    beta3 = np.random.randn(1, 1) * 0.1

    # 4. Run GNN Forward Pass
    prediction = graph_neural_network(A, X, Omega0, beta0, Omega1, beta1, Omega2, beta2, omega3, beta3)
    print(f"GNN Prediction (Probability): {prediction[0,0]:.4f}")

    # 5. Run Graph Attention
    # Re-init weights for attention dimensions
    omega_att = np.random.randn(n_hidden, n_in) * 0.1
    beta_att = np.random.randn(n_hidden, 1) * 0.1
    phi = np.random.randn(n_in, n_in) * 0.1
    
    att_output = graph_attention(X, omega_att, beta_att, phi, A)
    print(f"Attention Output Shape: {att_output.shape}")
    print("Success! GNN implementation is functional.")
