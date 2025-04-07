"""Based on https://github.com/zawagner22/transformers_math_experiments/blob/main/problem_triangle_free.jl
by Adam Wagner
"""
import numpy as np
import random
import statistics
import os
from tqdm import trange

def find_all_triangles(adjmat):
    n = adjmat.shape[0]
    triangles = []
    # Loop over all triples (i, j, k) with i < j < k
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                if adjmat[i, j] == 1 and adjmat[j, k] == 1 and adjmat[i, k] == 1:
                    triangles.append((i, j, k))
    return triangles

def convert_adjmat_to_string(adjmat):
    n = adjmat.shape[0]
    entries = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            entries.append(str(int(adjmat[i, j])))
        
        # Avoid adding a comma after the last row.
        if i < n - 2:
            entries.append(",")

    return "".join(entries)

def greedy_search_from_startpoint(adjmat):
    """
    Main greedy search algorithm.
    
    Starting from an input graph (represented as a string for the upper triangular entries)
    this function first removes edges to destroy all triangles and then greedily adds edges
    (without creating triangles) until no further addition is possible.
    
    Parameters:
        adjmat (np.ndarray): Input adjacency matrix of the graph.
    
    Returns:
        np.ndarray: The final adjacency matrix after greedy search.
    """
    n = adjmat.shape[0]

    # Remove edges until no triangles remain.
    triangles = find_all_triangles(adjmat)
    while triangles:
        # Count frequency of each edge in all triangles.
        edge_count = {}
        for (i, j, k) in triangles:
            for edge in [(i, j), (j, k), (i, k)]:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Find the most frequent edge.
        most_frequent_edge = max(edge_count, key=edge_count.get)
        i_edge, j_edge = most_frequent_edge
        
        # Remove the most frequent edge from the adjacency matrix.
        adjmat[i_edge, j_edge] = 0
        adjmat[j_edge, i_edge] = 0
        
        # Filter out triangles that include the removed edge.
        triangles = [t for t in triangles 
                     if most_frequent_edge not in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]]
    
    # Now add random edges without creating triangles until no allowed edge remains.
    allowed_edges = []
    # Compute the square of the adjacency matrix.
    adjmat2 = np.dot(adjmat, adjmat)
    # Identify allowed edges (i, j) where no path of length 2 exists (i.e. adjmat2[i, j] == 0)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if adjmat[i, j] == 0 and adjmat2[i, j] == 0:
                allowed_edges.append((i, j))
    
    while allowed_edges:
        # Randomly select an edge to add.
        edge = random.choice(allowed_edges)
        i_edge, j_edge = edge
        adjmat[i_edge, j_edge] = 1
        adjmat[j_edge, i_edge] = 1
        
        # Update allowed_edges by removing edges that now would create a triangle with the added edge.
        new_allowed_edges = []
        for (a, b) in allowed_edges:
            # Check if adding this edge now would form a triangle with the newly added edge.
            if (a == i_edge and adjmat[b, j_edge] == 1) or (a == j_edge and adjmat[b, i_edge] == 1) or \
               (b == i_edge and adjmat[a, j_edge] == 1) or (b == j_edge and adjmat[a, i_edge] == 1):
                continue
            # Also, remove the edge if it is the one we just added.
            if (a == i_edge and b == j_edge) or (a == j_edge and b == i_edge):
                continue
            new_allowed_edges.append((a, b))
        allowed_edges = new_allowed_edges

    return adjmat

def reward_calc(obj):
    """
    Calculate the reward for a given graph construction.
    
    In this example, the reward is defined as the number of edges in the graph,
    which is the count of '1's in the string representation.
    
    Parameters:
        obj (str): String representation of the graph (upper triangular entries).
    
    Returns:
        reward (int): The number of edges in the graph.
    """
    return obj.count('1')

def empty_starting_point(n):
    """
    Generate an empty starting point (graph) where no edges are present.
    
    Returns:
        obj (str): A string of '0's of length N*(N-1)/2.
    """
    return "0" * (n * (n - 1) // 2)


def deserialize_graph(s, n):
    """
    Deserialize a string representation back into an adjacency matrix.
    
    Parameters:
        s (str): String representation of the upper triangular part of the adjacency matrix.
    
    Returns:
        np.ndarray: An N x N adjacency matrix.
    """
    # Remove commas
    s = s.replace(",", "")
    adjmat = np.zeros((n, n), dtype=int)
    index = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            adjmat[i, j] = int(s[index])
            adjmat[j, i] = adjmat[i, j]
            index += 1
    return adjmat    


def load_starting_points(filename, n):
    if filename == "empty":
        return [empty_starting_point(n)]
    else:
        with open(filename, 'r') as f:
            serialized_graphs = [line.strip().replace(',', '') for line in f]
            serialized_graphs = [g for g in serialized_graphs if len(g) == n * (n - 1) // 2]
            # score each graph
            serialized_graphs = [(g, reward_calc(g)) for g in serialized_graphs]
            # get the top 10% of graphs
            serialized_graphs.sort(key=lambda x: x[1], reverse=True)
            serialized_graphs = [g[0] for g in serialized_graphs[:len(serialized_graphs) // 10]]

        return serialized_graphs

def save_graphs(graphs, rewards, output_dir='output'):
    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets with the top 10% of rewards
    top_10_percent_threshold = statistics.quantiles(rewards, n=10)[-1]
    top_10_percent_graphs = [graphs[i] for i in range(len(graphs)) if rewards[i] >= top_10_percent_threshold]
    output_file = os.path.join(output_dir, "top_10_percent_graphs.txt")
    with open(output_file, "w") as f:
        for graph in top_10_percent_graphs:
            f.write(f"{graph}\n")
    print(f"Saved {len(top_10_percent_graphs)} graphs with rewards in the top 10% to {output_file}")

    # Save the full dataset
    output_file = os.path.join(output_dir, "all_graphs.txt")
    with open(output_file, "w") as f:
        for graph in graphs:
            f.write(f"{graph}\n")
    print(f"Saved {len(graphs)} graphs to {output_file}")


def main(args):
    n = args.n
    num_runs = args.num_runs
    rewards = []
    graphs = []

    starting_points = load_starting_points(args.starting_point, n)

    for _ in trange(num_runs):
        start_obj = random.choice(starting_points)
        adjmat = deserialize_graph(start_obj, n)
        final_adjmat = greedy_search_from_startpoint(adjmat)
        final_graph = convert_adjmat_to_string(final_adjmat)
        reward = reward_calc(final_graph)
        rewards.append(reward)
        graphs.append(final_graph)

    save_graphs(graphs, rewards, args.output_dir)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Greedy search for triangle-free graphs.")
    parser.add_argument("--num_runs", type=int, default=100000, help="Number of runs to perform.")
    parser.add_argument("--n", type=int, default=20, help="Number of vertices in the graph.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save output files.")
    parser.add_argument("--starting_point", type=str, default="empty", help="Starting point for the search.")
    args = parser.parse_args()

    main(args)
