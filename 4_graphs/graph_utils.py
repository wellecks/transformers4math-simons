import numpy as np
import makemore
import torch
from tqdm import trange

def evaluate_graph(adj_matrix):
    # Number of edges
    num_edges = np.sum(adj_matrix) / 2
    return num_edges

def deserialize_graph(s, n=20):
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
    try:
        for i in range(n - 1):
            for j in range(i + 1, n):
                adjmat[i, j] = int(s[index])
                adjmat[j, i] = adjmat[i, j]
                index += 1
    except IndexError:
        return None
    return adjmat    

def batch_generate(model, train_dataset, num=100, device='cpu', batch_size=16, top_k=None):
    batch_size = min(num, batch_size)
    num_batches = max(num // batch_size, 1)
    model.eval()
    all_samples, new_samples = [], []
    for b in trange(num_batches):
        X_init = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
        top_k = top_k if top_k != -1 else None
        steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
        X_samp = makemore.generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
        for i in range(X_samp.size(0)):
            # get the i'th row of sampled integers, as python list
            row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
            # token 0 is the <STOP> token, so we crop the output sequence at that point
            crop_index = row.index(0) if 0 in row else len(row)
            row = row[:crop_index]
            word_samp = train_dataset.decode(row)
            all_samples.append(word_samp)
            if not train_dataset.contains(word_samp):
                new_samples.append(word_samp)
    model.train()
    return all_samples, new_samples