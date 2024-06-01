import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from RAGE import RAGE 

x_path = "/homes/gws/sqhang/proj/2024_sqhang_Transductive-bandits/data/tf_bind_8-x-0.npy"
x_arr = np.load(x_path, allow_pickle=True)
# print(x_arr.shape)

y_path = "/homes/gws/sqhang/proj/2024_sqhang_Transductive-bandits/data/tf_bind_8-y-0.npy"
y_arr = np.load(y_path, allow_pickle=True)
# print(y_arr.shape)

x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.2, random_state=42)

# print("Shape of x_train:", x_train.shape)
# print("Shape of x_test:", x_test.shape)
# print("Shape of y_train:", y_train.shape)
# print("Shape of y_test:", y_test.shape)

def make_kmer_list(k, alphabet="ACGT", upto=False):
    """Generate sorted list of k-mers for each k up to the given k, considering reverse complements."""
    kmer_dict = {}
    if upto:
        range_k = range(1, k + 1)
    else:
        range_k = [k]
    for current_k in range_k:
        kmer_set = set()
        for kmer_tuple in itertools.product(alphabet, repeat=current_k):
            kmer = ''.join(kmer_tuple)
            rev_kmer = ''.join(reversed([{'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}[x] for x in kmer]))
            kmer_set.add(min(kmer, rev_kmer))
        kmer_dict[current_k] = sorted(kmer_set)
    # Flatten dictionary into a list while preserving order by k-mer length
    kmer_list = [kmer for sublist in range_k for kmer in kmer_dict[sublist]]
    return kmer_list, kmer_dict

### Counting k-mers
def count_kmers(sequences, k, normalize=None, upto=False):
    """Count k-mers in sequences for all lengths up to k if 'upto' is True, and optionally normalize the counts."""
    alphabet = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    num_sequences = sequences.shape[0]
    kmer_list, kmer_dict = make_kmer_list(k, alphabet="ACGT", upto=upto)
    kmer_index = {kmer: idx for idx, kmer in enumerate(kmer_list)}
    kmer_matrix = np.zeros((num_sequences, len(kmer_list)), dtype=float)  # Changed dtype to float

    # Convert sequences to strings and count k-mers
    for i in range(num_sequences):
        sequence = ''.join(alphabet[b] for b in sequences[i])
        for current_k, kmers in kmer_dict.items():
            for j in range(len(sequence) - current_k + 1):
                kmer = sequence[j:j+current_k]
                rev_kmer = ''.join(reversed([{'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}[x] for x in kmer]))
                canonical_kmer = min(kmer, rev_kmer)
                if canonical_kmer in kmer_index:
                    kmer_matrix[i, kmer_index[canonical_kmer]] += 1

    if normalize:
        if normalize == "frequency":
            # Normalize separately for each k-length group
            for current_k, kmers in kmer_dict.items():
                indices = [kmer_index[kmer] for kmer in kmers]
                sums = kmer_matrix[:, indices].sum(axis=1, keepdims=True)
                kmer_matrix[:, indices] /= np.maximum(1, sums)
        elif normalize == "unitsphere":
            for current_k, kmers in kmer_dict.items():
                indices = [kmer_index[kmer] for kmer in kmers]
                norms = np.linalg.norm(kmer_matrix[:, indices], axis=1, keepdims=True)
                kmer_matrix[:, indices] /= np.maximum(1e-10, norms)

    return kmer_matrix

k = 4
x_train_kmer = count_kmers(x_train, k, normalize='frequency', upto=True)
# print(x_train_kmer.shape)

k = 4
x_test_kmer = count_kmers(x_test, k, normalize='frequency', upto=True)
# print(x_test_kmer.shape)

model_kmer = LinearRegression(fit_intercept=False)
model_kmer.fit(x_train_kmer, y_train)

theta_hat = model_kmer.coef_
intercept = model_kmer.intercept_
# print(theta_hat.T.shape)
# print(intercept)

X = x_test_kmer[:10000].astype(np.float32)
Z = x_test_kmer[:10000].astype(np.float32)
theta_star = theta_hat.T.astype(np.float32)

factor = 100
delta = 0.10

def main():
    rage_instance = RAGE(X, theta_star, factor, delta, Z)
    rage_instance.algorithm(42)

if __name__ == "__main__":
    main()