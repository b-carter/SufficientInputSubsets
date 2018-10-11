import os
import sys
import numpy as np

LEVENSHTEIN_CODE_DIR = 'packages/Levenshtein'
sys.path.insert(0, os.path.abspath(LEVENSHTEIN_CODE_DIR))
import Levenshtein


RATIONALES_FILENAME = 'rationales_greedy.txt'
OUTFILE_NAME = 'rationales_greedy_dists.txt.gz'
VERBOSE = False

# Truncates a rationale string so "NNNATNGNN" becomes just "ATNG"
# (Removes and leading and trailing N's, preserves internal N's)
def truncate_rationale(rationale, pad_char='N'):
        return rationale.strip(pad_char)

def load_rationales(filepath, verbose=True):
    rationales = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            # line consists of "i [rationale]"
            rationale = line.split(' ')[-1]
            rationales.append(rationale)
    if verbose:
        print('Loaded %d rationales' % len(rationales))
    return rationales

def save_distance_matrix(dists, outpath):
    np.savetxt(outpath, dists, fmt='%d')

def levenshtein_dist(s1, s2):
    return Levenshtein.distance(s1, s2)

def compute_distances(rationales):
    dists = np.zeros((len(rationales), len(rationales)))
    for i in range(len(rationales)):
        r1 = truncate_rationale(rationales[i])
        for j in range(i+1):
            r2 = truncate_rationale(rationales[j])
            dist = levenshtein_dist(r1, r2)
            dists[i, j] = dist
            dists[j, i] = dist
    return dists

def main():
    if len(sys.argv) <= 1:
        raise ValueError('Must add directory path as command line argument.')

    rationales_dir = sys.argv[1]
    rationales_file = os.path.join(rationales_dir, RATIONALES_FILENAME)

    # Load rationales from file
    rationales = load_rationales(rationales_file, verbose=VERBOSE)

    # Compute pairwise distances for rationales
    distances = compute_distances(rationales)

    # Write distance matrix to file
    outfile_path = os.path.join(rationales_dir, OUTFILE_NAME)
    save_distance_matrix(distances, outfile_path)


if __name__ == '__main__':
    main()
