import os
import sys
import argparse
import numpy as np

LEVENSHTEIN_CODE_DIR = 'packages/Levenshtein'
sys.path.insert(0, os.path.abspath(LEVENSHTEIN_CODE_DIR))
import Levenshtein


# Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-padchar', type=str, default='J')
    parser.add_argument('-infile', type=str, required=True)
    parser.add_argument('-outfile', type=str, required=True)

    parser.add_argument('--v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('--q', '--quiet', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    return args


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

def compute_distances(rationales, pad_char='N'):
    dists = np.zeros((len(rationales), len(rationales)))
    for i in range(len(rationales)):
        r1 = truncate_rationale(rationales[i], pad_char=pad_char)
        for j in range(i+1):
            r2 = truncate_rationale(rationales[j], pad_char=pad_char)
            dist = levenshtein_dist(r1, r2)
            dists[i, j] = dist
            dists[j, i] = dist
    return dists

def main():
    # Parse command line args
    args = parse_args()

    # Set params
    VERBOSE = args.verbose
    RATIONALES_FILE = args.infile
    OUTFILE = args.outfile
    PAD_CHAR = args.padchar

    # Load rationales from file
    rationales = load_rationales(RATIONALES_FILE, verbose=VERBOSE)

    # Compute pairwise distances for rationales
    distances = compute_distances(rationales, pad_char=PAD_CHAR)

    # Write distance matrix to file
    save_distance_matrix(distances, OUTFILE)


if __name__ == '__main__':
    main()
