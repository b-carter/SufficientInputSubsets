import numpy as np
import os
import gzip
import json
import copy
import argparse
import itertools

# For headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sis
import sis_visualizations as visualizations
import lime_helper
from sis import eval_bin_class_model
from python_utils import redirect_stdout, redirect_stderr
from numpy_encoder import NumpyJSONEncoder
from rationale_objects import DNASequence, DNASequenceContainer, Rationale, \
                              SIS_RATIONALE_KEY, \
                              IG_SUFF_RATIONALE_KEY, \
                              IG_TOP_RATIONALE_KEY, \
                              IG_FIXED_RATIONALE_KEY, \
                              LIME_FIXED_RATIONALE_KEY, \
                              PERTURB_FIXED_RATIONALE_KEY
from packages.IntegratedGradients.IntegratedGradients import integrated_gradients

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding
from keras.optimizers import Adam, Adadelta
from keras.preprocessing import sequence, text

from sklearn.model_selection import train_test_split
from scipy.stats import ranksums, entropy


#####################################
#        GLOBALS/HELPERS
#####################################
BASES = ['A', 'C', 'G', 'T']

NUC_TO_ONEHOT = {'A': np.array([1, 0, 0, 0]).reshape(-1, 1),
                 'C': np.array([0, 1, 0, 0]).reshape(-1, 1),
                 'G': np.array([0, 0, 1, 0]).reshape(-1, 1),
                 'T': np.array([0, 0, 0, 1]).reshape(-1, 1)}

NUC_TO_ONEHOT_WITH_N = copy.deepcopy(NUC_TO_ONEHOT)
NUC_TO_ONEHOT_WITH_N['N'] = np.array([0.25, 0.25, 0.25, 0.25]).reshape(-1, 1)

KNOWN_MOTIFS_MAP_PATH = 'data/motif/known_motifs/map'
KNOWN_MOTIFS_MEME_PATH = 'data/motif/known_motifs/ENCODEmotif'


ENCODE_FAST_MAPPING = {k: v for v, k in enumerate(BASES+['N'])}
ENCODE_FAST_EMBEDDINGS = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1],
                                   [0.25, 0.25, 0.25, 0.25]])

# Returns numpy array of shape 101 x 4 (where 101 is the length of seq)
def encode_seq(seq):
    return np.hstack((NUC_TO_ONEHOT[c] for c in seq)).T

# Returns numpy array of shape 101 x 4 (where 101 is the length of seq)
# Difference from `encode_seq()` is that this replaces 'N' characters with
#   uniform vectors, whereas encode_seq would throw a key error
def encode_rationale(seq):
    return np.hstack((NUC_TO_ONEHOT_WITH_N[c] for c in seq)).T

# Faster implementation (used in LIME pipeline for better performance)
def encode_rationale_fast(seq):
    base_map = [ENCODE_FAST_MAPPING[c] for c in seq]
    return ENCODE_FAST_EMBEDDINGS[base_map]

def decode_seq(x):
    base_idxs = np.argmax(x, axis=1)
    return [BASES[i] for i in base_idxs]

# Rationale vector is a one-hot array, for each position if element is
#   in rationale or not
def compute_rationale_vector(rationale_idxs, dim=101):
    rv = np.zeros(dim)
    rv[rationale_idxs] = 1
    return rv

# Converts decoded_seq and rationale_idxs into a string containing rationale
# Replaces any non-rationale characters with `nonrationale` (default is '-')
def plaintext_rationale(decoded_seq, rationale_idxs, joiner='',
                         nonrationale='-'):
    rationale_idxs_set = set(rationale_idxs)
    text = [decoded_seq[i] if i in rationale_idxs_set else nonrationale \
                for i in range(len(decoded_seq))]
    return joiner.join(text)

# Truncates a rationale string so "NNNATNGNN" becomes just "ATNG"
# (Removes and leading and trailing N's, preserves internal N's)
def truncate_rationale(rationale, pad_char='N'):
    return rationale.strip(pad_char)


#####################################
#        ARG PARSING
#####################################
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='1')

    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-thresholdpercentile', type=int, default=90)
    parser.add_argument('-databasedir', type=str,
                         default='data/motif/data')
    parser.add_argument('-outbasedir', type=str,
                         default='rationale_results/motif')
    parser.add_argument('-task', type=str, required=True)
    parser.add_argument('-log', type=str)

    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=False)
    # parser.add_argument('--tunehyperparams', dest='tunehyperparams', action='store_true')
    # parser.set_defaults(tunehyperparams=False)
    parser.add_argument('--runsis', dest='runsis', action='store_true')
    parser.set_defaults(runsis=False)
    parser.add_argument('--runalternatives', dest='runalternatives', action='store_true')
    parser.set_defaults(runalternatives=False)
    parser.add_argument('--comparemethods', dest='comparemethods',
                          action='store_true')
    parser.set_defaults(comparemethods=False)

    parser.add_argument('--v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('--q', '--quiet', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    return args


#####################################
#        PREPROCESSING
#####################################
def load_tf_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parsed_line = line.split()
            seq_id, seq, label = parsed_line
            label = int(label)
            data.append((seq_id, seq, label))
    return data

# input lists contain (seq_id, seq, label) triples from `load_tf_data`
# outputs new `test_data` list ensuring that no elements in
#   `test_data` contain the same seq as any input in `train_data`
def remove_test_duplicates(train_data, test_data, verbose=False):
    train_seqs = set((t[1] for t in train_data))
    filtered_test_data = [t for t in test_data if t[1] not in train_seqs]
    if verbose:
        num_filtered = len(test_data) - len(filtered_test_data)
        print('Removed %d test points found in train set.' % num_filtered)
    return filtered_test_data

# `data` is a list of (seq_id, seq, label) tuples
# Returns: X -- array of same length with (1 x 4 x n) one-hot representation
#                 of each seq
#          y -- 1-D array of labels
#    as (X, y)
def parse_tf_data(data, verbose=False):
    X = []
    y = []
    num_errors = 0
    lines_read = 0
    for seq_id, seq, label in data:
        lines_read += 1
        try:
            encoded_seq = encode_seq(seq)
        except KeyError:  # "N" in the input sequence, throw sequence away
            num_errors += 1
            continue
        X.append(encoded_seq)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    if verbose:
        print('Read %d lines, %d key errors' % (lines_read, num_errors))
    return (X, y)

def create_val_split(X_train, y_train, test_size=0.125, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                       test_size=0.125,
                                                       random_state=42)
    return (X_train, X_val, y_train, y_val)


#####################################
#        MODELS
#####################################
# Initialize Keras model for 1layer_128motif CNN architecture
def make_tf_binding_cnn_model(input_shape, filters=128, kernel_size=24,
                               dropout=0.5, optimizer=None):
    model = Sequential()
    model.add(Conv1D(filters, kernel_size,
                      padding='same',
                      input_shape=input_shape,
                      activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer is None:
        optimizer = Adadelta()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

# Only used by hyperparameter tuning
def make_adadelta_optimizer(epsilon=None, rho=0.95):
    optimizer = Adadelta(epsilon=epsilon, rho=rho)
    return optimizer

def make_empty_sequence(dims=(101, 4), fill=0.25):
    empty = np.zeros(dims)
    empty.fill(fill)
    return empty

def make_replacement_embedding(dims=4, fill=0.25):
    vec = np.zeros(dims)
    vec.fill(fill)
    return vec


#####################################
#        EVALUATION
#####################################
# Returns a dictionary of summary statistics on vals
def make_stats_dict(vals):
    res = {}
    res['mean'] = np.mean(vals)
    res['median'] = np.median(vals)
    res['min'] = np.min(vals)
    res['max'] = np.max(vals)
    res['std'] = np.std(vals)
    return res

def eval_tf_binding_model(model, X_train, y_train, X_val, y_val,
                            X_test, y_test, verbose=True):
    train_stats = {}

    train_acc, train_auc = eval_bin_class_model(model, X_train, y_train,
                                                    verbose=False)
    train_loss = model.evaluate(x=X_train, y=y_train, verbose=0)[0]
    if verbose:
        print('')
        print('Performance of best model (lowest val loss) ...')
        print('')
        print('Train Accuracy: %.3f' % train_acc)
        print('Train AUC: %.3f' % train_auc)
        print('Train Loss: %.3f' % train_loss)
        print('')

    val_acc, val_auc = eval_bin_class_model(model, X_val, y_val,
                                            verbose=False)
    val_loss = model.evaluate(x=X_val, y=y_val, verbose=0)[0]
    if verbose:
        print('Val Accuracy: %.3f' % val_acc)
        print('Val AUC: %.3f' % val_auc)
        print('Val Loss: %.3f' % val_loss)
        print('')

    test_acc, test_auc = eval_bin_class_model(model, X_test, y_test,
                                            verbose=False)
    test_loss = model.evaluate(x=X_test, y=y_test, verbose=0)[0]
    if verbose:
        print('Test Accuracy: %.3f' % test_acc)
        print('Test AUC: %.3f' % test_auc)
        print('Test Loss: %.3f' % test_loss)
        print('')

    if not 'train' in train_stats:
        train_stats['train'] = {}
    train_stats['train']['accuracy'] = train_acc
    train_stats['train']['auc'] = train_auc
    train_stats['train']['loss'] = train_loss
    train_stats['train']['size'] = len(y_train)
    if not 'val' in train_stats:
        train_stats['val'] = {}
    train_stats['val']['accuracy'] = val_acc
    train_stats['val']['auc'] = val_auc
    train_stats['val']['loss'] = val_loss
    train_stats['val']['size'] = len(y_val)
    if not 'test' in train_stats:
        train_stats['test'] = {}
    train_stats['test']['accuracy'] = test_acc
    train_stats['test']['auc'] = test_auc
    train_stats['test']['loss'] = test_loss
    train_stats['test']['size'] = len(y_test)

    empty = np.zeros((101, 4))
    all_zeros_pred = float(model.predict(np.array([empty]))[0])
    empty.fill(0.25)
    all_025s_pred = float(model.predict(np.array([empty]))[0])
    y_train_mean = np.mean(y_train)

    train_stats['all_zeros_pred'] = all_zeros_pred
    train_stats['all_025s_pred'] = all_025s_pred
    train_stats['y_train_mean'] = y_train_mean

    if verbose:
        print('Pred. on all zeros: ', all_zeros_pred)
        print('')
        print('Pred on all 0.25s: ', all_025s_pred)
        print('')
        print('y-train mean: ', y_train_mean)

    return train_stats

# Pad motif (n x 4) with `replacement` vectors before and after so total
#   size is (total_len x 4).
# Used for computing KL Divergence of aligned known motif to rationales.
def pad_motif(motif, start_idx, replacement, total_len=101):
    num_left_pad = start_idx
    num_right_pad = total_len - motif.shape[0] - num_left_pad
    left_pad = np.repeat(replacement.reshape(1, -1), num_left_pad, axis=0)
    right_pad = np.repeat(replacement.reshape(1, -1), num_right_pad, axis=0)
    padded_motif = np.vstack((left_pad, motif, right_pad))
    assert(padded_motif.shape[0] == total_len)
    assert(padded_motif.shape[1] == motif.shape[1])
    return padded_motif

# Find optimal alignment of motif in rationale, using sliding window
# rationale_window is same shape as motif
# computes log likelihood of rationale_window under motif
def compute_motif_log_likelihood(motif, rationale_window):
    prod = motif * rationale_window  # element-wise multiplication
    expectations = prod.sum(axis=1)  # sum the 4 values at each position
    log_likelihood = np.log2(expectations).sum()
    return log_likelihood

# Assumes motif is (m x 4) and rationale (n x 4), n >= m
# Returns: starting index motif optimally aligned with rationale
#          and score as (start, score) tuple
def align_motif_rationale(motif, rationale, normalize=True):
    best_align_start = None
    best_align_score = None
    motif_len = motif.shape[0]
    rationale_len = rationale.shape[0]
    for start in range(rationale_len - motif_len + 1):
        rationale_window = rationale[start:start+motif_len, :]
        window_score = compute_motif_log_likelihood(motif, rationale_window)
        if best_align_score is None or window_score > best_align_score:
            best_align_score = window_score
            best_align_start = start
    if normalize:
        best_align_score /= float(motif_len)
    return best_align_start, best_align_score

# Takes padded_motif and rationale as inputs (same shape)
#   and computes KL Divergence of the two dists at each position
# Returns 1D array of same length containing KL values
# Assumes P = rationale and Q = known motif
def kl_div_motif_rationale(padded_motif, rationale):
    assert(padded_motif.shape == rationale.shape)
    kl_vals = []
    # TODO: make this more efficient/vectorize
    for i in range(padded_motif.shape[0]):
        P = rationale[i]
        Q = padded_motif[i]
        kl = entropy(P, qk=Q)
        kl_vals.append(kl)
    return np.array(kl_vals)

# Compute entropy along each position of motif and sum across positions
# Assumes motif has shape (n x 4)
def motif_entropy(motif):
    entropy_vals = []
    for i in range(motif.shape[0]):
        ent = entropy(motif[i])
        entropy_vals.append(ent)
    return np.sum(entropy_vals)

# Levenshtein (edit) distance
def levenshtein_distance(s1, s2):
    # From: https://stackoverflow.com/questions/2460177/edit-distance-in-python
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1],
                                           distances_[-1])))
        distances = distances_
    return distances[-1]

# Was used for testing, not used any more
def predict_on_motif(motif, replacement, model):
    start_idx = 101-len(motif) #int((101 - len(motif)) / float(2))
    # Use the most likely base at each position to construct discrete motif
    motif_discrete = np.zeros(motif.shape)
    motif_discrete[np.where(motif == np.max(motif, axis=1, keepdims=True))] = 1
    padded_motif = pad_motif(motif_discrete, start_idx, replacement)
    pred = sis.predict_for_embed_sequence([padded_motif], model)[0]
    print(padded_motif)
    print(pred)
    return pred


#####################################
#        I/O
#####################################
def save_idxs(idxs, outfile):
    with open(outfile, 'w') as f:
        map_data = '\n'.join((str(i) for i in idxs))
        f.write(map_data)

def load_idxs(filepath):
    idxs = []
    with open(filepath, 'r') as f:
        for line in f:
            i = int(line.strip())
            idxs.append(i)
    return idxs

# Rationales is a list of (i, rationale_str) tuples
def save_rationale_texts(rationales, outfile):
    lines = ['%d %s' % (i, r) for i, r in rationales]
    with open(outfile, 'w') as f:
        map_data = '\n'.join(lines)
        f.write(map_data)

# save stats dictionary to json
def save_stats(stats, filepath):
    with open(filepath, 'w') as f:
        json.dump(stats, f, cls=NumpyJSONEncoder)

def load_stats(filepath):
    with open(filepath, 'r') as f:
        stats = json.load(f)
        return stats

def load_motifs_map(path):
    motifs_map = {}
    with open(path, 'r') as f:
        for line in f:
            motif, loc = line.strip().split()
            motifs_map[motif] = loc
    return motifs_map

def parse_meme(filepath, replace_zeros_eps=None):
    res = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split('\t')
            if len(data) != 4:  # line does not contain floats, not part of matrix
                continue
            vals = [float(x.strip()) for x in data]
            res.append(vals)
    res = np.array(res, dtype='float32')
    if replace_zeros_eps is not None:
        res[res == 0] = replace_zeros_eps
        res = res / np.linalg.norm(res, axis=1, ord=1, keepdims=True)
    return res


#####################################
#        MAIN
#####################################
def main():
    # Get command line args
    args = parse_args()

    # Configure TF/Keras
    sis.tf_config(cuda_visible_devices=args.gpu)

    # Set params
    VERBOSE = args.verbose
    KERAS_VERBOSE = 2 if VERBOSE else 0  # if verbose, hide progress bar

    TASK = args.task  # `motif_occupancy` or `motif_discovery`
    THRESHOLD_PERCENTILE = args.thresholdpercentile
    DATASET = args.dataset
    DATA_BASE_DIR = args.databasedir
    OUT_BASE_DIR = args.outbasedir

    DATA_DIR = os.path.join(DATA_BASE_DIR, TASK, DATASET)
    OUT_DIR = os.path.join(OUT_BASE_DIR, TASK, DATASET)

    # Make directories to OUT_DIR path if not exists
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    LOG = args.log
    if LOG is not None:
        LOG_PATH = os.path.join(OUT_DIR, LOG)
        redirect_stdout(LOG_PATH)

        err_path = os.path.join(OUT_DIR, 'stderr.txt')
        redirect_stderr(err_path)

    train_data_path = os.path.join(DATA_DIR, 'train.data')
    test_data_path = os.path.join(DATA_DIR, 'test.data')

    train_data = load_tf_data(train_data_path)
    test_data = load_tf_data(test_data_path)
    test_data = remove_test_duplicates(train_data, test_data, verbose=VERBOSE)

    X_train, y_train = parse_tf_data(train_data, verbose=VERBOSE)
    X_test, y_test = parse_tf_data(test_data, verbose=VERBOSE)

    # Take 1/8 of training data as validation data (following paper)
    X_train, X_val, y_train, y_val = create_val_split(X_train, y_train)

    assert(len(X_train) == len(y_train))
    assert(len(X_val) == len(y_val))
    assert(len(X_test) == len(y_test))

    if VERBOSE:
        print('# Train: ', len(X_train))
        print('# Val: ', len(X_val))
        print('# Test: ', len(X_test))
        print('')

    input_shape = X_train[0].shape

    model = None
    model_out_name = '1layer_128motif.h5'
    model_path = os.path.join(OUT_DIR, model_out_name)

    test_predictions = None
    threshold = None
    pos_threshold_f = None

    # if args.tunehyperparams:
    #     tuning_stats = {}
    #     tuning_stats_path = os.path.join(OUT_DIR, 'hyperparam_tuning.json')

    #     DROPOUT = [0.1, 0.5, 0.75]
    #     DELTA = [1e-04, 1e-06, 1e-08]
    #     MOMENT = [0.9, 0.99, 0.999]

    #     best_val_loss = None
    #     best_params = None

    #     for dropout, delta, moment in itertools.product(DROPOUT, DELTA, MOMENT):
    #         dropout_str = '%.2f' % dropout
    #         delta_str = '%.03e' % delta
    #         moment_str = '%.3f' % moment
    #         if VERBOSE:
    #             print('Dropout = %s, Delta = %s, Moment = %s' % \
    #                       (dropout_str, delta_str, moment_str))

    #         optimizer = make_adadelta_optimizer(epsilon=delta, rho=moment)
    #         model = make_tf_binding_cnn_model(input_shape, dropout=dropout,
    #                                           optimizer=optimizer)
    #         if VERBOSE:
    #             print(model.summary())

    #         model.fit(X_train, y_train,
    #                   validation_data=(X_val, y_val),
    #                   epochs=5,
    #                   batch_size=100,
    #                   callbacks=[],
    #                   verbose=KERAS_VERBOSE)

    #         model_stats = eval_tf_binding_model(model, X_train, y_train, X_val,
    #                                             y_val, X_test, y_test,
    #                                             verbose=VERBOSE)

    #         val_loss = model_stats['val']['loss']
    #         if best_val_loss is None or val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             best_params = {'dropout' : dropout_str,
    #                            'delta' : delta_str,
    #                            'moment' : moment_str}

    #         if 'dropout' not in tuning_stats:
    #             tuning_stats['dropout'] = {}
    #         if dropout_str not in tuning_stats['dropout']:
    #             tuning_stats['dropout'][dropout_str] = {}
    #         dropout_dict = tuning_stats['dropout'][dropout_str]
    #         if 'delta' not in dropout_dict:
    #             dropout_dict['delta'] = {}
    #         if delta_str not in dropout_dict['delta']:
    #             dropout_dict['delta'][delta_str] = {}
    #         delta_dict = dropout_dict['delta'][delta_str]
    #         if 'moment' not in delta_dict:
    #             delta_dict['moment'] = {}
    #         if moment_str not in delta_dict['moment']:
    #             delta_dict['moment'][moment_str] = {}
    #         delta_dict['moment'][moment_str] = model_stats

    #     tuning_stats['best_params'] = best_params

    #     # Save train stats dictionary to json
    #     save_stats(tuning_stats, tuning_stats_path)
    #     if VERBOSE:
    #         print('Dumped hyperparam tuning results to json file.\n')


    if args.train:
        train_stats_path = os.path.join(OUT_DIR, 'train_stats.json')

        model = make_tf_binding_cnn_model(input_shape)
        if VERBOSE:
            print(model.summary())
            print('')

        checkpointer = ModelCheckpoint(filepath=model_path,
                                       verbose=int(VERBOSE),
                                       monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=True)

        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=10,
                  batch_size=128,
                  callbacks=[checkpointer],
                  verbose=KERAS_VERBOSE)

        # Reload the best model from the checkpoint
        model.load_weights(model_path)

        train_stats = eval_tf_binding_model(model, X_train, y_train, X_val,
                                            y_val, X_test, y_test,
                                            verbose=VERBOSE)

        # Plots of predictive distributions
        train_predictions = model.predict(X_train).flatten()
        val_predictions = model.predict(X_val).flatten()
        test_predictions = model.predict(X_test).flatten()

        threshold = float(np.percentile(test_predictions, THRESHOLD_PERCENTILE))

        if VERBOSE:
            print('Computed threshold as %d percentile: %f' % \
                    (THRESHOLD_PERCENTILE, threshold))

        train_stats['threshold'] = threshold

        visualizations.plot_predictive_dist(train_predictions,
                        title='Predictive Distribution on Train Set',
                        savepath=os.path.join(OUT_DIR, 'predictive_train.png'))

        visualizations.plot_predictive_dist(val_predictions,
                        title='Predictive Distribution on Validation Set',
                        savepath=os.path.join(OUT_DIR, 'predictive_val.png'))

        visualizations.plot_predictive_dist(test_predictions,
                        title='Predictive Distribution on Test Set',
                        vertlines=[threshold],
                        savepath=os.path.join(OUT_DIR, 'predictive_test.png'))

        if VERBOSE:
            print('\nSaved plots of predictive distributions.\n')

        # Save train stats dictionary to json
        save_stats(train_stats, train_stats_path)
        if VERBOSE:
            print('Dumped train stats to json file.\n')

    # Load the model if not loaded yet
    if model is None:
        model = make_tf_binding_cnn_model(input_shape)
        model.load_weights(model_path)

    interesting_points_mapfile = os.path.join(OUT_DIR, 'test_map.txt')
    interesting_test_idxs = None

    container_path = os.path.join(OUT_DIR, 'rationales_container')
    container = None

    if args.runsis:
        if VERBOSE:
            print('Starting SIS...')

        replacement_embedding = make_replacement_embedding()
        if VERBOSE:
            print('\nReplacement embedding: ', str(replacement_embedding))

        if test_predictions is None:
            test_predictions = model.predict(X_test).flatten()

        if threshold is None:
            threshold = np.percentile(test_predictions, THRESHOLD_PERCENTILE)
            pos_threshold_f = lambda x: x >= threshold

        interesting_test_idxs = np.where(test_predictions >= threshold)[0]
        # In case any numpy cast/rounding issues, ensure that model predicts
        #   above threshold on all of these:
        interesting_test_idxs = [i for i in interesting_test_idxs if \
            pos_threshold_f(sis.predict_for_embed_sequence(
                            [X_test[i]], model)[0])]

        if VERBOSE:
            print('# interesting test points (pred >= %f): %d' % \
                    (threshold, len(interesting_test_idxs)))

        # Dump the indices of the "interesting" test points to a map file
        save_idxs(interesting_test_idxs, interesting_points_mapfile)

        # Create new container
        container = DNASequenceContainer(threshold=threshold,
            trained_model_path=model_path)

        # Run SIS
        for i in interesting_test_idxs:
            x = X_test[i]
            seq = DNASequence(x=x, i=i,
                                replacement=replacement_embedding,
                                threshold=threshold,
                                threshold_f=pos_threshold_f)
            # Compute all SIS rationales
            seq.run_sis_rationales(model, first_only=False,
                verbose=False)
            seq.set_predictions(
                model,
                [seq.get_rationales(SIS_RATIONALE_KEY)[0]])
            container.add_sequence(seq)

        # Dump container to file
        container.dump_data(container_path)

        if VERBOSE:
            print('\nSaved container with SIS rationales.\n')


    # Load container if not yet loaded
    if container is None and (args.runalternatives or args.comparemethods):
        container = DNASequenceContainer.load_data(container_path)

    if args.runalternatives:
        if VERBOSE:
            print('Running alternative methods...')

        ig = integrated_gradients(model)
        igs_baseline = make_empty_sequence()
        igs_top_baseline = make_empty_sequence(fill=0)
        lime_pipeline = lime_helper.make_pipeline_dna_seq(model,
            encode_rationale_fast)

        # Run alternative methods
        for seq in container.get_sequences():
            seq.run_integrated_gradients_rationale(ig, model, igs_baseline,
                verbose=False)
            seq.run_integrated_gradients_top_rationale(ig, model,
                igs_top_baseline, verbose=False)
            seq.run_integrated_gradients_fixed_length_rationale(ig, model,
                igs_top_baseline, verbose=False)
            seq.run_lime_fixed_length_rationale(lime_pipeline, decode_seq,
                model, verbose=False)
            seq.run_perturb_fixed_length_rationale(model, verbose=False)

        # Dump container to file, with IG rationales
        container.dump_data(container_path)

        if VERBOSE:
            print('\nSaved container with alternative methods rationales.\n')


    stats = {}
    stats_path = os.path.join(OUT_DIR, 'stats.json')

    if args.comparemethods:
        if VERBOSE:
            print('Starting comparison analysis of SIS and IG...')

        threshold = container.get_threshold()
        stats['threshold'] = threshold

        # Compute stats on number of sufficient rationales
        sis_num_suffic_rationales = [len(s.get_rationales(SIS_RATIONALE_KEY)) \
                                            for s in container.get_sequences()]
        stats['num_suffic_rationales'] = make_stats_dict(
            sis_num_suffic_rationales)

        # Use all SIS rationales
        rationale_lens_sis = []
        for s in container.get_sequences():
            for rationale in s.get_rationales(SIS_RATIONALE_KEY):
                rationale_lens_sis.append(len(rationale))
        rationale_lens_ig = [len(s.get_rationales(IG_SUFF_RATIONALE_KEY)[0]) \
                                    for s in container.get_sequences()]
        rationale_lens_top_ig = [len(s.get_rationales(IG_TOP_RATIONALE_KEY)[0]) \
                                    for s in container.get_sequences()]
        rationale_lens_fixed_length_ig = [len(s.get_rationales(IG_FIXED_RATIONALE_KEY)[0]) \
                                    for s in container.get_sequences()]
        rationale_lens_fixed_length_lime = [len(s.get_rationales(LIME_FIXED_RATIONALE_KEY)[0]) \
                                    for s in container.get_sequences()]
        rationale_lens_fixed_length_pert = [len(s.get_rationales(PERTURB_FIXED_RATIONALE_KEY)[0]) \
                                    for s in container.get_sequences()]

        # Create histogram and scatter plots comparing rationale lengths
        lens_hist_data = [(rationale_lens_sis, 10, 'SIS'),
                          (rationale_lens_ig, 25, 'IG'),
                          (rationale_lens_top_ig, 25, 'Top IG')]
        visualizations.plot_hist(lens_hist_data,
                                 title='Rationale Lengths',
                                 xlabel='Rationale Lengths',
                                 ylabel='Normalized Frequency',
                                 normed=True,
                                 savepath=os.path.join(OUT_DIR,
                                     'rationale_lens_hist.png'))

        # Ranksum test comparing the rationale lengths
        rs_stat_ig, rs_p_ig = ranksums(
            rationale_lens_sis, rationale_lens_ig)
        rs_stat_top_ig, rs_p_top_ig = ranksums(
            rationale_lens_sis, rationale_lens_top_ig)

        if not 'rationale_lengths' in stats:
            stats['rationale_lengths'] = {}
        stats['rationale_lengths']['sis'] = make_stats_dict(
            rationale_lens_sis)
        stats['rationale_lengths']['ig'] = make_stats_dict(
            rationale_lens_ig)
        stats['rationale_lengths']['top_ig'] = make_stats_dict(
            rationale_lens_top_ig)
        stats['rationale_lengths']['fixed_length_ig'] = make_stats_dict(
            rationale_lens_fixed_length_ig)
        stats['rationale_lengths']['fixed_length_lime'] = make_stats_dict(
            rationale_lens_fixed_length_lime)
        stats['rationale_lengths']['fixed_length_pert'] = make_stats_dict(
            rationale_lens_fixed_length_pert)
        stats['rationale_lengths']['sis_vs_ig'] = {}
        stats['rationale_lengths']['sis_vs_ig']['ranksum_statistic'] = float(rs_stat_ig)
        stats['rationale_lengths']['sis_vs_ig']['ranksum_pvalue'] = float(rs_p_ig)
        stats['rationale_lengths']['sis_vs_top_ig'] = {}
        stats['rationale_lengths']['sis_vs_top_ig']['ranksum_statistic'] = float(rs_stat_top_ig)
        stats['rationale_lengths']['sis_vs_top_ig']['ranksum_pvalue'] = float(rs_p_top_ig)

        if VERBOSE:
            print('Ranksums test comparing rationale lengths from SIS',
                    'and IG : stat=%f, p=%f' % (rs_stat_ig, rs_p_ig))
            print('Ranksums test comparing rationale lengths from SIS',
                    'and Top IG : stat=%f, p=%f' % \
                    (rs_stat_top_ig, rs_p_top_ig))

        # Compute statistics on Top IG rationale predictions where all
        #  non-rationale features are masked:
        #  -what percentage have predicted scores above threshold?
        #  -predictive distribution statistics (mean, median, etc.)
        top_ig_rationale_batch = []
        for seq in container.get_sequences():
            x_top_ig = seq.get_x_rationale_only(
                [seq.get_rationales(IG_TOP_RATIONALE_KEY)[0]])
            top_ig_rationale_batch.append(x_top_ig)
        top_ig_predictions = sis.predict_for_embed_sequence(
            top_ig_rationale_batch, model)
        frac_top_above = np.sum(
            top_ig_predictions >= threshold) / float(len(top_ig_predictions))
        stats['top_ig'] = {}
        stats['top_ig']['predictive_dist_stats'] = make_stats_dict(
            top_ig_predictions)
        stats['top_ig']['fraction_above_threshold'] = frac_top_above

        # Rationale-only predictions
        # Predictive distribution on rationales only
        stats['rationale_only_predictions'] = {}
        sis_x_rationales_only = []
        ig_x_rationales_only = []
        top_ig_x_rationales_only = []
        ig_fixed_length_x_rationales_only = []
        lime_fixed_length_x_rationales_only = []
        pert_fixed_length_x_rationales_only = []
        for seq in container.get_sequences():
            sis_rationales = seq.get_rationales(SIS_RATIONALE_KEY)
            for r in sis_rationales:
                sis_x_rationales_only.append(seq.get_x_rationale_only([r]))
            ig_x_rationales_only.append(seq.get_x_rationale_only(
                [seq.get_rationales(IG_SUFF_RATIONALE_KEY)[0]]))
            top_ig_x_rationales_only.append(seq.get_x_rationale_only(
                [seq.get_rationales(IG_TOP_RATIONALE_KEY)[0]]))
            ig_fixed_length_x_rationales_only.append(seq.get_x_rationale_only(
                [seq.get_rationales(IG_FIXED_RATIONALE_KEY)[0]]))
            lime_fixed_length_x_rationales_only.append(seq.get_x_rationale_only(
                [seq.get_rationales(LIME_FIXED_RATIONALE_KEY)[0]]))
            pert_fixed_length_x_rationales_only.append(seq.get_x_rationale_only(
                [seq.get_rationales(PERTURB_FIXED_RATIONALE_KEY)[0]]))
        sis_x_rationales_only_preds = sis.predict_for_embed_sequence(
            sis_x_rationales_only, model)
        ig_x_rationales_only_preds = sis.predict_for_embed_sequence(
            ig_x_rationales_only, model)
        top_ig_x_rationales_only_preds = sis.predict_for_embed_sequence(
            top_ig_x_rationales_only, model)
        ig_fixed_length_x_rationales_only_preds = sis.predict_for_embed_sequence(
            ig_fixed_length_x_rationales_only, model)
        lime_fixed_length_x_rationales_only_preds = sis.predict_for_embed_sequence(
            lime_fixed_length_x_rationales_only, model)
        pert_fixed_length_x_rationales_only_preds = sis.predict_for_embed_sequence(
            pert_fixed_length_x_rationales_only, model)
        stats['rationale_only_predictions']['sis'] = make_stats_dict(
            sis_x_rationales_only_preds)
        stats['rationale_only_predictions']['ig'] = make_stats_dict(
            ig_x_rationales_only_preds)
        stats['rationale_only_predictions']['top_ig'] = make_stats_dict(
            top_ig_x_rationales_only_preds)
        stats['rationale_only_predictions']['ig_fixed_length'] = make_stats_dict(
            ig_fixed_length_x_rationales_only_preds)
        stats['rationale_only_predictions']['lime_fixed_length'] = make_stats_dict(
            lime_fixed_length_x_rationales_only_preds)
        stats['rationale_only_predictions']['pert_fixed_length'] = make_stats_dict(
            pert_fixed_length_x_rationales_only_preds)

        # Perturbation analysis (removing individual elements, comparing
        #   effect from removing rationale elements vs. non-rationale elements)
        rationale_diffs_sis = []
        nonrationale_diffs_sis = []
        for seq in container.get_sequences():
            for rationale in seq.get_rationales(SIS_RATIONALE_KEY):
                rationale_diffs, nonrationale_diffs = seq.perturbation_rationale(
                    model, [rationale])
                rationale_diffs_sis.append(rationale_diffs)
                nonrationale_diffs_sis.append(nonrationale_diffs)

        rationale_diffs_sis_vals = np.concatenate(rationale_diffs_sis).ravel()
        nonrationale_diffs_sis_vals = np.concatenate(nonrationale_diffs_sis).ravel()
        rs_stat_pert_sis, rs_p_pert_sis = ranksums(
            rationale_diffs_sis_vals,
            nonrationale_diffs_sis_vals)

        if not 'perturbation' in stats:
            stats['perturbation'] = {}
        if not 'sis' in stats['perturbation']:
            stats['perturbation']['sis'] = {}
        stats['perturbation']['sis']['rationale'] = make_stats_dict(rationale_diffs_sis_vals)
        stats['perturbation']['sis']['nonrationale'] = make_stats_dict(nonrationale_diffs_sis_vals)
        stats['perturbation']['sis']['ranksum_statistic'] = float(rs_stat_pert_sis)
        stats['perturbation']['sis']['ranksum_pvalue'] = float(rs_p_pert_sis)

        if VERBOSE:
            print('Ranksums test comparing perturbation deltas from',
                    'SIS rationale vs non-rationale',
                    'stat=%f, p=%f' % (rs_stat_pert_sis, rs_p_pert_sis))

        pert_hist_sis = [(rationale_diffs_sis_vals, 25, 'Rationale'),
                            (nonrationale_diffs_sis_vals, 25, 'Non-rationale')]
        visualizations.plot_hist(pert_hist_sis,
                                 title='Perturbation Deltas, SIS',
                                 xlabel='original prediction - prediction without word',
                                 ylabel='Normalized Frequency',
                                 normed=True,
                                 savepath=os.path.join(OUT_DIR,
                                     'perturbation_hist_sis.png'))

        rationale_diffs_ig = []
        nonrationale_diffs_ig = []
        for seq in container.get_sequences():
            rationale_diffs, nonrationale_diffs = seq.perturbation_rationale(
                model,
                [seq.get_rationales(IG_SUFF_RATIONALE_KEY)[0]])
            rationale_diffs_ig.append(rationale_diffs)
            nonrationale_diffs_ig.append(nonrationale_diffs)

        rationale_diffs_ig_vals = np.concatenate(rationale_diffs_ig).ravel()
        nonrationale_diffs_ig_vals = np.concatenate(nonrationale_diffs_ig).ravel()
        rs_stat_pert_ig, rs_p_pert_ig = ranksums(
            rationale_diffs_ig_vals,
            nonrationale_diffs_ig_vals)

        if not 'ig' in stats['perturbation']:
            stats['perturbation']['ig'] = {}
        stats['perturbation']['ig']['rationale'] = make_stats_dict(rationale_diffs_ig_vals)
        stats['perturbation']['ig']['nonrationale'] = make_stats_dict(nonrationale_diffs_ig_vals)
        stats['perturbation']['ig']['ranksum_statistic'] = float(rs_stat_pert_ig)
        stats['perturbation']['ig']['ranksum_pvalue'] = float(rs_p_pert_ig)

        if VERBOSE:
            print('Ranksums test comparing perturbation deltas from',
                    'Integrated Gradients rationale vs non-rationale',
                    'stat=%f, p=%f' % (rs_stat_pert_ig, rs_p_pert_ig))
        pert_hist_ig = [(rationale_diffs_ig_vals, 25, 'Rationale'),
                        (nonrationale_diffs_ig_vals, 25, 'Non-rationale')]
        visualizations.plot_hist(pert_hist_ig,
                                 title='Perturbation Deltas, Integrated Gradients',
                                 xlabel='original prediction - prediction without word',
                                 ylabel='Normalized Frequency',
                                 normed=True,
                                 savepath=os.path.join(OUT_DIR,
                                     'perturbation_hist_ig.png'))

        rationale_diffs_top_ig = []
        nonrationale_diffs_top_ig = []
        for seq in container.get_sequences():
            rationale_diffs, nonrationale_diffs = seq.perturbation_rationale(
                model,
                [seq.get_rationales(IG_TOP_RATIONALE_KEY)[0]])
            rationale_diffs_top_ig.append(rationale_diffs)
            nonrationale_diffs_top_ig.append(nonrationale_diffs)

        rationale_diffs_top_ig_vals = np.concatenate(rationale_diffs_top_ig).ravel()
        nonrationale_diffs_top_ig_vals = np.concatenate(nonrationale_diffs_top_ig).ravel()
        rs_stat_pert_top_ig, rs_p_pert_top_ig = ranksums(
            rationale_diffs_top_ig_vals,
            nonrationale_diffs_top_ig_vals)

        if not 'top_ig' in stats['perturbation']:
            stats['perturbation']['top_ig'] = {}
        stats['perturbation']['top_ig']['rationale'] = make_stats_dict(
            rationale_diffs_top_ig_vals)
        stats['perturbation']['top_ig']['nonrationale'] = make_stats_dict(
            nonrationale_diffs_top_ig_vals)
        stats['perturbation']['top_ig']['ranksum_statistic'] = float(
            rs_stat_pert_top_ig)
        stats['perturbation']['top_ig']['ranksum_pvalue'] = float(
            rs_p_pert_top_ig)

        if VERBOSE:
            print('Ranksums test comparing perturbation deltas from',
                    'Top IG rationale vs non-rationale',
                    'stat=%f, p=%f' % \
                    (rs_stat_pert_top_ig, rs_p_pert_top_ig))
        pert_hist_top_ig = [(rationale_diffs_top_ig_vals, 25, 'Rationale'),
                                (nonrationale_diffs_top_ig_vals, 25, 'Non-rationale')]
        visualizations.plot_hist(pert_hist_top_ig,
                                 title='Perturbation Deltas, Top IG',
                                 xlabel='original prediction - prediction without word',
                                 ylabel='Normalized Frequency',
                                 normed=True,
                                 savepath=os.path.join(OUT_DIR,
                                     'perturbation_hist_top_ig.png'))

        rationale_diffs_ig_fixed_length = []
        nonrationale_diffs_ig_fixed_length = []
        rationale_diffs_lime_fixed_length = []
        nonrationale_diffs_lime_fixed_length = []
        rationale_diffs_pert_fixed_length = []
        nonrationale_diffs_pert_fixed_length = []
        for seq in container.get_sequences():
            rationale_diffs, nonrationale_diffs = seq.perturbation_rationale(
                model,
                [seq.get_rationales(IG_FIXED_RATIONALE_KEY)[0]])
            rationale_diffs_ig_fixed_length.append(rationale_diffs)
            nonrationale_diffs_ig_fixed_length.append(nonrationale_diffs)

            rationale_diffs, nonrationale_diffs = seq.perturbation_rationale(
                model,
                [seq.get_rationales(LIME_FIXED_RATIONALE_KEY)[0]])
            rationale_diffs_lime_fixed_length.append(rationale_diffs)
            nonrationale_diffs_lime_fixed_length.append(nonrationale_diffs)

            rationale_diffs, nonrationale_diffs = seq.perturbation_rationale(
                model,
                [seq.get_rationales(PERTURB_FIXED_RATIONALE_KEY)[0]])
            rationale_diffs_pert_fixed_length.append(rationale_diffs)
            nonrationale_diffs_pert_fixed_length.append(nonrationale_diffs)

        rationale_diffs_ig_fixed_length_vals = np.concatenate(rationale_diffs_ig_fixed_length).ravel()
        nonrationale_diffs_ig_fixed_length_vals = np.concatenate(nonrationale_diffs_ig_fixed_length).ravel()
        rs_stat_pert_ig_fixed_length, rs_p_pert_ig_fixed_length = ranksums(
            rationale_diffs_ig_fixed_length_vals,
            nonrationale_diffs_ig_fixed_length_vals)
        if not 'ig_fixed_length' in stats['perturbation']:
            stats['perturbation']['ig_fixed_length'] = {}
        stats['perturbation']['ig_fixed_length']['rationale'] = make_stats_dict(
            rationale_diffs_ig_fixed_length_vals)
        stats['perturbation']['ig_fixed_length']['nonrationale'] = make_stats_dict(
            nonrationale_diffs_ig_fixed_length_vals)
        stats['perturbation']['ig_fixed_length']['ranksum_statistic'] = float(
            rs_stat_pert_ig_fixed_length)
        stats['perturbation']['ig_fixed_length']['ranksum_pvalue'] = float(
            rs_p_pert_ig_fixed_length)

        if VERBOSE:
            print('Ranksums test comparing perturbation deltas from',
                    'IG fixed length rationale vs non-rationale',
                    'stat=%f, p=%f' % \
                    (rs_stat_pert_ig_fixed_length, rs_p_pert_ig_fixed_length))
        pert_hist_ig_fixed_length = [(rationale_diffs_ig_fixed_length_vals, 25, 'Rationale'),
                                (nonrationale_diffs_ig_fixed_length_vals, 25, 'Non-rationale')]
        visualizations.plot_hist(pert_hist_ig_fixed_length,
                                 title='Perturbation Deltas, IG Fixed Length',
                                 xlabel='original prediction - prediction without word',
                                 ylabel='Normalized Frequency',
                                 normed=True,
                                 savepath=os.path.join(OUT_DIR,
                                     'perturbation_hist_ig_fixed_length.png'))

        rationale_diffs_lime_fixed_length_vals = np.concatenate(rationale_diffs_lime_fixed_length).ravel()
        nonrationale_diffs_lime_fixed_length_vals = np.concatenate(nonrationale_diffs_lime_fixed_length).ravel()
        rs_stat_pert_lime_fixed_length, rs_p_pert_lime_fixed_length = ranksums(
            rationale_diffs_lime_fixed_length_vals,
            nonrationale_diffs_lime_fixed_length_vals)
        if not 'lime_fixed_length' in stats['perturbation']:
            stats['perturbation']['lime_fixed_length'] = {}
        stats['perturbation']['lime_fixed_length']['rationale'] = make_stats_dict(
            rationale_diffs_lime_fixed_length_vals)
        stats['perturbation']['lime_fixed_length']['nonrationale'] = make_stats_dict(
            nonrationale_diffs_lime_fixed_length_vals)
        stats['perturbation']['lime_fixed_length']['ranksum_statistic'] = float(
            rs_stat_pert_lime_fixed_length)
        stats['perturbation']['lime_fixed_length']['ranksum_pvalue'] = float(
            rs_p_pert_lime_fixed_length)

        if VERBOSE:
            print('Ranksums test comparing perturbation deltas from',
                    'LIME fixed length rationale vs non-rationale',
                    'stat=%f, p=%f' % \
                    (rs_stat_pert_lime_fixed_length, rs_p_pert_lime_fixed_length))
        pert_hist_lime_fixed_length = [(rationale_diffs_lime_fixed_length_vals, 25, 'Rationale'),
                                (nonrationale_diffs_lime_fixed_length_vals, 25, 'Non-rationale')]
        visualizations.plot_hist(pert_hist_lime_fixed_length,
                                 title='Perturbation Deltas, LIME Fixed Length',
                                 xlabel='original prediction - prediction without word',
                                 ylabel='Normalized Frequency',
                                 normed=True,
                                 savepath=os.path.join(OUT_DIR,
                                     'perturbation_hist_lime_fixed_length.png'))

        rationale_diffs_pert_fixed_length_vals = np.concatenate(rationale_diffs_pert_fixed_length).ravel()
        nonrationale_diffs_pert_fixed_length_vals = np.concatenate(nonrationale_diffs_pert_fixed_length).ravel()
        rs_stat_pert_pert_fixed_length, rs_p_pert_pert_fixed_length = ranksums(
            rationale_diffs_pert_fixed_length_vals,
            nonrationale_diffs_pert_fixed_length_vals)
        if not 'pert_fixed_length' in stats['perturbation']:
            stats['perturbation']['pert_fixed_length'] = {}
        stats['perturbation']['pert_fixed_length']['rationale'] = make_stats_dict(
            rationale_diffs_pert_fixed_length_vals)
        stats['perturbation']['pert_fixed_length']['nonrationale'] = make_stats_dict(
            nonrationale_diffs_pert_fixed_length_vals)
        stats['perturbation']['pert_fixed_length']['ranksum_statistic'] = float(
            rs_stat_pert_pert_fixed_length)
        stats['perturbation']['pert_fixed_length']['ranksum_pvalue'] = float(
            rs_p_pert_pert_fixed_length)

        if VERBOSE:
            print('Ranksums test comparing perturbation deltas from',
                    'Pert. fixed length rationale vs non-rationale',
                    'stat=%f, p=%f' % \
                    (rs_stat_pert_pert_fixed_length, rs_p_pert_pert_fixed_length))
        pert_hist_pert_fixed_length = [(rationale_diffs_pert_fixed_length_vals, 25, 'Rationale'),
                                (nonrationale_diffs_pert_fixed_length_vals, 25, 'Non-rationale')]
        visualizations.plot_hist(pert_hist_pert_fixed_length,
                                 title='Perturbation Deltas, Pert. Fixed Length',
                                 xlabel='original prediction - prediction without word',
                                 ylabel='Normalized Frequency',
                                 normed=True,
                                 savepath=os.path.join(OUT_DIR,
                                     'perturbation_hist_pert_fixed_length.png'))


        rs_stat_pert_bothrationales_sis_vs_ig, rs_p_pert_bothrationales_sis_vs_ig = ranksums(
            rationale_diffs_sis_vals,
            rationale_diffs_ig_vals)
        if VERBOSE:
            print('Ranksums test comparing perturbation deltas from',
                    'SIS rationales vs IG rationales', 'stat=%f, p=%f' % \
                    (rs_stat_pert_bothrationales_sis_vs_ig,
                     rs_p_pert_bothrationales_sis_vs_ig))
        rs_stat_pert_bothrationales_sis_vs_top_ig, rs_p_pert_bothrationales_sis_vs_top_ig = ranksums(
            rationale_diffs_sis_vals,
            rationale_diffs_top_ig_vals)
        if VERBOSE:
            print('Ranksums test comparing perturbation deltas from',
                    'SIS rationales vs Top IG rationales', 'stat=%f, p=%f' % \
                    (rs_stat_pert_bothrationales_sis_vs_top_ig,
                     rs_p_pert_bothrationales_sis_vs_top_ig))
        pert_hist_allrationales = [(rationale_diffs_sis_vals, 25, 'SIS Rationales'),
                                   (rationale_diffs_ig_vals, 25, 'Integrated Gradients Rationales'),
                                   (rationale_diffs_top_ig_vals, 25, 'Top IG Rationales')]
        visualizations.plot_hist(pert_hist_allrationales,
                                 title='Perturbation Deltas, Rationales Only',
                                 xlabel='original prediction - prediction without word',
                                 ylabel='Normalized Frequency',
                                 normed=True,
                                 savepath=os.path.join(OUT_DIR,
                                     'perturbation_hist_allrationales.png'))

        if not 'bothrationales' in stats['perturbation']:
            stats['perturbation']['bothrationales'] = {}
            stats['perturbation']['bothrationales']['sis_vs_ig'] = {}
            stats['perturbation']['bothrationales']['sis_vs_top_ig'] = {}
        stats['perturbation']['bothrationales']['sis_vs_ig']['ranksum_statistic'] = \
            float(rs_stat_pert_bothrationales_sis_vs_ig)
        stats['perturbation']['bothrationales']['sis_vs_ig']['ranksum_pvalue'] = \
            float(rs_p_pert_bothrationales_sis_vs_ig)
        stats['perturbation']['bothrationales']['sis_vs_top_ig']['ranksum_statistic'] = \
            float(rs_stat_pert_bothrationales_sis_vs_top_ig)
        stats['perturbation']['bothrationales']['sis_vs_top_ig']['ranksum_pvalue'] = \
            float(rs_p_pert_bothrationales_sis_vs_top_ig)

        # HTML output visualizing SIS and IG rationales
        html_path = os.path.join(OUT_DIR, 'rationales.html')

        header = '''
        <h2>TF Binding Rationales (test set)</h2>
        <p>Highlighting rationales identified by SIS and integrated gradients.
        Only showing examples in test set where prediction >= threshold (default >= 90th percentile).
        Highlight shade indicates base importance, darker means more important, with linear scaling.</p>
        '''
        header += '<p>Task: %s</p>' % TASK
        header += '<p>Dataset: %s</p>' % DATASET
        header += '<p>Threshold: %.5f</p>' % container.get_threshold()
        html = ''
        for seq in container.get_sequences():
            x = seq.get_x()
            decoded_seq = decode_seq(x)
            html += '<p>Example %d</p>' % seq.i
            html += '<p>Predicted: %.3f  /  Actual: %.3f<br>' % \
                        (seq.original_prediction, y_test[seq.i])
            sis_rationales = seq.get_rationales(SIS_RATIONALE_KEY)
            ig_rationale = seq.get_rationales(IG_SUFF_RATIONALE_KEY)[0]
            ig_top_rationale = seq.get_rationales(IG_TOP_RATIONALE_KEY)[0]
            # Rationale Lengths
            html += 'Rationale length SIS = %s<br>' % \
                        str([len(r) for r in sis_rationales])
            html += 'Rationale length IG = %d<br>' % len(ig_rationale)
            html += 'Rationale length Top IG = %d<br>' % len(ig_top_rationale)
            # SIS Rationales
            html += '<p>SIS:</p>'
            for i, sis_rationale in enumerate(sis_rationales):
                if len(sis_rationale) > 1:
                    html += '<p><i>Sufficient Rationale %d:</i></p>' % (i + 1)
                html += visualizations.highlight_annot_tf(decoded_seq,
                    sis_rationale)
            # IG Rationale
            html += '<p>(Suff.) Integrated Gradients:</p>'
            html += visualizations.highlight_annot_tf(decoded_seq,
                ig_rationale)
            # Top IG Rationale
            html += '<p>Top IG:</p>'
            html += visualizations.highlight_annot_tf(decoded_seq,
                ig_top_rationale)
            html += '<hr>'

        visualizations.save_html(html, html_path, header)

        if VERBOSE:
            print('Rationales html dumped to file.\n')

        # Dump rationales to text file, replace non-rationale chars with 'N'
        rationale_texts_sis = []
        rationale_texts_ig = []
        rationale_texts_top_ig = []
        rationale_texts_fixed_length_ig = []
        rationale_texts_fixed_length_lime = []
        rationale_texts_fixed_length_pert = []

        sis_rationales_path = os.path.join(OUT_DIR, 'rationales_sis.txt')
        ig_rationales_path = os.path.join(OUT_DIR, 'rationales_ig.txt')
        top_ig_rationales_path = os.path.join(OUT_DIR,
            'rationales_top_ig.txt')
        fixed_length_ig_rationales_path = os.path.join(OUT_DIR, 'rationales_fixed_length_ig.txt')
        fixed_length_lime_rationales_path = os.path.join(OUT_DIR, 'rationales_fixed_length_lime.txt')
        fixed_length_pert_rationales_path = os.path.join(OUT_DIR, 'rationales_fixed_length_pert.txt')

        for seq in container.get_sequences():
            x = seq.get_x()
            decoded_seq = decode_seq(x)
            # SIS Rationales
            sis_rationales = seq.get_rationales(SIS_RATIONALE_KEY)
            for rationale in sis_rationales:
                rationale_text_sis = plaintext_rationale(decoded_seq,
                                            rationale.get_elms(),
                                            nonrationale='N')
                rationale_texts_sis.append((seq.i, rationale_text_sis))
            # IG Rationale
            ig_rationale = seq.get_rationales(IG_SUFF_RATIONALE_KEY)[0]
            rationale_text_ig = plaintext_rationale(decoded_seq,
                                    ig_rationale.get_elms(),
                                    nonrationale='N')
            rationale_texts_ig.append((seq.i, rationale_text_ig))
            # Top IG Rationale
            top_ig_rationale = seq.get_rationales(
                IG_TOP_RATIONALE_KEY)[0]
            rationale_text_top_ig = plaintext_rationale(decoded_seq,
                                    top_ig_rationale.get_elms(),
                                    nonrationale='N')
            rationale_texts_top_ig.append((seq.i, rationale_text_top_ig))
            # Fixed length IG Rationale
            fixed_length_ig_rationale = seq.get_rationales(IG_FIXED_RATIONALE_KEY)[0]
            rationale_text_fixed_length_ig = plaintext_rationale(decoded_seq,
                                    fixed_length_ig_rationale.get_elms(),
                                    nonrationale='N')
            rationale_texts_fixed_length_ig.append((seq.i, rationale_text_fixed_length_ig))
            # Fixed length LIME Rationale
            fixed_length_lime_rationale = seq.get_rationales(LIME_FIXED_RATIONALE_KEY)[0]
            rationale_text_fixed_length_lime = plaintext_rationale(decoded_seq,
                                    fixed_length_lime_rationale.get_elms(),
                                    nonrationale='N')
            rationale_texts_fixed_length_lime.append((seq.i, rationale_text_fixed_length_lime))
            # Fixed length Pert Rationale
            fixed_length_pert_rationale = seq.get_rationales(PERTURB_FIXED_RATIONALE_KEY)[0]
            rationale_text_fixed_length_pert = plaintext_rationale(decoded_seq,
                                    fixed_length_pert_rationale.get_elms(),
                                    nonrationale='N')
            rationale_texts_fixed_length_pert.append((seq.i, rationale_text_fixed_length_pert))

        save_rationale_texts(rationale_texts_sis, sis_rationales_path)
        save_rationale_texts(rationale_texts_ig, ig_rationales_path)
        save_rationale_texts(rationale_texts_top_ig,
            top_ig_rationales_path)
        save_rationale_texts(rationale_texts_fixed_length_ig,
            fixed_length_ig_rationales_path)
        save_rationale_texts(rationale_texts_fixed_length_lime,
            fixed_length_lime_rationales_path)
        save_rationale_texts(rationale_texts_fixed_length_pert,
            fixed_length_pert_rationales_path)

        if VERBOSE:
            print('Rationale texts dumped to file.\n')

        # Aligning rationales to known motifs (from JASPAR), compute log
        #   likelihood per base under the model of the known motif
        motifs_map = load_motifs_map(KNOWN_MOTIFS_MAP_PATH)
        if DATASET in motifs_map:
            if VERBOSE:
                print('Have known motif. Aligning and finding likelihood.')

            meme_path = os.path.join(KNOWN_MOTIFS_MEME_PATH, motifs_map[DATASET])
            motif = parse_meme(meme_path, replace_zeros_eps=1e-6)

            # Add motif entropy into stats dict
            stats['known_motif'] = {}
            stats['known_motif']['entropy_sum'] = motif_entropy(motif)

            replacement = make_replacement_embedding()

            align_scores_sis = []
            for i, rationale in rationale_texts_sis:
                rationale_encoded = encode_rationale(rationale)
                alignment_res = align_motif_rationale(motif, rationale_encoded,
                                                        normalize=True)
                align_start_idx = alignment_res[0]
                padded_motif = pad_motif(motif, align_start_idx, replacement)
                kl_vals = kl_div_motif_rationale(padded_motif, rationale_encoded)
                align_score = np.sum(kl_vals)
                align_scores_sis.append(align_score)

            align_scores_ig = []
            for i, rationale in rationale_texts_ig:
                rationale_encoded = encode_rationale(rationale)
                alignment_res = align_motif_rationale(motif, rationale_encoded,
                                                        normalize=True)
                align_start_idx = alignment_res[0]
                padded_motif = pad_motif(motif, align_start_idx, replacement)
                kl_vals = kl_div_motif_rationale(padded_motif, rationale_encoded)
                align_score = np.sum(kl_vals)
                align_scores_ig.append(align_score)

            align_scores_top_ig = []
            for i, rationale in rationale_texts_top_ig:
                rationale_encoded = encode_rationale(rationale)
                alignment_res = align_motif_rationale(motif, rationale_encoded,
                                                        normalize=True)
                align_start_idx = alignment_res[0]
                padded_motif = pad_motif(motif, align_start_idx, replacement)
                kl_vals = kl_div_motif_rationale(padded_motif, rationale_encoded)
                align_score = np.sum(kl_vals)
                align_scores_top_ig.append(align_score)

            align_scores_fixed_length_ig = []
            for i, rationale in rationale_texts_fixed_length_ig:
                rationale_encoded = encode_rationale(rationale)
                alignment_res = align_motif_rationale(motif, rationale_encoded,
                                                        normalize=True)
                align_start_idx = alignment_res[0]
                padded_motif = pad_motif(motif, align_start_idx, replacement)
                kl_vals = kl_div_motif_rationale(padded_motif, rationale_encoded)
                align_score = np.sum(kl_vals)
                align_scores_fixed_length_ig.append(align_score)

            align_scores_fixed_length_lime = []
            for i, rationale in rationale_texts_fixed_length_lime:
                rationale_encoded = encode_rationale(rationale)
                alignment_res = align_motif_rationale(motif, rationale_encoded,
                                                        normalize=True)
                align_start_idx = alignment_res[0]
                padded_motif = pad_motif(motif, align_start_idx, replacement)
                kl_vals = kl_div_motif_rationale(padded_motif, rationale_encoded)
                align_score = np.sum(kl_vals)
                align_scores_fixed_length_lime.append(align_score)

            align_scores_fixed_length_pert = []
            for i, rationale in rationale_texts_fixed_length_pert:
                rationale_encoded = encode_rationale(rationale)
                alignment_res = align_motif_rationale(motif, rationale_encoded,
                                                        normalize=True)
                align_start_idx = alignment_res[0]
                padded_motif = pad_motif(motif, align_start_idx, replacement)
                kl_vals = kl_div_motif_rationale(padded_motif, rationale_encoded)
                align_score = np.sum(kl_vals)
                align_scores_fixed_length_pert.append(align_score)

            alignment_hist_data = [(np.array(align_scores_sis),
                                        25, 'SIS'),
                                   (np.array(align_scores_ig),
                                        25, 'IG'),
                                   (np.array(align_scores_top_ig),
                                        25, 'Top IG')]
            visualizations.plot_hist(alignment_hist_data,
                title='Scoring Rationales against Known Motifs',
                xlabel='sum(KL divergence) over positions of after optimal known motif alignment',
                ylabel='Normalized Frequency',
                normed=True,
                legend_loc='upper right',
                savepath=os.path.join(OUT_DIR, 'motif_alignment_hist.png'))

            # Add alignment scores summary statistics to stats dict
            align_scores_sis_stats = make_stats_dict(align_scores_sis)
            align_scores_ig_stats = make_stats_dict(align_scores_ig)
            align_scores_top_ig_stats = make_stats_dict(align_scores_top_ig)
            align_scores_fixed_length_ig_stats = make_stats_dict(align_scores_fixed_length_ig)
            align_scores_fixed_length_lime_stats = make_stats_dict(align_scores_fixed_length_lime)
            align_scores_fixed_length_pert_stats = make_stats_dict(align_scores_fixed_length_pert)

            rs_stat_alignment_sis_vs_ig, rs_p_alignment_sis_vs_ig = ranksums(
                align_scores_sis,
                align_scores_ig)
            rs_stat_alignment_sis_vs_top_ig, rs_p_alignment_sis_vs_top_ig = ranksums(
                align_scores_sis,
                align_scores_top_ig)

            if 'alignment' not in stats:
                stats['alignment'] = {}
            stats['alignment']['sis'] = align_scores_sis_stats
            stats['alignment']['ig'] = align_scores_ig_stats
            stats['alignment']['fixed_length_ig'] = align_scores_fixed_length_ig_stats
            stats['alignment']['fixed_length_lime'] = align_scores_fixed_length_lime_stats
            stats['alignment']['fixed_length_pert'] = align_scores_fixed_length_pert_stats
            stats['alignment']['top_ig'] = align_scores_top_ig_stats
            stats['alignment']['sis_vs_ig'] = {}
            stats['alignment']['sis_vs_ig']['ranksum_statistic'] = float(rs_stat_alignment_sis_vs_ig)
            stats['alignment']['sis_vs_ig']['ranksum_pvalue'] = float(rs_p_alignment_sis_vs_ig)
            stats['alignment']['sis_vs_top_ig'] = {}
            stats['alignment']['sis_vs_top_ig']['ranksum_statistic'] = float(rs_stat_alignment_sis_vs_top_ig)
            stats['alignment']['sis_vs_top_ig']['ranksum_pvalue'] = float(rs_p_alignment_sis_vs_top_ig)

            if VERBOSE:
                print('Done with motif/rationale alignment.\n')

        # Add idxs for "interesting" test examples to stats dict
        stats['idxs'] = container.get_idxs()

        # Save stats dictionary to json
        save_stats(stats, stats_path)
        if VERBOSE:
            print('Dumped stats to json file.\n')


if __name__ == '__main__':
    main()
