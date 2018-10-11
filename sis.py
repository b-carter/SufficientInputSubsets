import numpy as np
import os
import gzip
import json
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, Reshape, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing import sequence, text
from keras import backend as K

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


#####################################
#        TENSORFLOW/KERAS CONFIG
#####################################
def tf_config(cuda_visible_devices='1'):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # ensures specific ordering of GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)


#####################################
#        PREPROCESSING
#####################################
def load_reviews(path, verbose=True):
    data_x, data_y = [ ], [ ]
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            line = line.decode('ascii')
            y, sep, x = line.partition("\t")
            # x = x.split()
            y = y.split()
            if len(x) == 0: continue
            y = np.asarray([ float(v) for v in y ])
            data_x.append(x)
            data_y.append(y)

    if verbose:
        print("{} examples loaded from {}".format(len(data_x), path))
        print("max text length: {}".format(max(len(x) for x in data_x)))

    return data_x, data_y

def create_splits(X, y, test_size=3000, random_state=42):
    return train_test_split(X,
                            y,
                            test_size=3000,
                            random_state=42)

def create_tokenizer(X, top_words=10000):
    tokenizer = text.Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(X)
    return tokenizer

def pad_sequences(X, max_words=500):
    return sequence.pad_sequences(X, maxlen=max_words)


#####################################
#        MODELS
#####################################
# Create a model that can directly accept embeddings as input (rather than raw sequence)
# No dropout layers since we don't explicitly train this model
def make_lstm_model_feed_embeddings(max_words=500, embed_dim=100, lstm_dim=200):
    model = Sequential()
    model.add(LSTM(lstm_dim, input_shape=(max_words, embed_dim), return_sequences=True))
    model.add(LSTM(lstm_dim))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=['mse', 'mae'])
    return model

def copy_layer_weights(from_model, to_model):
    # If using dropout, need to modify the from_model layer indices around dropout layers (to 2, 4, 6)
    to_model.layers[0].set_weights(from_model.layers[1].get_weights())  # LSTM 1
    to_model.layers[1].set_weights(from_model.layers[2].get_weights())  # LSTM 2
    to_model.layers[2].set_weights(from_model.layers[3].get_weights())  # Dense output

def make_text_cnn_model_feed_embeddings(max_words=500, embed_dim=100, num_filters=128, filter_size=3):
    model = Sequential()
    model.add(Reshape((500, 100, 1), input_shape=(max_words, embed_dim)))
    model.add(Conv2D(num_filters, kernel_size=(filter_size, 100), padding='valid', activation='relu'))
    model.add(MaxPool2D(pool_size=(500 - filter_size + 1, 1), padding='valid'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse', 'mae', coeff_determination_metric])
    return model

def copy_layer_weights_text_cnn(from_model, to_model):
    to_model.layers[1].set_weights(from_model.layers[2].get_weights())  # Conv2D
    to_model.layers[-1].set_weights(from_model.layers[-1].get_weights())  # Dense output

def get_embeddings(model):
    return model.layers[0].get_weights()[0]

def coeff_determination_metric(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res / (SS_tot + K.epsilon()))

def predicted_probs(model, X_test):
    y_pred = model.predict_proba(X_test, verbose=0)
    y_pred = y_pred[:,0]  # flatten to 1-D array
    return y_pred

def eval_bin_class_model(model, X_test, y_test, verbose=True):
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracy = scores[1]
    y_pred = predicted_probs(model, X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    if verbose:
        print('Accuracy: %.2f%%' % (accuracy*100.0))
        print('ROC AUC: %.3f' % roc_auc)
    return (accuracy, roc_auc)


#####################################
#        SIS
#####################################
def predict_for_embed_sequence(batch, model, batch_size=128):
    batch_reshaped = [np.array(seq) for seq in batch]
    pred = model.predict(np.array(batch_reshaped), batch_size=batch_size)
    pred = pred.reshape(-1)  # flatten
    return pred

def predict_for_images(batch, model, batch_size=128):
    batch_reshaped = [np.array(x) for x in batch]
    pred = model.predict(np.array(batch_reshaped), batch_size=batch_size)
    return pred

def probas_to_label(probas):
    return np.argmax(probas, axis=-1)

def pred_class_and_prob(probas):
    pred_class = probas_to_label(probas)
    prob = probas[pred_class]
    return pred_class, prob

# `embeddings` is the vocabulary embeddings matrix
# `replace_with` determines what to put in place of the removed word,
#    can be either 'zeros' (zero-vector) or 'mean' (mean of latent embeddings)
def replacement(embeddings, replace_with='zeros'):
    if replace_with == 'zeros':
        replacement_embedding = np.zeros(embeddings.shape[1])
    elif replace_with == 'mean':
        replacement_embedding = np.mean(embeddings, axis=0)
    else:
        raise NotImplementedError()
    return replacement_embedding

# `sequence` is input sequence to modify, as padded sequence of embeddings,
#    e.g. embeddings[X_train[i]] (pad characters are ignored)
# `model` is a predictive model that accepts embedded as input
# `num_pad` is TODO
# `replacement_embedding` TODO
# Returns: list of predictions, where each value is the prediction if that element
#    is removed (ignores padding characters)
def removed_word_predictions(embedded_sequence, model, num_pad,
                                replacement_embedding):
    batch = []
    for i in range(num_pad, embedded_sequence.shape[0]):
        modified_sequence = embedded_sequence.copy()
        modified_sequence[i] = replacement_embedding
        batch.append(modified_sequence)
    removed_scores = predict_for_embed_sequence(batch, model)
    return removed_scores

# sets the i-th row of `seq` to vec
# assumes each row of seq is an encoded-vector
def replace_at_tf(seq, vec, i):
    seq[i, :] = vec

def removed_word_predictions_tf(seq, model, replacement_embedding):
    batch = []
    for i in range(seq.shape[0]):
        modified_seq = seq.copy()
        replace_at_tf(modified_seq, replacement_embedding, i)
        batch.append(modified_seq)
    removed_scores = predict_for_embed_sequence(batch, model)
    return removed_scores

def sis_removal(sequence, model, embeddings,
                    replace_with='mean',
                    return_history=False,
                    verbose=True,
                    index_to_token=None,
                    embedded_input=None,
                    replacement_embedding=None):
    if embedded_input is not None:  # assumes input already consists of embeddings
        current_seq = np.copy(embedded_input)
    else:
        current_seq = embeddings[sequence]
    starting_score = predict_for_embed_sequence([current_seq], model)[0]
    if return_history:
        history = []  # [starting_score]
    if verbose:
        print('Starting at: ', starting_score)
    num_pad = np.count_nonzero(sequence == 0)
    if starting_score >= 0.5:
        condition = lambda score: score >= 0 #0.5
        get_best_idx = np.nanargmax
    else:
        condition = lambda score: score < 1 #0.5
        get_best_idx = np.nanargmin
    current_score = starting_score
    if replacement_embedding is None:
        replacement_embedding = replacement(embeddings, replace_with=replace_with)
    num_removed = 0
    removed_elts = []
    removed_elts_bool = np.zeros(np.count_nonzero(sequence), dtype=bool)
    while condition(current_score) and not np.all(removed_elts_bool):
        removed_scores = removed_word_predictions(current_seq, model, num_pad,
                                                    replacement_embedding)
        # put nans in positions where element already removed
        removed_scores[removed_elts_bool] = np.nan
        best_to_remove_idx = get_best_idx(removed_scores)
        current_score = removed_scores[best_to_remove_idx]
        if return_history:
            history.append(current_score)
        if verbose:
            if index_to_token is None:
                print('Error: Need `index_to_token` to print verbose update.')
            else:
                print(best_to_remove_idx,
                      index_to_token[sequence[num_pad+best_to_remove_idx]],
                      current_score)
        if condition(current_score):  # actually do the removal
            current_seq[num_pad+best_to_remove_idx] = replacement_embedding
            removed_elts.append(best_to_remove_idx)
            removed_elts_bool[best_to_remove_idx] = True
            num_removed += 1
    if return_history:
        return removed_elts, history
    return removed_elts

def sis_removal_tf(start_seq, model, replacement_embedding, return_history=True, verbose=True):
    current_seq = np.array(start_seq.copy(), dtype='float32')
    starting_score = predict_for_embed_sequence([current_seq], model)[0]
    if return_history:
        history = []
    if verbose:
        print('Starting at: ', starting_score)
    get_best_idx = np.nanargmax  # always trying to maximize prediction
    current_score = starting_score
    num_removed = 0
    removed_elts = []
    removed_elts_bool = np.zeros(current_seq.shape[0], dtype=bool)
    while not np.all(removed_elts_bool):
        removed_scores = removed_word_predictions_tf(current_seq, model, replacement_embedding)
        # put nans in positions where element already removed
        removed_scores[removed_elts_bool] = np.nan
        best_to_remove_idx = get_best_idx(removed_scores)
        current_score = removed_scores[best_to_remove_idx]
        history.append(current_score)
        if verbose:
            print(best_to_remove_idx, current_score)
        replace_at_tf(current_seq, replacement_embedding, best_to_remove_idx)
        removed_elts.append(best_to_remove_idx)
        removed_elts_bool[best_to_remove_idx] = True
        num_removed += 1
    if return_history:
        return removed_elts, history
    return removed_elts

def replace_at_img(x, replacement, pos):
    x[pos] = replacement

def removed_word_predictions_img(x, pos_to_remove, model, replacement):
    batch = []
    for pos in pos_to_remove:
            modified_x = x.copy()
            replace_at_img(modified_x, replacement, pos)
            batch.append(modified_x)
    removed_preds = predict_for_images(batch, model)
    return removed_preds

def sis_removal_img_classif(start_x, image, class_idx, model,
                                replacement, return_history=True,
                                verbose=True):
    current_x = start_x.copy()
    current_preds = predict_for_images([current_x], model)[0]
    current_prob = current_preds[class_idx]
    if return_history:
        history = []
    if verbose:
        print('Predicting class %d with prob %.5f: ' % \
                (class_idx, current_prob))
    num_removed = 0
    removed_elts = []
    removed_elts_bool = np.zeros(image.get_num_pixels(), dtype=bool)
    while not np.all(removed_elts_bool):
        is_to_remove = np.where(np.logical_not(removed_elts_bool))[0]
        pos_to_remove = [image.i_to_pos(i) for i in is_to_remove]
        removed_preds = removed_word_predictions_img(current_x, pos_to_remove,
            model, replacement)
        removed_preds_for_class = removed_preds[:, class_idx]
        best_to_remove_idx = np.argmax(removed_preds_for_class)
        best_to_remove_i = is_to_remove[best_to_remove_idx]
        best_to_remove_pos = pos_to_remove[best_to_remove_idx]
        current_prob = removed_preds_for_class[best_to_remove_idx]
        history.append(current_prob)
        # actually do the removal
        replace_at_img(current_x, replacement, best_to_remove_pos)
        removed_elts.append(best_to_remove_i)
        removed_elts_bool[best_to_remove_i] = True
        num_removed += 1
        if verbose:
            print('Removed at pos: %s, prediction: %.5f, num removed: %d' % \
                    (str(best_to_remove_pos), current_prob, num_removed))
    if return_history:
        return removed_elts, history
    return removed_elts


#####################################
#        EVALUATION
#####################################
def load_rationale_annotations(path, verbose=True):
    data = []
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
    if verbose:
        print('Loaded %d annotations.' % len(data))
    return data

# Retokenize the annotation using our trained tokenizer
# `annotation` is the list of tokens from the given annotation (e.g. annotations[i]['x'])
# `tokenizer` is our trained tokenizer (Keras object)
# Returns: tokenized example
# `annotation_indices` is the list of token indices for annotated sentences; if not None,
#     also returns a list of same length where each 
def retokenize_annotation(annotation, tokenizer, annotation_indices=None, count_missing=False):
    joined_annotation = ' '.join(annotation)
    retokenized = tokenizer.texts_to_sequences([joined_annotation])[0]
    if count_missing:
        word_sequence = text.text_to_word_sequence(joined_annotation,
                                                   tokenizer.filters,
                                                   tokenizer.lower,
                                                   tokenizer.split)
        num_missing = len(word_sequence) - len(retokenized)
    else:
        num_missing = None
    return retokenized, num_missing

def find_sub_list(sl, l):
    res = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            res.append((ind, ind + sll -1))
    return res

def get_annot_idxs(i, annotations, tokenizer, aspect):
    tokenized_ex, _ = retokenize_annotation(annotations[i]['x'], tokenizer, count_missing=False)
    annot_idxs_orig = annotations[i][str(aspect)]
    annot_idxs = []
    for a, b in annot_idxs_orig:
        tokenized_annot, _ = retokenize_annotation(annotations[i]['x'][a:b], tokenizer, count_missing=False)
        if len(tokenized_annot) == 0:
            continue
        idxs = find_sub_list(tokenized_annot, list(tokenized_ex))
        # make sure the annotation sequence doesn't appear twice
        if len(idxs) > 1:
            # this happens rarely, maybe once per dataset
            # need to look outside the retokenized annotation to resolve the ambiguity
            # in theory if subsequence was all the way to the right of the sequence,
            # this code would break, but it should be fine in practice
            offset = 1
            while len(idxs) > 1:
                tokenized_annot, _ = retokenize_annotation(annotations[i]['x'][a:b+offset],
                                                            tokenizer,
                                                            count_missing=False)
                idxs = find_sub_list(tokenized_annot, list(tokenized_ex))
            idxs = [(idxs[0][0], idxs[0][1] - offset)]
        assert(len(idxs) == 1)
        annot_idxs.append(idxs[0])
    return annot_idxs

# Starting from the end of `history`, going backwards, determine how many words needed
#   in order to satisfy `threshold_f`.
#   threshold_f(x) returns True if x satisfies the constraint, otherwise returns False
# If `ignore_last` is True, ignores the last element of history (since this usually corresponds to score after
#   all sequence elements have been removed)
def find_min_words_needed(history, threshold_f, ignore_last=True):
    num_needed = 0
    vals = history[::-1]
    if ignore_last:
        vals = vals[1:]
    found = False
    for val in vals:
        num_needed += 1
        if threshold_f(val):
            found = True
            break
    if ignore_last and not found:  # need to use full input
        num_needed += 1
    return num_needed

# Computes min needed to satisfy threshold function for examples in X given by indices in example_idxs,
#   assumes threshold_f is the same for all examples
# `remove_annots` is a list (indexed into by idxs) of tuples (sorted_word_order, history) - history used here
# Returns: (min_needed, min_needed_percentage), each of which has same length as example_idxs
#             and percentage list is the percentage of the total number of words in that example
def min_needed_for_idxs(example_idxs, remove_annots, threshold_f):
    min_needed = []
    min_needed_percentage = []
    for i in example_idxs:
        history = remove_annots[i][1]
        min_needed_i = find_min_words_needed(history, threshold_f)
        min_needed.append(min_needed_i)
        percentage = min_needed_i / float(len(history)) * 100.0
        min_needed_percentage.append(percentage)
    return (min_needed, min_needed_percentage)

def find_score_history_given_order(x, sorted_word_importances, num_pad, model,
                                    mean_embedding, pad_embedding, embeddings):
    seq = np.vstack([np.repeat(pad_embedding.reshape((1, pad_embedding.shape[0])), num_pad, axis=0),
                     np.repeat(mean_embedding.reshape((1, mean_embedding.shape[0])), len(sorted_word_importances),
                                axis=0)])
    batch_seqs = [np.copy(seq)]
    # reverse order to add back most important words first
    for idx in sorted_word_importances[::-1]:
        word = x[num_pad + idx]
        word_embedding = embeddings[word]
        seq[num_pad + idx] = word_embedding
        batch_seqs.append(np.copy(seq))
    history = predict_for_embed_sequence(batch_seqs, model)
    # reverse order and remove first element, so first element of history list is score
    #    of sequence after first word removed
    history = history[::-1][1:]
    return history

# `seq` is a (4 x L) sequence (the final sequence in one-hot representation)
# `empty_input` is a (4 x L) empty input sequence (probably initially all
#    0's or 0.25's)
def find_score_history_given_order_tf(final_seq, sorted_word_importances,
                                        model, empty_input):
    current_seq = empty_input.copy()
    batch_seqs = [np.copy(current_seq)]
    # reverse order to add back most important words first
    for idx in sorted_word_importances[::-1]:
        current_seq[idx, :] = final_seq[idx, :]
        batch_seqs.append(np.copy(current_seq))
    history = predict_for_embed_sequence(batch_seqs, model)
    # reverse order and remove first element, so first element of history list
    #   is score of sequence after first word removed
    history = history[::-1][1:]
    return history

# `indices_min_needed` comes from `zip(example_idxs, min_needed)` so it
#    contains tuples of (i, k) pairs where i is the index of the example
#    in the annotaiton set and k is the number of elems in the rationale
# For each element in the sequence, replace with `replacement_embedding`
#   and predict using `model` (which accepts embeddings directly as input)
# Predictions are then made where each element has been individually removed.
# Applies `diffs_transform_f` to the returned list as numpy array (e.g. can do
#   `lambda preds, orig: orig - x` for differences from original prediction)
# Returns tuple of (rationale_diffs, nonrationale_diffs) where rationale_diffs
#   contains the score delta in the elements that are included in rationale,
#   and nonrationale diffs contains the same but for non-rationale elements.
def perturbation_removal_rationale(indices_min_needed, X_annotation, model,
                          embeddings, replacement_embedding, remove_annots,
                          original_predictions,
                          diffs_transform_f=lambda preds_orig: preds_orig[1] - preds_orig[0]):
    rationale_diffs = []
    nonrationale_diffs = []
    for i, k in indices_min_needed:
        x = X_annotation[i]
        num_pad = np.count_nonzero(x == 0)
        x_embed = embeddings[x]
        bottom_k = remove_annots[i][0][-k:]
        preds = removed_word_predictions(x_embed, model, num_pad,
                                          replacement_embedding)
        original_pred = float(original_predictions[i])
        diffs = diffs_transform_f((np.array(preds), original_pred))
        bottom_k_diffs = np.take(diffs, bottom_k)
        other_diffs = np.delete(diffs, bottom_k)
        assert(diffs.shape[0] == bottom_k_diffs.shape[0] + other_diffs.shape[0])
        rationale_diffs.append(bottom_k_diffs)
        nonrationale_diffs.append(other_diffs)
    return (rationale_diffs, nonrationale_diffs)

def perturbation_removal_rationale_tf(example_idxs, rationale_lens, X, model, replacement_embedding,
                                      remove_results, original_predictions,
                                      diffs_transform_f=lambda preds_orig: preds_orig[1] - preds_orig[0]):
    assert(len(example_idxs) == len(rationale_lens))
    rationale_diffs = []
    nonrationale_diffs = []
    for j, i in enumerate(example_idxs):
        k = rationale_lens[j]
        x = X[i]
        bottom_k = remove_results[j][0][-k:]
        preds = removed_word_predictions_tf(x, model, replacement_embedding)
        original_pred = float(original_predictions[i])
        diffs = diffs_transform_f((np.array(preds), original_pred))
        bottom_k_diffs = np.take(diffs, bottom_k)
        other_diffs = np.delete(diffs, bottom_k)
        assert(diffs.shape[0] == bottom_k_diffs.shape[0] + other_diffs.shape[0])
        rationale_diffs.append(bottom_k_diffs)
        nonrationale_diffs.append(other_diffs)
    return (rationale_diffs, nonrationale_diffs)
