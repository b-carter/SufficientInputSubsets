import numpy as np
import os
import json
import types

import sis
import lime_helper
from numpy_encoder import NumpyJSONEncoder
from packages.IntegratedGradients.IntegratedGradients import integrated_gradients


##########################################
##  Rationale keys for various methods (SIS and alternative methods) for dumps
##  of Example objects:

SIS_RATIONALE_KEY = 'sis'

IG_SUFF_RATIONALE_KEY = 'ig_sufficient'
IG_FIXED_RATIONALE_KEY = 'ig_fixed_length'
IG_TOP_RATIONALE_KEY = 'ig_top'

LIME_SUFF_RATIONALE_KEY = 'lime_sufficient'
LIME_FIXED_RATIONALE_KEY = 'lime_fixed_length'

PERTURB_SUFF_RATIONALE_KEY = 'perturb_sufficient'
PERTURB_FIXED_RATIONALE_KEY = 'perturb_fixed_length'

##########################################


def make_threshold_f(threshold, is_pos):
    if not isinstance(is_pos, bool):
        raise TypeError('`is_pos` must be a boolean type')
    if is_pos:
        return lambda x: x >= threshold
    else:
        return lambda x: x <= threshold


class Rationale(object):
    def __init__(self, elms=[], history=None):
        self.elms = elms
        self.history = history

    def add(self, e):
        self.elms.append(e)

    def get_elms(self):
        return self.elms

    def get_length(self):
        return len(self.elms)

    def get_history(self):
        return self.history

    def __len__(self):
        return self.get_length()

    def __iter__(self):
        return iter(self.elms)

    def to_json_str(self):
        json_str = json.dumps(self.__dict__, cls=NumpyJSONEncoder)
        return json_str

    @staticmethod
    def from_json_str(json_str):
        data = json.loads(json_str)
        rationale = Rationale()
        for k, v in data.items():
            setattr(rationale, k, v)
        return rationale


class ExampleJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Rationale):
            return obj.to_json_str()
        elif isinstance(obj, types.FunctionType):  # cannot serialize functions
            return None
        else:
            return super(ExampleJSONEncoder, self).default(obj)


# Define class for maintaining example (with input and index in dataset)
class Example(object):
    x = None  # input
    i = None  # index in dataset

    def __init__(self, x=None, i=None):
        self.x = x
        self.i = i


def compute_mean_embedding(embeddings):
    if embeddings is None:
        raise TypeError('`embeddings` cannot be None')
    return np.mean(embeddings, axis=0)


class BeerReview(Example):
    def __init__(self, x=None, i=None, embeddings=None, pad_char=0):
        super(BeerReview, self).__init__(x=x, i=i)
        self.embeddings = embeddings
        self.annot_idxs = []
        self.rationales = {}
        self.pad_char = pad_char
        self.num_pad = np.count_nonzero(x == pad_char)
        self.original_prediction = None
        self.prediction_rationale_only = None
        self.prediction_nonrationale_only = None
        self.prediction_annotation_only = None
        self.prediction_nonannotation_only = None
        self.threshold = None  # "interesting" threshold
        self.is_pos = None
        self.threshold_f = None

    def get_pad_embedding(self):
        return self.embeddings[self.pad_char]

    def set_annotation_idxs(self, annot_idxs):
        self.annot_idxs = annot_idxs

    def get_annotation_idxs(self):
        return self.annot_idxs

    def has_annotation(self):
        return len(self.get_annotation_idxs()) > 0

    def get_embeddings(self):
        return embeddings

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def set_threshold_f(self, threshold_f):
        self.threshold_f = threshold_f

    def get_rationales(self, method):
        if method not in self.rationales:
            return []
        return self.rationales[method]

    def get_num_tokens(self):
        return self.x.shape[0] - self.num_pad

    def add_rationale(self, rationale, method):
        if method not in self.rationales:
            self.rationales[method] = []
        self.rationales[method].append(rationale)

    def get_replacement_embedding(self, replacement_embedding='mean'):
        if isinstance(replacement_embedding, str) and \
                replacement_embedding == 'mean':
            replacement_embedding = compute_mean_embedding(self.embeddings)
        return replacement_embedding

    def get_embedded_sequence(self, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        return np.copy(embeddings[self.x])

    def get_embedded_sequence_annotations_only(self, replacement_embedding='mean'):
        x_embed = self.get_embedded_sequence()
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        modified_seq = np.repeat(replacement_embedding.reshape(
                                   (1, replacement_embedding.shape[0])),
                                 x_embed.shape[0], axis=0)
        for x, y in self.get_annotation_idxs():
            x = x + self.num_pad
            y = y + self.num_pad
            modified_seq[x:y+1,:] = x_embed[x:y+1,:]
        return modified_seq

    # TODO: fix `return_none_no_annot` param to be consistent with previous function
    def get_embedded_sequence_nonannotations_only(self, replacement_embedding='mean',
                                                   return_none_no_annot=False):
        annot_idxs = self.get_annotation_idxs()
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        if len(annot_idxs) == 0 and return_none_no_annot:
            return None
        modified_seq = self.get_embedded_sequence()
        for x, y in annot_idxs:
            for j in range(x, y + 1):
                modified_seq[j + self.num_pad] = replacement_embedding
        return modified_seq

    def get_all_rationale_idxs(self, rationales):
        rationale_idxs = []
        for rationale in rationales:
            rationale_idxs += list(rationale)
        return np.asarray(rationale_idxs)

    def get_nonrationale_idxs(self, rationales):
        num_tokens = self.get_num_tokens()
        rationale_idxs = self.get_all_rationale_idxs(rationales)
        non_rationale_idxs = np.delete(np.arange(num_tokens), rationale_idxs)
        return non_rationale_idxs

    def get_embedded_sequence_rationale_only(self, rationales,
                                              replacement_embedding='mean',
                                              embeddings=None):
        modified_seq = self.get_embedded_sequence(embeddings=embeddings)
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        non_rationale_idxs = self.get_nonrationale_idxs(rationales)
        non_rationale_idxs_with_offset = self.num_pad + non_rationale_idxs
        modified_seq[non_rationale_idxs_with_offset] = replacement_embedding
        return modified_seq

    def get_embedded_sequence_nonrationale_only(self, rationales,
                                                 replacement_embedding='mean'):
        modified_seq = self.get_embedded_sequence()
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        rationale_idxs = self.get_all_rationale_idxs(rationales)
        rationale_idxs_with_offset = rationale_idxs + self.num_pad
        if len(rationale_idxs_with_offset) > 0:
            modified_seq[rationale_idxs_with_offset] = replacement_embedding
        return modified_seq

    def frac_rationale_in_annotation(self, rationales):
        annot_idxs = self.get_annotation_idxs()
        count_in_annots = 0
        rationale_length = 0
        for rationale in rationales:
            for el in rationale:
                rationale_length += 1
                for x, y in annot_idxs:
                    if el >= x and el <= y:
                        count_in_annots += 1
                        break
        frac = float(count_in_annots) / rationale_length
        return frac

    def set_predictions(self, model, rationales, replacement_embedding='mean'):
        self.original_prediction = sis.predict_for_embed_sequence(
            [self.get_embedded_sequence()], model)[0]
        self.prediction_rationale_only = sis.predict_for_embed_sequence(
            [self.get_embedded_sequence_rationale_only(rationales,
                replacement_embedding=replacement_embedding)],
            model)[0]
        self.prediction_nonrationale_only = sis.predict_for_embed_sequence(
            [self.get_embedded_sequence_nonrationale_only(rationales,
                replacement_embedding=replacement_embedding)],
            model)[0]
        self.prediction_annotation_only = sis.predict_for_embed_sequence(
            [self.get_embedded_sequence_annotations_only(
                replacement_embedding=replacement_embedding)], model)[0]
        self.prediction_nonannotation_only = sis.predict_for_embed_sequence(
            [self.get_embedded_sequence_nonannotations_only(
                replacement_embedding=replacement_embedding)], model)[0]

    def run_sis_rationales(self, model, replacement_embedding='mean',
                                       first_only=True, verbose=False):
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        all_rationales = self.get_rationales(SIS_RATIONALE_KEY)
        x_nonrationale = self.get_embedded_sequence_nonrationale_only(
            all_rationales,
            replacement_embedding=replacement_embedding)
        current_nonrationale_pred = sis.predict_for_embed_sequence(
            [x_nonrationale], model)[0]
        if first_only and len(all_rationales) >= 1:
            if verbose:
                print('Already >= 1 rationale and first_only=True, returning.')
            return None
        if verbose:
            print('Starting prediction %.3f' % current_nonrationale_pred)
        while self.threshold_f(current_nonrationale_pred):
            # prediction on non-rationale is still beyond threshold
            if verbose:
                print('Prediction beyond threshold, extracting rationale')
            removed_elts, history = sis.sis_removal(
                self.x,
                model,
                self.embeddings,
                embedded_input=x_nonrationale,
                replacement_embedding=replacement_embedding,
                return_history=True,
                verbose=False)
            rationale_length = sis.find_min_words_needed(history,
                self.threshold_f)
            rationale_elems = removed_elts[-rationale_length:]
            rationale = Rationale(elms=rationale_elems[::-1],
                                  history=(removed_elts, history))
            self.add_rationale(rationale, SIS_RATIONALE_KEY)
            # mask new rationale in the sequence and re-predict
            all_rationales = self.get_rationales(SIS_RATIONALE_KEY)
            x_nonrationale = self.get_embedded_sequence_nonrationale_only(
                all_rationales,
                replacement_embedding=replacement_embedding)
            current_nonrationale_pred = sis.predict_for_embed_sequence(
                [x_nonrationale], model)[0]
            if verbose:
                print('New predicted score %.3f' % current_nonrationale_pred)
            if first_only:
                if verbose:
                    print('Only 1 rationale, first_only=True, breaking.')
                break
        if verbose:
            print('Done building rationales.')

    def run_perturbative_baseline_rationale(self, embed_model,
                                            replacement_embedding='mean',
                                            verbose=False):
        if len(self.get_rationales(PERTURB_SUFF_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have perturbative baseline rationale,',
                        'returning.')
            return None
        if verbose:
            print('Running perturbative baseline rationale.')
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        x_embed = self.get_embedded_sequence()
        removed_scores = sis.removed_word_predictions(x_embed,
                                                         embed_model,
                                                         self.num_pad,
                                                         replacement_embedding)
        sorted_words = removed_scores.argsort()
        if self.is_pos:
            # word with biggest drop in score (lowest final score) at end of
            #   sorted list
            sorted_words = sorted_words[::-1]
        score_history = sis.find_score_history_given_order(
            self.x,
            sorted_words,
            self.num_pad,
            embed_model,
            replacement_embedding,
            self.get_pad_embedding(),
            self.embeddings)
        rationale_length = sis.find_min_words_needed(score_history,
            self.threshold_f)
        rationale_elems = sorted_words[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(sorted_words, score_history))
        self.add_rationale(rationale, PERTURB_SUFF_RATIONALE_KEY)
        if verbose:
            print('Done with perturbative baseline.')

    def run_integrated_gradients_rationale(self, ig_model, embed_model,
                                            baseline,
                                            replacement_embedding='mean',
                                            verbose=False):
        if len(self.get_rationales(IG_SUFF_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have IG rationale, returning.')
            return None
        if verbose:
            print('Running integrated gradients rationale.')
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        x_embed = self.get_embedded_sequence()
        igs = ig_model.explain(x_embed, reference=baseline)
        igs = np.linalg.norm(igs, ord=1, axis=1)  # L1 norm along embeddings
        sorted_words = igs[self.num_pad:].argsort()
        score_history = sis.find_score_history_given_order(
            self.x,
            sorted_words,
            self.num_pad,
            embed_model,
            replacement_embedding,
            self.get_pad_embedding(),
            self.embeddings)
        rationale_length = sis.find_min_words_needed(score_history,
            self.threshold_f)
        rationale_elems = sorted_words[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(sorted_words, score_history))
        self.add_rationale(rationale, IG_SUFF_RATIONALE_KEY)
        if verbose:
            print('Done with integrated gradients.')

    def run_lime_rationale(self, text_pipeline, embed_model, index_to_token,
                            replacement_embedding='mean', verbose=False):
        if len(self.get_rationales(LIME_SUFF_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have LIME rationale, returning.')
            return None
        if verbose:
            print('Running LIME rationale.')
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        text = self.to_text(index_to_token, str_joiner=' ')
        explainer = lime_helper.make_explainer(verbose=False)
        explanation = lime_helper.explain(text, explainer, text_pipeline)
        sorted_words = lime_helper.extract_word_order(explanation)
        score_history = sis.find_score_history_given_order(
            self.x,
            sorted_words,
            self.num_pad,
            embed_model,
            replacement_embedding,
            self.get_pad_embedding(),
            self.embeddings)
        rationale_length = sis.find_min_words_needed(score_history,
            self.threshold_f)
        rationale_elems = sorted_words[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(sorted_words, score_history))
        self.add_rationale(rationale, LIME_SUFF_RATIONALE_KEY)
        if verbose:
            print('Done with LIME.')

    def run_integrated_gradients_fixed_length_rationale(self, ig_model,
                                                         embed_model,
                                                         baseline,
                                                         verbose=False):
        if len(self.get_rationales(IG_FIXED_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have fixed length IG rationale, returning.')
            return None
        if verbose:
            print('Running fixed length IG rationale.')
        x_embed = self.get_embedded_sequence()
        igs = ig_model.explain(x_embed, reference=baseline)
        igs = np.linalg.norm(igs, ord=1, axis=1)  # L1 norm along embeddings
        sorted_words = igs[self.num_pad:].argsort()
        rationale_length = self.get_fixed_baseline_length()
        rationale_elems = sorted_words[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(sorted_words, None))
        self.add_rationale(rationale, IG_FIXED_RATIONALE_KEY)
        if verbose:
            print('Done with fixed length IG.')

    # LIME baseline where length is fixed to same as median SIS length
    def run_lime_fixed_length_rationale(self, text_pipeline, embed_model,
                                         index_to_token, verbose=False):
        if len(self.get_rationales(LIME_FIXED_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have fixed length LIME rationale, returning.')
            return None
        if verbose:
            print('Running fixed length LIME rationale.')
        text = self.to_text(index_to_token, str_joiner=' ')
        explainer = lime_helper.make_explainer(verbose=False)
        explanation = lime_helper.explain(text, explainer, text_pipeline)
        sorted_words = lime_helper.extract_word_order(explanation)
        rationale_length = self.get_fixed_baseline_length()
        rationale_elems = sorted_words[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(sorted_words, None))
        self.add_rationale(rationale, LIME_FIXED_RATIONALE_KEY)
        if verbose:
            print('Done with LIME.')

    # Perturbative baseline where length is fixed to same as median SIS length
    def run_perturb_fixed_length_rationale(self, embed_model,
                                            replacement_embedding='mean',
                                            verbose=False):
        if len(self.get_rationales(PERTURB_FIXED_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have fixed length perturbative baseline rationale,',
                        'returning.')
            return None
        if verbose:
            print('Running perturbative baseline rationale.')
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        x_embed = self.get_embedded_sequence()
        removed_scores = sis.removed_word_predictions(x_embed,
                                                         embed_model,
                                                         self.num_pad,
                                                         replacement_embedding)
        sorted_words = removed_scores.argsort()
        if self.is_pos:
            # word with biggest drop in score (lowest final score) at end of
            #   sorted list
            sorted_words = sorted_words[::-1]
        rationale_length = self.get_fixed_baseline_length()
        rationale_elems = sorted_words[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(sorted_words, None))
        self.add_rationale(rationale, PERTURB_FIXED_RATIONALE_KEY)
        if verbose:
            print('Done with fixed length perturbative baseline.')

    # Returns length to use for fixed-length IG, LIME, and perturbative baselines
    # If only single SIS, returns that length. If multiple, returns
    #   median SIS length rounded to nearest integer.
    def get_fixed_baseline_length(self):
        sis_rationales = self.get_rationales(SIS_RATIONALE_KEY)
        if len(sis_rationales) == 0:
            raise ValueError('Must first compute SIS rationales.')
        if len(sis_rationales) == 1:
            return len(sis_rationales[0])
        med = np.median([len(r) for r in sis_rationales])
        return int(np.rint(med))

    def perturbation(self, model, replacement_embedding='mean',
                     diffs_transform_f=lambda preds_orig: \
                         preds_orig[1] - preds_orig[0]):
        perturb_idxs_scores = []
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        x_embed = self.get_embedded_sequence()
        preds = sis.removed_word_predictions(x_embed, model, self.num_pad,
                                                replacement_embedding)
        original_pred = self.original_prediction if self.original_prediction \
                            is not None else sis.predict_for_embed_sequence(
                            [self.get_embedded_sequence()], model)[0]
        diffs = diffs_transform_f((np.array(preds), original_pred))
        return diffs

    def perturbation_rationale(self, model, rationales,
                                replacement_embedding='mean',
                                diffs_transform_f=lambda preds_orig: \
                                    preds_orig[1] - preds_orig[0]):
        diffs = self.perturbation(model,
            replacement_embedding=replacement_embedding,
            diffs_transform_f=diffs_transform_f)
        rationale_idxs = self.get_all_rationale_idxs(rationales)
        rationale_diffs = np.take(diffs, rationale_idxs)
        nonrationale_diffs = np.delete(diffs, rationale_idxs)
        assert(diffs.shape[0] == \
               rationale_diffs.shape[0] + nonrationale_diffs.shape[0])
        return rationale_diffs, nonrationale_diffs

    def to_text(self, index_to_token, str_joiner=None):
        non_pad_elems = self.x[self.num_pad:]
        text = [index_to_token[e] for e in non_pad_elems]
        if str_joiner is not None:
            text = str_joiner.join(text)
        return text

    def to_html(self, rationale):
        pass

    def to_json(self, f, include_embeddings=False):
        json_str = json.dumps(self.__dict__, cls=ExampleJSONEncoder)
        # re-create object to guarantee no modifications to __dict__
        json_dict = json.loads(json_str)
        if not include_embeddings:
            json_dict['embeddings'] = None
        json.dump(json_dict, f)

    @staticmethod
    def from_json(f, set_threshold_f=True):
        data = json.load(f)
        review = BeerReview()
        for k, v in data.items():
            if k == 'rationales':  # construct Rationale objects
                rationales = {}
                for k_ in v.keys():
                    for i in range(len(v[k_])):
                        v[k_][i] = Rationale.from_json_str(v[k_][i])
            elif (k == 'embeddings' or k == 'x') and v is not None:
                # cast to np array
                v = np.array(v)
            setattr(review, k, v)
        if set_threshold_f:
            try:
                threshold_f = make_threshold_f(review.threshold, review.is_pos)
                review.set_threshold_f(threshold_f)
            except TypeError:
                print('WARNING: cannot set `threshold_f`, `is_pos` is not True or False')
        return review

    def __len__(self):
        return self.get_num_tokens()


# Container class for storing BeerReview objects
class BeerReviewContainer(object):
    def __init__(self, embeddings, index_to_token, aspect, trained_model_path,
                  pad_char):
        self.pos_reviews = []
        self.neg_reviews = []
        self.i_to_review = {}
        self.embeddings = embeddings
        self.index_to_token = index_to_token
        self.aspect = aspect
        self.trained_model_path = trained_model_path
        self.pad_char = pad_char

    def add_pos_review(self, review):
        if review.i in self.i_to_review:
            raise KeyError('Review %d already in container' % (review.i))
        self.pos_reviews.append(review)
        self.i_to_review[review.i] = review

    def add_neg_review(self, review):
        if review.i in self.i_to_review:
            raise KeyError('Review %d already in container' % (review.i))
        self.neg_reviews.append(review)
        self.i_to_review[review.i] = review

    def get_review(self, i):
        if i not in self.i_to_review:
            raise KeyError('Review %d not in container' % (i))
        return self.i_to_review[i]

    def get_pos_reviews(self):
        return self.pos_reviews

    def get_neg_reviews(self):
        return self.neg_reviews

    def get_all_reviews(self):
        return self.pos_reviews + self.neg_reviews

    def get_index_to_token(self):
        return self.index_to_token()

    # Set `embeddings` attribute in all reviews in the container
    def set_embeddings_all(self):
        for review in self.get_all_reviews():
            review.set_embeddings(self.embeddings)

    def __len__(self):
        return len(self.i_to_review)

    @staticmethod
    def metadata_filename():
        return 'metadata.json'

    def dump_data(self, dir_path):
        # Make directories to dirpath path if not exists
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        metadata = {}

        metadata['trained_model_path'] = self.trained_model_path
        metadata['index_to_token'] = self.index_to_token
        metadata['pad_char'] = self.pad_char
        metadata['aspect'] = self.aspect

        # Dump embeddings to numpy npz file
        embeddings_filename = 'embeddings.txt'
        metadata['embeddings_file'] = embeddings_filename
        embeddings_filepath = os.path.join(dir_path, embeddings_filename)
        np.savetxt(embeddings_filepath, self.embeddings)

        # Dump pos and neg reviews to JSON files
        metadata['pos_reviews'] = []
        metadata['neg_reviews'] = []

        reviews_dir = 'reviews'
        reviews_dir_path = os.path.join(dir_path, reviews_dir)
        if not os.path.isdir(reviews_dir_path):
            os.makedirs(reviews_dir_path)

        for review in self.get_pos_reviews():
            review_file = os.path.join(reviews_dir, '%d.json' % (review.i))
            metadata['pos_reviews'].append(review_file)
            review_path = os.path.join(dir_path, review_file)
            with open(review_path, 'w') as f:
                review.to_json(f)

        for review in self.get_neg_reviews():
            review_file = os.path.join(reviews_dir, '%d.json' % (review.i))
            metadata['neg_reviews'].append(review_file)
            review_path = os.path.join(dir_path, review_file)
            with open(review_path, 'w') as f:
                review.to_json(f)

        metadata_file = os.path.join(dir_path, self.metadata_filename())
        with open(metadata_file, 'w') as outfile:
            json.dump(metadata, outfile)

    @staticmethod
    def load_data(dir_path):
        metadata_file = os.path.join(dir_path,
                                     BeerReviewContainer.metadata_filename())
        with open(metadata_file, 'r') as infile:
            metadata = json.load(infile)

        args = {}
        args['trained_model_path'] = metadata['trained_model_path']
        args['index_to_token'] = metadata['index_to_token']
        args['pad_char'] = metadata['pad_char']
        args['aspect'] = metadata['aspect']
        # Keys in index_to_token should be integers
        args['index_to_token'] = {int(k): v for k, v in \
            args['index_to_token'].items()}

        # Load embeddings
        embeddings_filename = metadata['embeddings_file']
        embeddings_filepath = os.path.join(dir_path, embeddings_filename)
        embeddings = np.loadtxt(embeddings_filepath)
        args['embeddings'] = embeddings

        container = BeerReviewContainer(**args)

        # Load review objects
        for review_file in metadata['pos_reviews']:
            review_path = os.path.join(dir_path, review_file)
            with open(review_path, 'r') as f:
                try:
                    review = BeerReview.from_json(f)
                except:
                    continue
            container.add_pos_review(review)

        for review_file in metadata['neg_reviews']:
            review_path = os.path.join(dir_path, review_file)
            with open(review_path, 'r') as f:
                try:
                    review = BeerReview.from_json(f)
                except:
                    continue
            container.add_neg_review(review)

        container.set_embeddings_all()

        return container


class DNASequence(Example):
    JSON_NUMPY_ATTRIBS = ['x', 'replacement']

    def __init__(self, x=None, i=None, replacement=None,
                    threshold=None, threshold_f=None):
        super(DNASequence, self).__init__(x=x, i=i)
        self.rationales = {}
        self.replacement = replacement
        self.original_prediction = None
        self.prediction_rationale_only = None
        self.prediction_nonrationale_only = None
        self.threshold = threshold
        self.threshold_f = threshold_f
        if threshold is not None and threshold_f is None:
            self.make_threshold_f()

    @staticmethod
    def replace_at(seq, vec, i):
        sis.replace_at_tf(seq, vec, i)

    def get_rationales(self, method):
        if method not in self.rationales:
            return []
        return self.rationales[method]

    def get_x(self, copy=True):
        if copy:
            return np.copy(self.x)
        return self.x

    def get_shape(self):
        shape = self.x.shape
        return shape

    def get_num_bases(self):
        return self.x.shape[0]

    def add_rationale(self, rationale, method):
        if method not in self.rationales:
            self.rationales[method] = []
        self.rationales[method].append(rationale)

    def set_replacement(self, replacement):
        self.replacement = replacement

    def get_replacement(self):
        if self.replacement is None:
            return TypeError('Must set replacement. Cannot be None.')
        return self.replacement

    def get_all_rationale_idxs(self, rationales):
        rationale_idxs = []
        for rationale in rationales:
            rationale_idxs += list(rationale)
        return np.asarray(rationale_idxs)

    def get_nonrationale_idxs(self, rationales):
        num_bases = self.get_num_bases()
        rationale_idxs = self.get_all_rationale_idxs(rationales)
        non_rationale_idxs = np.delete(np.arange(num_bases), rationale_idxs)
        return non_rationale_idxs

    def get_x_rationale_only(self, rationales, replacement=None):
        if replacement is None:
            replacement = self.get_replacement()
        modified_x = np.array(self.get_x(copy=True), dtype='float32')
        non_rationale_idxs = self.get_nonrationale_idxs(rationales)
        for i in non_rationale_idxs:
            self.replace_at(modified_x, replacement, i)
        return modified_x

    def get_x_nonrationale_only(self, rationales, replacement=None):
        if replacement is None:
            replacement = self.get_replacement()
        modified_x = np.array(self.get_x(copy=True), dtype='float32')
        rationale_idxs = self.get_all_rationale_idxs(rationales)
        for i in rationale_idxs:
            self.replace_at(modified_x, replacement, i)
        return modified_x

    def set_predictions(self, model, rationales, replacement=None):
        self.original_prediction = sis.predict_for_embed_sequence(
            [self.get_x()], model)[0]
        self.prediction_rationale_only = sis.predict_for_embed_sequence(
            [self.get_x_rationale_only(rationales,
             replacement=replacement)], model)[0]
        self.prediction_nonrationale_only = sis.predict_for_embed_sequence(
            [self.get_x_nonrationale_only(rationales,
             replacement=replacement)], model)[0]

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_threshold_f(self, threshold_f):
        self.threshold_f = threshold_f

    def make_threshold_f(self):
        if self.threshold is None:
            raise TypeError('Must set threshold attribute.')
        threshold_f = lambda prob: prob >= self.threshold
        self.set_threshold_f(threshold_f)

    def run_sis_rationales(self, model, replacement=None,
                                       first_only=True, verbose=False):
        if replacement is None:
            replacement = self.get_replacement()
        all_rationales = self.get_rationales(SIS_RATIONALE_KEY)
        if first_only and len(all_rationales) >= 1:
            if verbose:
                print('Already >= 1 rationale and first_only=True, returning.')
            return None
        x_nonrationale = self.get_x_nonrationale_only(all_rationales,
            replacement=replacement)
        current_nonrationale_pred = sis.predict_for_embed_sequence(
            [x_nonrationale], model)[0]
        if verbose:
            print('Starting prediction %.3f' % current_nonrationale_pred)
        while self.threshold_f(current_nonrationale_pred):
            # prediction on non-rationale is still beyond threshold
            if verbose:
                print('Prediction beyond threshold, extracting rationale')
            removed_elts, history = sis.sis_removal_tf(
                x_nonrationale.copy(),
                model,
                replacement,
                return_history=True,
                verbose=False)
            rationale_length = sis.find_min_words_needed(history,
                self.threshold_f)
            rationale_elems = removed_elts[-rationale_length:]
            rationale = Rationale(elms=rationale_elems[::-1],
                                  history=(removed_elts, history))
            self.add_rationale(rationale, SIS_RATIONALE_KEY)
            # mask new rationale in the sequence and re-predict
            all_rationales = self.get_rationales(SIS_RATIONALE_KEY)
            x_nonrationale = self.get_x_nonrationale_only(
                all_rationales, replacement=replacement)
            current_nonrationale_pred = sis.predict_for_embed_sequence(
                [x_nonrationale], model)[0]
            if verbose:
                print('New predicted score %.3f' % current_nonrationale_pred)
            if first_only:
                if verbose:
                    print('Only 1 rationale, first_only=True, breaking.')
                break
        if verbose:
            print('Done building rationales.')

    def run_integrated_gradients_rationale(self, ig_model, model, baseline,
                                            replacement=None, verbose=False):
        if len(self.get_rationales(IG_SUFF_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have IG rationale, returning.')
            return None
        if verbose:
            print('Running integrated gradients rationale.')
        if replacement is None:
            replacement = self.get_replacement()
        x = self.get_x(copy=True)
        ig_vals = ig_model.explain(x, reference=baseline, num_steps=300)
        # L1 norm along the 4-dim one-hot embeddings axis
        ig_vals = np.linalg.norm(ig_vals, ord=1, axis=1)
        ig_order = np.argsort(ig_vals)
        score_history = sis.find_score_history_given_order_tf(
            x, ig_order, model, baseline)
        rationale_length = sis.find_min_words_needed(score_history,
            self.threshold_f)
        rationale_elems = ig_order[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(ig_order, score_history))
        self.add_rationale(rationale, IG_SUFF_RATIONALE_KEY)
        if verbose:
            print('Done with integrated gradients.')

    # In this Top IG baseline, determine the rationale length by first
    #  ordering positions by masked L1 norm along each position's embedding.
    # So only using abs(IG) value for the correct base at each position.
    # Compute `target_igs = threshold - prediction at IG baseline`
    # Then using the L1 position ordering, add elements into rationale
    #  (largest L1 norm first) until sum of all IGs (non-absolute sum)
    #  of rationale elements is >= target_igs.
    # Should use a all-zeros baseline here so the IG vals are also one-hot
    #   prior to masking / L1 (absolute value).
    def run_integrated_gradients_top_rationale(self, ig_model, model,
                                                    baseline, replacement=None,
                                                    verbose=False):
        if len(self.get_rationales(IG_TOP_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have Top IG rationale, returning.')
            return None
        if verbose:
            print('Running Top IG rationale.')
        if replacement is None:
            replacement = self.get_replacement()
        x = self.get_x(copy=True)
        ig_baseline_prediction = sis.predict_for_embed_sequence([baseline],
            model)[0]
        target_igs_sum = self.threshold - ig_baseline_prediction
        ig_vals = ig_model.explain(x, reference=baseline, num_steps=300)
        ig_masked = np.multiply(ig_vals, x)
        # L1 norm along the 4-dim one-hot embeddings axis
        #  (only 1 non-zero element though, after masking)
        ig_masked_abs = np.linalg.norm(ig_masked, ord=1, axis=1)
        ig_sums = np.sum(ig_vals, axis=1)
        ig_abs_sums = list(zip(ig_masked_abs, ig_sums))
        ig_abs_sums_sorted = sorted(ig_abs_sums, key=lambda x: x[0])
        ig_abs_sorted, ig_sums_sorted = zip(*ig_abs_sums_sorted)
        ig_sums_sorted_cumsum = np.cumsum(ig_sums_sorted[::-1])[::-1]
        ig_order = np.argsort(ig_masked_abs)
        rationale_length = sis.find_min_words_needed(ig_sums_sorted_cumsum,
            lambda s: s >= target_igs_sum)
        rationale_elems = ig_order[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(ig_order, ig_sums_sorted_cumsum))
        self.add_rationale(rationale, IG_TOP_RATIONALE_KEY)
        if verbose:
            print('Done with Top IG.')

    # Integrated gradients baseline where length is fixed to same as median SIS length
    # Should input a baseline of zeros vectors.
    def run_integrated_gradients_fixed_length_rationale(self, ig_model, model,
                                                        baseline,
                                                        replacement=None,
                                                        verbose=False):
        if len(self.get_rationales(IG_FIXED_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have fixed length IG rationale, returning.')
            return None
        if verbose:
            print('Running fixed length IG rationale.')
        if replacement is None:
            replacement = self.get_replacement()
        x = self.get_x(copy=True)
        ig_vals = ig_model.explain(x, reference=baseline, num_steps=300)
        ig_masked = np.multiply(ig_vals, x)
        # L1 norm along the 4-dim one-hot embeddings axis
        #  (only 1 non-zero element though, after masking)
        ig_masked_abs = np.linalg.norm(ig_masked, ord=1, axis=1)
        ig_order = np.argsort(ig_masked_abs)
        rationale_length = self.get_fixed_baseline_length()
        rationale_elems = ig_order[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(ig_order, None))
        self.add_rationale(rationale, IG_FIXED_RATIONALE_KEY)
        if verbose:
            print('Done with fixed length IG.')

    # LIME baseline where length is fixed to same as median SIS length
    def run_lime_fixed_length_rationale(self, pipeline, decoder, model,
                                         verbose=False):
        if len(self.get_rationales(LIME_FIXED_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have fixed length LIME rationale, returning.')
            return None
        if verbose:
            print('Running fixed length LIME rationale.')
        seq_string = ' '.join(decoder(self.get_x()))
        explainer = lime_helper.make_explainer(verbose=False)
        explanation = lime_helper.explain(seq_string, explainer, pipeline,
            num_features=101)
        sorted_words = lime_helper.extract_word_order(explanation)
        rationale_length = self.get_fixed_baseline_length()
        rationale_elems = sorted_words[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(sorted_words, None))
        self.add_rationale(rationale, LIME_FIXED_RATIONALE_KEY)
        if verbose:
            print('Done with LIME.')

    # Perturbative baseline where length is fixed to same as median SIS length
    def run_perturb_fixed_length_rationale(self, model, replacement=None,
                                            verbose=False):
        if len(self.get_rationales(PERTURB_FIXED_RATIONALE_KEY)) >= 1:
            if verbose:
                print('Already have fixed length perturbative baseline rationale,',
                        'returning.')
            return None
        if verbose:
            print('Running perturbative baseline rationale.')
        if replacement is None:
            replacement = self.get_replacement()
        x = self.get_x()
        removed_scores = sis.removed_word_predictions_tf(x, model,
            replacement)
        sorted_words = removed_scores.argsort()
        # word with biggest drop in score (lowest final score) at end of
        #   sorted list
        sorted_words = sorted_words[::-1]
        rationale_length = self.get_fixed_baseline_length()
        rationale_elems = sorted_words[-rationale_length:]
        rationale = Rationale(elms=rationale_elems[::-1],
                              history=(sorted_words, None))
        self.add_rationale(rationale, PERTURB_FIXED_RATIONALE_KEY)
        if verbose:
            print('Done with fixed length perturbative baseline.')

    # Returns length to use for fixed-length IG, LIME, and perturbative baselines
    # If only single SIS, returns that length. If multiple, returns
    #   median rounded to nearest integer.
    def get_fixed_baseline_length(self):
        sis_rationales = self.get_rationales(SIS_RATIONALE_KEY)
        if len(sis_rationales) == 0:
            raise ValueError('Must first compute SIS rationales.')
        if len(sis_rationales) == 1:
            return len(sis_rationales[0])
        med = np.median([len(r) for r in sis_rationales])
        return int(np.rint(med))

    def perturbation(self, model, replacement=None,
                     diffs_transform_f=lambda preds_orig: \
                         preds_orig[1] - preds_orig[0]):
        perturb_idxs_scores = []
        if replacement is None:
            replacement = self.get_replacement()
        x = self.get_x(copy=True)
        preds = sis.removed_word_predictions_tf(x, model, replacement)
        original_pred = self.original_prediction if self.original_prediction \
                            is not None else sis.predict_for_embed_sequence(
                            [self.get_x()], model)[0]
        diffs = diffs_transform_f((np.array(preds), original_pred))
        return diffs

    def perturbation_rationale(self, model, rationales,
                                replacement=None,
                                diffs_transform_f=lambda preds_orig: \
                                    preds_orig[1] - preds_orig[0]):
        diffs = self.perturbation(model,
            replacement=replacement,
            diffs_transform_f=diffs_transform_f)
        rationale_idxs = self.get_all_rationale_idxs(rationales)
        rationale_diffs = np.take(diffs, rationale_idxs)
        nonrationale_diffs = np.delete(diffs, rationale_idxs)
        assert(diffs.shape[0] == \
               rationale_diffs.shape[0] + nonrationale_diffs.shape[0])
        return rationale_diffs, nonrationale_diffs

    def to_html(self, rationale):
        pass

    def to_text(self, rationale):
        pass

    def to_json(self, f):
        json.dump(self.__dict__, f, cls=ExampleJSONEncoder)

    @classmethod
    def from_json(cls, f, set_threshold_f=True):
        data = json.load(f)
        seq = cls()
        for k, v in data.items():
            if k == 'rationales':  # construct Rationale objects
                rationales = {}
                for k_ in v.keys():
                    for i in range(len(v[k_])):
                        v[k_][i] = Rationale.from_json_str(v[k_][i])
            elif (k in cls.JSON_NUMPY_ATTRIBS) and v is not None:
                # cast to np array
                v = np.array(v)
            setattr(seq, k, v)
        if set_threshold_f:
            seq.make_threshold_f()
        return seq


# Container class for storing DNASequence objects
class DNASequenceContainer(object):
    def __init__(self, threshold=None, trained_model_path=None):
        self.sequences = []
        self.i_to_sequence = {}
        self.threshold = threshold
        self.trained_model_path = trained_model_path

    def add_sequence(self, sequence):
        if sequence.i in self.i_to_sequence:
            raise KeyError('Review %d already in container' % (sequence.i))
        self.sequences.append(sequence)
        self.i_to_sequence[sequence.i] = sequence

    def get_sequence(self, i):
        if i not in self.i_to_sequence:
            raise KeyError('Sequence %d not in container' % (i))
        return self.i_to_sequence[i]

    def get_sequences(self):
        return self.sequences

    def get_threshold(self):
        return self.threshold

    def get_idxs(self):
        return sorted(self.i_to_sequence.keys())

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def metadata_filename():
        return 'metadata.json'

    def dump_data(self, dir_path):
        # Make directories to dirpath path if not exists
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        metadata = {}

        metadata['trained_model_path'] = self.trained_model_path
        metadata['threshold'] = self.threshold

        # Dump sequences to JSON files
        metadata['sequences'] = []

        sequences_dir = 'sequences'
        sequences_dir_path = os.path.join(dir_path, sequences_dir)
        if not os.path.isdir(sequences_dir_path):
            os.makedirs(sequences_dir_path)

        for seq in self.get_sequences():
            sequence_file = os.path.join(sequences_dir, '%d.json' % (seq.i))
            metadata['sequences'].append(sequence_file)
            review_path = os.path.join(dir_path, sequence_file)
            with open(review_path, 'w') as f:
                seq.to_json(f)

        metadata_file = os.path.join(dir_path, self.metadata_filename())
        with open(metadata_file, 'w') as outfile:
            json.dump(metadata, outfile)

    @staticmethod
    def load_data(dir_path, target_cls=DNASequence):
        metadata_file = os.path.join(dir_path,
                                     DNASequenceContainer.metadata_filename())
        with open(metadata_file, 'r') as infile:
            metadata = json.load(infile)

        args = {}
        args['trained_model_path'] = metadata['trained_model_path']
        args['threshold'] = metadata['threshold']

        container = DNASequenceContainer(**args)

        # Load DNASequence objects
        for sequence_file in metadata['sequences']:
            sequence_path = os.path.join(dir_path, sequence_file)
            with open(sequence_path, 'r') as f:
                seq = target_cls.from_json(f)
            container.add_sequence(seq)

        return container


class Image(Example):
    def __init__(self, x=None, i=None, replacement_pixel=None, class_idx=None,
                    threshold=None, threshold_f=None):
        super(Image, self).__init__(x=x, i=i)
        self.rationales = {}
        self.replacement_pixel = replacement_pixel
        self.original_prediction = None
        self.prediction_rationale_only = None
        self.prediction_nonrationale_only = None
        self.class_idx = class_idx
        self.threshold = threshold
        self.threshold_f = threshold_f
        if threshold is not None and threshold_f is None:
            self.make_threshold_f()

    def get_rationales(self, method):
        if method not in self.rationales:
            return []
        return self.rationales[method]

    def get_x(self, copy=True):
        if copy:
            return np.copy(self.x)
        return self.x

    def get_shape(self, channels_last=True):
        shape = self.x.shape
        if channels_last:
            shape = shape[:2]
        return shape

    def get_num_pixels(self):
        return np.prod(self.get_shape(channels_last=True))

    def i_to_pos(self, i):
        return np.unravel_index(i, self.x.shape)

    def pos_to_i(self, pos_tuple):
        return np.ravel_multi_index(pos_tuple, self.x.shape)

    def add_rationale(self, rationale, method):
        if method not in self.rationales:
            self.rationales[method] = []
        self.rationales[method].append(rationale)

    def set_replacement_pixel(self, replacement_pixel):
        self.replacement_pixel = replacement_pixel

    def get_replacement_pixel(self):
        if self.replacement_pixel is None:
            return TypeError('Must set replacement_pixel. Cannot be None.')
        return self.replacement_pixel

    def get_all_rationale_idxs(self, rationales):
        rationale_idxs = []
        for rationale in rationales:
            rationale_idxs += list(rationale)
        return np.asarray(rationale_idxs)

    def get_nonrationale_idxs(self, rationales):
        num_pixels = self.get_num_pixels()
        rationale_idxs = self.get_all_rationale_idxs(rationales)
        non_rationale_idxs = np.delete(np.arange(num_pixels), rationale_idxs)
        return non_rationale_idxs

    def get_x_rationale_only(self, rationales, replacement=None):
        if replacement is None:
            replacement = self.get_replacement_pixel()
        modified_x = self.get_x(copy=True)
        non_rationale_idxs = self.get_nonrationale_idxs(rationales)
        for i in non_rationale_idxs:
            pos = self.i_to_pos(i)
            modified_x[pos] = replacement
        return modified_x

    def get_x_nonrationale_only(self, rationales, replacement=None):
        if replacement is None:
            replacement = self.get_replacement_pixel()
        modified_x = self.get_x(copy=True)
        rationale_idxs = self.get_all_rationale_idxs(rationales)
        for i in rationale_idxs:
            pos = self.i_to_pos(i)
            modified_x[pos] = replacement
        return modified_x

    def set_predictions(self, model, rationales, replacement=None):
        self.original_prediction = sis.predict_for_images(
            [self.get_x()], model)[0]
        self.prediction_rationale_only = sis.predict_for_images(
            [self.get_x_rationale_only(rationales,
             replacement=replacement)], model)[0]
        self.prediction_nonrationale_only = sis.predict_for_images(
            [self.get_x_nonrationale_only(rationales,
             replacement=replacement)], model)[0]

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_threshold_f(self, threshold_f):
        self.threshold_f = threshold_f

    def make_threshold_f(self):
        if self.threshold is None:
            raise TypeError('Must set threshold attribute.')
        threshold_f = lambda prob: prob >= self.threshold
        self.set_threshold_f(threshold_f)

    def run_sis_rationales(self, model, replacement=None,
                                       first_only=True, verbose=False):
        if replacement is None:
            replacement = self.get_replacement_pixel()
        if self.original_prediction is None:
            self.original_prediction = sis.predict_for_images(
                [self.get_x()], model)[0]
        orig_predicted_class, orig_class_prob = sis.pred_class_and_prob(
            self.original_prediction)
        all_rationales = self.get_rationales(SIS_RATIONALE_KEY)
        if first_only and len(all_rationales) >= 1:
            if verbose:
                print('Already >= 1 rationale and first_only=True, returning.')
            return None
        if verbose:
            print('Starting probability for class %d: %.5f' % \
                    (orig_predicted_class, orig_class_prob))
        x_nonrationale = self.get_x_nonrationale_only(all_rationales,
            replacement=replacement)
        current_nonrationale_pred = sis.predict_for_images(
            [x_nonrationale], model)[0][self.class_idx]
        while self.threshold_f(current_nonrationale_pred):
            # prediction on non-rationale is still beyond threshold
            if verbose:
                print('Prediction beyond threshold, extracting rationale')
            removed_elts, history = sis.sis_removal_img_classif(
                x_nonrationale.copy(),
                self,
                self.class_idx,
                model,
                replacement,
                return_history=True,
                verbose=False)
            rationale_length = sis.find_min_words_needed(history,
                self.threshold_f)
            rationale_elems = removed_elts[-rationale_length:]
            rationale = Rationale(elms=rationale_elems[::-1],
                                  history=(removed_elts, history))
            self.add_rationale(rationale, SIS_RATIONALE_KEY)
            # mask new rationale in the sequence and re-predict
            all_rationales = self.get_rationales(SIS_RATIONALE_KEY)
            x_nonrationale = self.get_x_nonrationale_only(
                all_rationales, replacement=replacement)
            current_nonrationale_pred = sis.predict_for_images(
                [x_nonrationale], model)[0][self.class_idx]
            if verbose:
                print('New probability %.5f' % current_nonrationale_pred)
            if first_only:
                if verbose:
                    print('Only 1 rationale, first_only=True, breaking.')
                break
        if verbose:
            print('Done building rationales.')

    def perturbation(self, model, replacement_embedding='mean',
                     diffs_transform_f=lambda preds_orig: \
                         preds_orig[1] - preds_orig[0]):
        perturb_idxs_scores = []
        replacement_embedding = self.get_replacement_embedding(
            replacement_embedding=replacement_embedding)
        x_embed = self.get_embedded_sequence()
        preds = sis.removed_word_predictions(x_embed, model, self.num_pad,
                                                replacement_embedding)
        original_pred = self.original_prediction if self.original_prediction \
                            is not None else sis.predict_for_embed_sequence(
                            [self.get_embedded_sequence()], model)[0]
        diffs = diffs_transform_f((np.array(preds), original_pred))
        return diffs

    def perturbation_rationale(self, model, rationales,
                                replacement_embedding='mean',
                                diffs_transform_f=lambda preds_orig: \
                                    preds_orig[1] - preds_orig[0]):
        diffs = self.perturbation(model,
            replacement_embedding=replacement_embedding,
            diffs_transform_f=diffs_transform_f)
        rationale_idxs = self.get_all_rationale_idxs(rationales)
        rationale_diffs = np.take(diffs, rationale_idxs)
        nonrationale_diffs = np.delete(diffs, rationale_idxs)
        assert(diffs.shape[0] == \
               rationale_diffs.shape[0] + nonrationale_diffs.shape[0])
        return rationale_diffs, nonrationale_diffs

    def to_html(self, rationale):
        pass

    def to_json(self, f):
        json.dump(self.__dict__, f, cls=ExampleJSONEncoder)

    @staticmethod
    def from_json(f, set_threshold_f=True):
        data = json.load(f)
        image = Image()
        for k, v in data.items():
            if k == 'rationales':  # construct Rationale objects
                rationales = {}
                for k_ in v.keys():
                    for i in range(len(v[k_])):
                        v[k_][i] = Rationale.from_json_str(v[k_][i])
            elif (k == 'x') and v is not None:
                # cast to np array
                v = np.array(v)
            setattr(image, k, v)
        if set_threshold_f:
            image.make_threshold_f()
        return image


class ImageContainer(object):
    def __init__(self):
        pass
