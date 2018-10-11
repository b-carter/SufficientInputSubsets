import os

import sis
from lime.lime_text import LimeTextExplainer


## Based on some code at:
##   http://data4thought.com/deep-lime.html

class TextPipeline(object):
    def __init__(self, model, tokenizer, padder):
        self.model = model
        self.tokenizer = tokenizer
        self.padder = padder
    
    # `texts` is a list of d strings
    def predict_proba(self, texts):
        tokenized_texts = self.tokenizer.texts_to_sequences(texts)
        padded_seqs = self.padder(tokenized_texts)
        predictions = self.model.predict(padded_seqs, batch_size=128)
        return predictions


class DNASequencePipeline(object):
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder

    # `seqs` is a list of d strings, each base separated by space
    #    (may also include 'UNKWORDZ' token)
    # Returns (d x 1) list of predictions
    def predict_proba(self, seqs):
        seqs_parsed = [s.replace('UNKWORDZ', 'N').replace(' ', '') \
                            for s in seqs]
        encoded_seqs = [self.encoder(s) for s in seqs_parsed]
        predictions = sis.predict_for_embed_sequence(encoded_seqs,
            self.model, batch_size=5000)
        return predictions.reshape(-1, 1)


# For beer reviews data
def make_pipeline(model, tokenizer, max_words):
    pipeline = TextPipeline(model, tokenizer,
        lambda s: sis.pad_sequences(s, max_words=max_words))
    return pipeline

# For DNA sequence data
def make_pipeline_dna_seq(model, encoder):
    pipeline = DNASequencePipeline(model, encoder)
    return pipeline

def make_explainer(verbose=False):
    explainer = LimeTextExplainer(class_names=['prediction'],
                                  split_expression=' ',
                                  bow=False,
                                  verbose=verbose)
    return explainer

def explain(text, explainer, pipeline, num_features=500, num_samples=5000):
    explanation = explainer.explain_instance(text,
                                             pipeline.predict_proba,
                                             labels=(0,),
                                             num_features=num_features,
                                             num_samples=num_samples)
    return explanation

def extract_word_order(explanation):
    word_order, weight = zip(*explanation.as_map()[0][::-1])
    # should be ordered by absolute value of weights, highest weight last
    return word_order
