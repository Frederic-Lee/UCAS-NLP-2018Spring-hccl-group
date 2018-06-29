import tensorflow as tf
import os
from scripts.elmo.bilm import Batcher, BidirectionalLanguageModel, weight_layers

def elmo_func(vocab_file, options_file, weight_file):
    # Input placeholders to the biLM.
    context_character_ids  = tf.placeholder('int32', shape=(None, None, None))
    question_character_ids = tf.placeholder('int32', shape=(None, None, None))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file)

    # Get ops to compute the LM embeddings.
    context_embeddings_op  = bilm(context_character_ids)
    question_embeddings_op = bilm(question_character_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
    with tf.variable_scope('', reuse=True):
        # the reuse=True scope reuses weights from the context for the question
        elmo_question_input = weight_layers(
            'input', question_embeddings_op, l2_coef=0.0
        )
