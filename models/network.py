import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from models.nn.linear import * 
from models.nn.rnn import *


def char_cnn(inputs, dim_char=100, scope=None, nlayers=1, win_len=3, num_filters=30, keep_prob=0.5):
    """Convolutional Character Embedding Network
    Args:
        inputs: input embeddings without dropout
        dim_char: input_channels
        win_len:  filter_width
        num_filters: output_channels
        keep_prob: dropout initial-embedd before input to cnn
    Returns:
        shape = (batch size, max sentence length, char hidden size)

    """
    with tf.variable_scope(scope or "char_cnn"):
        # reshape and dropout
        s = tf.shape(inputs)
        embedd = tf.reshape(inputs, shape=[s[0] * s[1], s[2], dim_char])
        embedd_dropout = tf.nn.dropout(embedd, keep_prob)

        # filter shape is [filter_width, in_width, out_channels]
        filter_shape = [win_len, dim_char, num_filters]
        W_conv = tf.get_variable("W_conv", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
        b_conv = tf.get_variable("b_conv", initializer=tf.constant(0.1, shape=[num_filters]))
        
        # convolution and pooling
        conv = tf.nn.conv1d(embedd_dropout,
                            W_conv,
                            stride=1,
                            padding="SAME",
                            name="conv")
        conv1 = tf.nn.tanh(conv + b_conv)
        pooled = tf.reduce_max(
            conv1,
            axis=1,
            keepdims=True
        )

        return tf.reshape(pooled, [s[0], s[1], num_filters], name="char_pool_flat")


def char_bilstm(inputs, word_len, dim_char=100, scope=None, nlayers=1, cell_sz=50):
    """ BiLSTM Character Embedding Network
    Args:
        inputs: input embeddings without dropout, [batch, max_step, max_char, depth]
        word_len: sequence length of each word, [batch, max_step]
        cell_sz: lstm-cell size
    Returns:
        shape = [batch, max_step, 2*cell_sz]

    """
    with tf.variable_scope(scope or "char_bilstm"):
        # put the time dimension on axis=1
        shapes = tf.shape(inputs)
        embedd = tf.reshape(inputs, shape=[shapes[0] * shapes[1], shapes[-2], dim_char])
        word_len = tf.reshape(word_len, shape=[shapes[0] * shapes[1]])

        # bilstm on chars, output -> [batch*max_step, depth]
        _, output = birnn_multilayer(embedd, word_len, nlayers=nlayers, cell_sz=cell_sz, 
        rnn_dropout=True, vi_dropout=False, highway=False, self_attention=False, 
        input_keep_prob=0.67, output_keep_prob=1.0, state_keep_prob=0.67)

        return tf.reshape(output, shape=[shapes[0], shapes[1], 2*cell_sz])


def birnn_encoder(inputs, seq_len, scope=None, rnn_cell="lstm", nlayers=3, cell_sz=300, 
        rnn_dropout=True, vi_dropout=True, highway=True, self_attention=False, 
        input_keep_prob=0.67, output_keep_prob=1.0, state_keep_prob=0.67,
        attention_mechanism="additive", attention_sz=100, dot_norm=True):
    """ Invoking birnn() """
    birnn_logits, _ =  birnn_multilayer(inputs, seq_len, scope, rnn_cell, nlayers, cell_sz, 
        rnn_dropout, vi_dropout, highway, self_attention, 
        input_keep_prob, output_keep_prob, state_keep_prob,
        attention_mechanism, attention_sz, dot_norm)

    return birnn_logits


def attention_rnn(inputs, seq_len, cell_sz=300, scope=None,
        rnn_dropout=True, vi_dropout=True, keep_prob=0.67,
        attention_mechanism="additive", attention_sz=100, dot_norm=True):
    """ Invoking birnn() """
    attention_logits, _ =  birnn_multilayer(inputs, seq_len, scope=scope,
            rnn_cell="lstm", nlayers=1, cell_sz=cell_sz, 
            rnn_dropout=rnn_dropout, vi_dropout=vi_dropout, highway=False, self_attention=True, 
            input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob,
            attention_mechanism=attention_mechanism, attention_sz=attention_sz, dot_norm=dot_norm)

    return attention_logits


def linear_projection(inputs, output_sz, scope=None,  n_split=1, bias=False, activate_func=None, keep_prob=1.0):
    """ Linear Layer 
    Args:
        inputs: should not be tuple
    Returns:
        n_split-part affine_projections
        
    """
    if keep_prob < 1.0:
        ndims = inputs.shape.ndims
        shape = inputs.get_shape().as_list()
        noise_shape = [shape[0]]+[1]*(ndims-2)+[shape[-1]]
        inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

    return affine(inputs, output_sz, n_split, bias, activate_func, scope=scope)


def linear_classifier(inputs, output_sz, scope=None, n_split=1, bias=False, activate_func=None, keep_prob=1.0):
    """ Linear classifier 
    Returns:
        list(n_split>1): outputs_list, probs_list, preds_list
        tensor(n_split=1): outputs, probs, preds

    """
    if keep_prob < 1.0:
        ndims = inputs.shape.ndims
        shape = inputs.shape.as_list()
        noise_shape = [shape[0]]+[1]*(ndims-2)+[shape[-1]]
        inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

    outputs = affine(inputs, output_sz, n_split, bias, activate_func, scope=scope)
    if n_split > 1:
        probs_list = []
        preds_list = []
        for output in outputs:
            probs = tf.nn.softmax(output, axis=-1)
            preds = tf.argmax(probs, axis=-1)
            probs_list.append(probs)
            preds_list.append(preds)

        return outputs, probs_list, preds_list
    else:
        probs = tf.nn.softmax(output, axis=-1)
        preds = tf.argmax(probs, axis=-1)

        return outputs, probs, preds


def biaffine_projection(inputs1, inputs2, scope=None, output_sz=1, 
        bias1=True, bias2=True, activate_func=None, keep_prob=1.0):
    """ Biaffine Projection 
    Returns:
        outputs: [batch, max_step1, output_sz, max_step2], 
            if output_sz is 1, squeeze to [batch, max_step1, max_step2]
    """
    if keep_prob < 1.0:
        ndims1 = inputs1.shape.ndims
        shape1 = inputs1.get_shape().as_list()
        noise_shape1 = [shape1[0]]+[1]*(ndims1-2)+[shape1[-1]]
        inputs1 = tf.nn.dropout(inputs1, keep_prob=keep_prob)   
        ndims2 = inputs2.shape.ndims
        shape2 = inputs2.shape.as_list()
        noise_shape2 = [shape2[0]]+[1]*(ndims2-2)+[shape2[-1]]
        inputs2 = tf.nn.dropout(inputs2, keep_prob=keep_prob) 
    
    # outputs shape: [batch, max_step1, output_sz, max_step2]
    outputs = biaffine(inputs1, inputs2, output_sz, bias1, bias2, activate_func, scope=scope)
    if output_sz == 1:
        outputs = tf.squeeze(outputs, 2)

    return outputs


def attention_layer(memory, query, additive=True, dot_product=False, 
        scope="attention", reuse=None, memory_len=None, norm_dot=True,
        attention_sz=100, bias=False, activate_func=tf.tanh):
    """ Invoking attention() """
    if query is None:
        # convert memory[batch, step, depth] -> context[batch, attn_sz]
        context, score = attention(memory, query, scope, reuse, memory_len, 
                additive, dot_product, norm_dot, attention_sz, bias, activate_func)
    elif query.shape.ndims == 3:
        # convert memory[batch, step, depth] & query[batch, step, depth] ->
        # global_seq_context[batch, step, depth]
        context, score = attention_global(memory, query, scope, reuse, memory_len, 
                additive, dot_product, norm_dot, attention_sz, bias, activate_func)
    else: 
        # convert memory[batch, step, depth] & query[batch, 1, depth] ->
        # local_seq_context[batch, depth]
        context, score = attention(memory, query, scope, reuse, memory_len, 
                additive, dot_product, norm_dot, attention_sz, bias, activate_func)

    return context


def highway_layer(input_tensor, num_layers, dim, scope=None):
    """Highway Network

    """
    with tf.variable_scope(scope or "highway_network"):
        prev = input_tensor
        cur = None
        for layer_idx in range(num_layers):
            cur = highway(input_tensor, dim, scope="layer_{}".format(layer_idx))
            prev = cur
            
        return cur


def gated_concat_connection(common_input, specific_input, scope=None):
    """Gated connection of two input-embeddings

    """
    with tf.variable_scope(scope or "gated_concat_layer"):
        orig_shape = tf.shape(specific_input)
        concat_input   = tf.concat([common_input, specific_input], axis=-1)
        common_input   = tf.reshape(common_input, shape=[orig_shape[0]*orig_shape[1], orig_shape[-1]])
        specific_input = tf.reshape(specific_input, shape=[orig_shape[0]*orig_shape[1], orig_shape[-1]])
        added_input    = tf.add(common_input, specific_input)

        dim = orig_shape[-1].value
        W_g = tf.get_variable("W_g", dtype=tf.float32, shape=[dim, dim])
        b_g = tf.get_variable("b_g", dtype=tf.float32, shape=[dim], initializer=tf.zeros_initializer())
        gate = tf.nn.xw_plus_b(x=added_input, weights=W_g, biases=b_g)
        gate = tf.nn.sigmoid(gate)

        outputs = gate * added_input + (1 - gate) * specific_input
        output_tensor = tf.reshape(outputs, shape=orig_shape)

        return output_tensor


