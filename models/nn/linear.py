import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def affine(inputs, output_sz, n_split=1, bias=False, activate_func=None, scope=None):
    """ Dense layer for affine transformation 
    Args:
        inputs: list of tensors to be affined separately
        output_sz: outputs size for single part
        n_split: if n_split>1, outputs *= n_split
    Returns:
        added affine transformations
        if n_split>1, return list of outputs; else single tensor

    """
    with tf.variable_scope(scope or "affine"):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        ranks = [inp.shape.ndims for inp in inputs]
        rank  = ranks[0]
        if any([rk-rank for rk in ranks]):
            raise ValueError("inputs do not agree on ranks: %s" % ranks)
        if n_split > 1:
            output_sz = output_sz * n_split
        
        # affine transformation
        affine_trans_list = []
        for i, _input in enumerate(inputs):
            input_sz = _input.get_shape().as_list()[-1]
            output_shape = []
            _shape = tf.shape(_input)
            for j in range(rank-1):
                output_shape.append(_shape[j])
            output_shape.append(output_sz)
            output_shape = tf.stack(output_shape)

            _input = tf.reshape(_input, shape=[-1, input_sz])
            #_input: shape=[-1, input_sz], weight: shape=[input_sz, output_sz]
            weight = tf.get_variable(
                    name="weight_%d"%(i),
                    shape=[input_sz, output_sz],
                    dtype=tf.float32,
                    regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4))
            affine_trans = tf.matmul(_input, weight)
            affine_trans = tf.reshape(affine_trans, shape=output_shape)
            affine_trans_list.append(affine_trans)
        # sum(list[]) same as a+b, tf.add_n(list[]) can make error.
        # e.g. [32,50,1,100]+[32,1,80,100] -> [32,50,80,100]
        outputs = sum(affine_trans_list)

        # add_bias and activation function
        if bias is True:
            bias = tf.get_variable(
                    name="bias",
                    shape=[output_sz],
                    dtype=tf.float32)
            outputs = outputs + bias
        if activate_func is not None:
            outputs = activate_func(outputs)

        if n_split > 1:
            outputs = tf.split(outputs, num_or_size_splits=n_split, axis=-1)

        return outputs


def biaffine(inputs1, inputs2, output_sz, bias1=True, bias2=True, activate_func=None, scope=None):
    """ Dense layer for affine transformation 
    Args:
        inputs1, inputs2: tensor to be affined
        output_sz: outputs size
        bias1, bias2: bias for the two inputs
        bias: bias for total inputs
    Returns:
        added affine transformations

    """
    with tf.variable_scope(scope or "biaffine"):
        # inputs shape: [batch, max_step, depth]
        shape1 = tf.shape(inputs1)
        shape2 = tf.shape(inputs2)
        input1_sz = inputs1.get_shape().as_list()[-1]+bias1
        input2_sz = inputs2.get_shape().as_list()[-1]+bias2
        # outputs shape: [batch, max_step1, depth, max_step2]
        output_shape = tf.stack([shape1[0], shape1[1], output_sz, shape2[1]])

        # add bias
        if bias1 is True:
            inputs1 = tf.concat([inputs1, tf.ones(tf.stack([shape1[0], shape1[1], 1]))], -1)
        if bias2 is True:
            inputs2 = tf.concat([inputs2, tf.ones(tf.stack([shape2[0], shape2[1], 1]))], -1)
        
        # affine transformation
        weight = tf.get_variable(
                name="weight",
                shape=[input1_sz, output_sz, input2_sz],
                dtype=tf.float32,
                initializer = tf.orthogonal_initializer,
                regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4))
        # [batch*max_step1, output_sz*depth2]
        affine = tf.matmul(tf.reshape(inputs1, [-1, input1_sz]), 
                tf.reshape(weight, [input1_sz, -1]))
        # [batch, max_step1*output_sz, max_step2]
        biaffine = tf.matmul(tf.reshape(affine, tf.stack([-1, shape1[1]*output_sz, input2_sz])),
                inputs2, transpose_b=True)
        # [batch, max_step1, output_sz, max_step2]
        biaffine = tf.reshape(biaffine, tf.stack([-1, output_sz, shape2[1]]))
        outputs  = tf.reshape(biaffine, output_shape)

        if activate_func is not None:
            outputs = activate_func(outputs)

        return outputs


def attention(memory, query, scope="attention", reuse=None, memory_len=None, 
        mechanism="additive", attention_sz=100, bias=False, activate_func=tf.tanh,
        dot_norm=True):
    """Attention with single query.
    Support one-query to many-memory, e.g. each step for RNN.

    Args:
        memory: key-value embeddings, expected to be [batch, depth] 
        query: query embeddings that multiply scores, [batch, max_step_query, depth]
        additive: additive attention, f_att(q, M) = v^T * tanh(W_m*M+W_q*q)
            "Neural Machine Translation by Jointly Learning to Align and Translate."
            ICLR 2015. https://arxiv.org/abs/1409.0473
        dot_product: dot_product attention, f_att(q, m_i) = q^T*m_i
            "Effective Approaches to Attention-based Neural Machine Translation."
            EMNLP 2015.  https://arxiv.org/abs/1508.04025
        memory_len: length of memory, [batch], for masking
        norm_dot: normalize over dot_product score, f_att()/sqrt(dim_key) 
        attention_sz, bias, activate_func: for additive attention
    Returns:
        context: context vector
        score: softmax scores

    """
    with tf.variable_scope(scope, reuse=reuse):
        # query: [batch, 1, dim]
        # key & value: [batch, max_step_m, dim]
        if query is not None:
            query = tf.expand_dims(query, 1)
        value = memory
        key   = memory

        if mechanism is "additive":
            # affine transform and activate: [batch, max_step_m, att_sz]
            inputs  = key if query is None else [query, key]
            outputs = affine(inputs, attention_sz, 1, bias, activate_func)
            # v^T * tanh(W_q*q+W_m*m)
            weight_v = tf.get_variable(
                    name="weight_v",
                    shape=[attention_sz],
                    dtype=tf.float32,
                    regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4))
            # score: [batch, max_step_m]
            score = tf.reduce_sum(tf.multiply(outputs, weight_v), axis=-1, keepdims=False)

        if mechanism is "dot_product":
            # [batch, 1, dim]*[batch, max_step_m, dim] -> [batch, 1, max_step_m]
            score = tf.matmul(query, key, transpose_b=True)
            # [batch, max_step_m]
            score = tf.squeeze(score, [1])
            if dot_norm:
                score = score / sqrt(key.shape[-1].value)

        # score masking
        if memory_len is not None:
            mask = tf.sequence_mask(memory_len, maxlen=score.shape[-1].value)
            mask = tf.to_float(mask)
            mask_values = -1e8 * (1.0 - mask)
            score = score + mask_values

        # scoring and attention pooling
        # score: [batch, max_step_m]
        score = tf.nn.softmax(score, axis=-1)
        # context: [batch, dim_memory]
        attention_pool = tf.reduce_sum(tf.expand_dims(score, -1)*value, 1)
        print (score, attention_pool)
        #attention_pool = tf.reshape(attention_pool, shape=tf.shape(value))

        return attention_pool, score


def attention_global(memory, query, scope="attention", reuse=None, memory_len=None,
        mechanism="additive", attention_sz=100, bias=False, activate_func=tf.tanh,
        dot_norm=True):
    """Attention with global query.
    Support memory and query are the same shape, i.e. many-query and many-memory.

    Args:
        memory: key-value embeddings, expected to be [batch, max_step_memory, depth] 
        query: query embeddings that multiply scores, [batch, max_step_query, depth]
        additive: additive attention, f_att(q, M) = v^T * tanh(W_m*M+W_q*q)
        dot_product: dot_product attention, f_att(q, m_i) = q^T*m_i
        memory_len: do not support in this function
        norm_dot: normalize over dot_product score, f_att()/sqrt(dim_key) 
        attention_sz, bias, activate_func: for additive attention
    Returns:
        context: context vector
        score: softmax scores

    """
    with tf.variable_scope(scope, reuse=reuse):
        query = query
        value = memory
        key   = memory

        if additive is True:
            # reshape tensors for additive attention
            # query: [batch, max_step_q, 1, dim] ; key: [batch, 1, max_step_m, dim]
            query = tf.expand_dims(query, 2)
            key   = tf.expand_dims(memory, 1)
            # affine transform: [batch, max_step_q, max_step_m, att_sz]
            outputs = affine([query, key], attention_sz, 1, bias, activate_func)

            # output: [batch, max_step_q, max_step_m]
            atten_cont_vec = tf.get_variable(
                    name="atten_cont_vec",
                    shape=[attention_sz],
                    dtype=tf.float32,
                    regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4))
            score = tf.reduce_sum(tf.multiply(outputs, atten_cont_vec), axis=-1, keepdims=False)

        if dot_product is True:
            # [batch, max_step_q, dim]*[batch, max_step_m, dim] -> [batch, max_step_q, max_step_m]
            score = tf.matmul(query, key, transpose_b=True)
            if dot_norm:
                score = score / sqrt(key.shape[-1].value)

        # score masking
        if memory_len is not None:
            mask = tf.sequence_mask(memory_len, maxlen=score.shape[-1].value)
            mask = tf.to_float(mask)
            mask_values = -1e8 * (1.0 - mask)
            mask_values = tf.expand_dims(mask_values, 1)
            score = score + mask_values

        # scoring and attention pooling
        # score: [batch, max_step_q, max_step_m]
        score = tf.nn.softmax(score, axis=-1)
        # context: [batch, max_step_q, dim_memory]
        attention_pool = tf.matmul(score, value)
        attention_pool = tf.reshape(attention_pool, shape=tf.shape(value))

        return attention_pool, score


def highway(inputs, dim, scope="highway_layer"):
    """Highway Mechanism

    """
    with tf.variable_scope(scope):
        orig_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, shape=[orig_shape[0]*orig_shape[1], orig_shape[-1]])

        W_t = tf.get_variable("W_t", dtype=tf.float32, shape=[dim, dim])
        b_t = tf.get_variable("b_t", dtype=tf.float32, shape=[dim], initializer=tf.zeros_initializer())
        trans = tf.nn.xw_plus_b(x=inputs, weights=W_t, biases=b_t)
        trans = tf.nn.relu(trans)

        W_g = tf.get_variable("W_g", dtype=tf.float32, shape=[dim, dim])
        b_g = tf.get_variable("b_g", dtype=tf.float32, shape=[dim], initializer=tf.zeros_initializer())
        gate = tf.nn.xw_plus_b(x=inputs, weights=W_g, biases=b_g)
        gate = tf.nn.sigmoid(gate)

        outputs = gate * trans + (1 - gate) * inputs
        output_tensor = tf.reshape(outputs, shape=orig_shape)

        return output_tensor


