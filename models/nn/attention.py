import tensorflow as tf
import math

from models.nn.linear import affine


### Positional-Timing-Signal ###
def positional_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4, scope=None):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a
    different frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can
    be experessed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float
    Returns:
        a Tensor the same shape as x.

    """
    with tf.name_scope(scope or "add_timing_signal"):
        length   = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return signal


### Feed-Forward Network ###
def position_ffn(inputs, hidden_sz, output_sz, keep_prob=1.0, scope=None):
    """ Positional Feed-Forward Layer.
    Args:
        hidden_sz: inner hidden-size
        output_sz: default is same as inputs
        keep_prob: dropout over inner affine
    Returns:
        tensor with same shape as inputs

    """
    with tf.variable_scope(scope or "position_ffn_block"):
        hidden = affine(inputs, hidden_sz, bias=True, activate_func=tf.nn.relu, scope="linear_1")
        if keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)
        output = affine(hidden, output_sz, bias=True, activate_func=None, scope="linear_2")

        return output


### Multi-head Attention ###
def _scaled_dot_product_attention(query, key, value, keep_prob=1.0, mask=None):
    """ Scaled-dot-product attention.
    Args:
        query: a Tensor with shape [batch, heads, max_step_q, depth_k]
        key: a Tensor with shape [batch, heads, max_step_kv, depth_k]
        value: a Tensor with shape [batch, heads, max_step_kv, depth_v]
        keep_prob: a floating point number
        mask: memory mask with shape [batch, max_step_kv]
    Returns:
        Tensor with shape: [batch, heads, max_step_q, depth_v]

    """
    with tf.variable_scope("dot_product_attention"):
        # dot-product: [batch, n_head, query_step, key_step]
        logits = tf.matmul(query, key, transpose_b=True)
        # scaled
        logits = logits * (key.shape[-1].value ** -0.5)

        # query masking
        if mask is not None:
            print("scaled-attention-masking, %d"%(key.shape[-1].value))
            mask_values = -1e8 * (1.0 - mask)
            mask_values = tf.expand_dims(tf.expand_dims(mask_values, 1), 1)
            logits = logits + mask_values

        # scoring
        scores = tf.nn.softmax(logits)
        # dropping out the attention links for each of the heads
        if keep_prob < 1.0:
            scores = tf.nn.dropout(scores, keep_prob)

        return tf.matmul(scores, value)


def _split_heads(x, n_head):
    """ Split tensor into n_head.
    Args:
        x: input tensor, shape: [batch, max_step, depth]
    Returns:
        tensor: [batch, n_head, max_step, depth]

    """
    old_shape = x.get_shape().dims
    last_dim  = old_shape[-1]
    new_shape = old_shape[:-1] + [n_head] + [last_dim // n_head if last_dim else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n_head, -1]], 0))
    ret.set_shape(new_shape)

    return tf.transpose(ret, [0, 2, 1, 3])


def _combine_heads(x):
    """Combine seperate heads.
    Args:
        x: [batch, heads, max_step, depth]
    Returns:
        tensor: [batch, max_step, depth]

    """
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().dims
    heads, channel = old_shape[-2:]
    new_shape = old_shape[:-2] + [heads * channel if heads and channel else None]
    x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    x.set_shape(new_shape)

    return x


def multihead_attention(query, memory, key_sz, value_sz, output_sz, n_head=8,
        mask=None, keep_prob=None, scope=None):
    """ Multihead scaled-dot-product attention with input/output transformations.
    Args:
        query:  a Tensor with shape [batch, max_step_q, depth]
        memory: a Tensor with shape [batch, max_len_m, depth]
        key_sz: transform query and key from input_sz to key_sz for all heads
        value_sz: transform value from input_sz to key_sz for all heads
        output_sz: transform multi-head from nhead*value_sz to output_sz
        n_head: dividing total_key_depth and total_value_depth
        keep_prob: inner-attention dropout
    Returns:
        A Tensor.

    """
    if key_sz % n_head != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_sz, n_head))
    if value_sz % n_head != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_sz, n_head))

    with tf.variable_scope(scope or "multihead_attention"):
        # projection: [batch, max_step, total_depth]
        #combined = affine(query, key_sz*2+value_sz, n_split=1, bias=False, scope="qkv_transform")
        #q, k, v  = tf.split(combined, [key_sz, key_sz, value_sz], axis=-1)
        q = affine(query, key_sz, n_split=1, bias=False, scope="query_transform")
        k = affine(memory, key_sz, n_split=1, bias=False, scope="key_transform")
        v = affine(memory, value_sz, n_split=1, bias=False, scope="value_transform")
        # split heads: [batch, n_head, max_step, total_depth//n_head]
        q = _split_heads(q, n_head)
        k = _split_heads(k, n_head)
        v = _split_heads(v, n_head)
        # scaled-dot-product attention
        heads = _scaled_dot_product_attention(q, k, v, keep_prob, mask)
        # combine heads: [batch, max_step_q, total_depth_v]
        heads = _combine_heads(heads)
        # output projection: [batch, max_step_q, output_sz]
        outputs = affine(heads, output_sz, scope="output_transform")

        return outputs


### Layer-Normalization and Residual Blocks ###
def layer_normalization(inputs, epsilon=1e-6, scope=None):
    """Layer-Normalization.
    Returns:
        normalized inputs with original shape.

    """
    with tf.variable_scope(scope or "norm_layer"):
        channel_size = inputs.get_shape().as_list()[-1]

        scale  = tf.get_variable("scale", shape=[channel_size],
                                initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", shape=[channel_size],
                                 initializer=tf.zeros_initializer())

        mean     = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

        return norm_inputs * scale + offset


def norm_residual_block(inputs, residual, keep_prob=1.0, scope=None):
    """Layer-Normalization over Residual Block.
    Returns:
        layer_norm(x + sub-layer(x))

    """
    if keep_prob < 1.0:
        residual = tf.nn.dropout(residual, keep_prob)

    return layer_normalization(inputs + residual, scope=scope)


### Transformer Attentive Encoder ###
def self_attentive_encoder(encoder_inputs, inputs_len, scope=None,
        partitioned=False, multi_branch=False,
        nlayers=6, model_sz=512, residual_keep_prob=1.0,
        pos_nn="ffn", pos_hidden_sz=2048, relu_keep_prob=1.0,
        n_head=8, key_sz=512, value_sz=512, attn_keep_prob=1.0):
    """Transformer Encoder.

    """
    with tf.variable_scope(scope or "attentive_encoder"):
        inputs = encoder_inputs
        if model_sz != inputs.shape[-1].value:
            # raise ValueError("Model-size is not compatitble with inputs-size!")
            inputs = affine(inputs, model_sz, bias=False, scope="input_affine")
        # masking inputs
        seq_mask = tf.sequence_mask(inputs_len, maxlen=inputs.shape[1].value)
        seq_mask = tf.to_float(seq_mask)
        inputs = inputs * tf.expand_dims(seq_mask, axis=-1)
        # inputs bias
        bias = tf.get_variable("inputs_bias", [model_sz])
        inputs = tf.nn.bias_add(inputs, bias)
        # add or concat timing signal
        timing_signal = positional_timing_signal(inputs)
        #timing_signal = timing_signal * tf.expand_dims(seq_mask, axis=-1)
        if partitioned:
            inputs = tf.concat([inputs, timing_signal], axis=-1)
        else:
            inputs = inputs + timing_signal

        # multi-stacks
        for layer in range(nlayers):
            with tf.variable_scope("layer_%d"%(layer)):
                # multi-head attention block
                outputs = multihead_attention(
                    query=inputs,
                    memory=inputs,
                    mask=seq_mask,
                    key_sz=key_sz,
                    value_sz=value_sz,
                    output_sz=model_sz,
                    n_head=n_head,
                    keep_prob=attn_keep_prob
                )
                inputs = norm_residual_block(inputs, outputs, residual_keep_prob,
                        scope="multi-head-attention_norm-layer")

                # feed-forword-network block
                outputs = position_ffn(
                    inputs,
                    hidden_sz=pos_hidden_sz,
                    output_sz=model_sz,
                    keep_prob=relu_keep_prob
                )
                inputs = norm_residual_block(inputs, outputs, residual_keep_prob,
                        scope="position-ffn_norm-layer")
                        
        return inputs
