import tensorflow as tf

def focal_loss(labels, logits, gamma=2.0, alpha=0.25):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Args:
        labels: ground truth labels, shape of [batch_size]
        logits: model's output, shape of [batch_size, num_cls]
        gamma:
        alpha:
    Return:
        shape of [batch_size]

    """
    # convert labels to one-hot tensor, logits to tensor
    labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    n_class = logits.shape[1]
    onehot_labels = tf.one_hot(labels, n_class)
    
    # softmax
    epsilon = 1.e-9
    logits  = tf.add(logits, epsilon)
    softmax_logits     = tf.nn.softmax(logits)
    log_softmax_logits = tf.nn.log_softmax(logits)
    # focal loss
    cross_entropy = tf.multiply(onehot_labels, -log_softmax_logits)
    weight        = tf.pow(tf.subtract(1., tf.multiply(onehot_labels, softmax_logits)), gamma)
    focal_loss = tf.multiply(alpha, tf.multiply(weight, cross_entropy))
    focal_loss = tf.reduce_sum(focal_loss, axis=1)
    loss       = tf.reduce_sum(cross_entropy, axis=1)

    return focal_loss
