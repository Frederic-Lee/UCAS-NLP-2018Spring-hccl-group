import numpy as np
import tensorflow as tf
import os

from utils.logger import *
from models.base_model import BaseModel
from models.network import *
from models.nn.rnn import birnn_multilayer
from models.nn.attention import self_attentive_encoder
from scripts.elmo.bilm import BidirectionalLanguageModel, weight_layers


class TaggerBuilder(object):
    """ Tagging Sub-Model for single device.
    Building model on a device.

    """
    def __init__(self, config, data_collector):
        """
        Args:
            config: configure instance for a treebank
            data_collector: vocabs, n_vocabs, embedds, train or dev or test datasets

        """
        self.config = config
        self.data   = data_collector


    def add_placeholders_op(self):
        """Placeholders to computational graph

        """
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")

        # shape = (batch size, max length of sentence in batch)
        self.emword_ids = tf.placeholder(tf.int32, shape=[None, None], name="emword_ids")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.upos_ids = tf.placeholder(tf.int32, shape=[None, None], name="upos_ids")

        # shape = (batch size)
        if self.config.seqlabFLAG:
            self.seqlab_ids = tf.placeholder(tf.int32, shape=[None], name="seqlab_ids")


    def add_embeddings_op(self):
        """Token and Subtoken Embeddings
        Default initializer is uniform[-sqrt(6/(fan_in + fan_out)), +sqrt()]
        Default adopting dropout, pay attention to repetition of dropout.

        Define self.embeddings

        """
        def get_char_embedd(self):
            """Char embedding function"""
            # get char embeddings matrix
            _char_embedd = tf.get_variable(
                    name="_char_embedd",
                    dtype=tf.float32,
                    shape=[self.data.nchar, self.config.dim_char])
            char_embedd_lookup = tf.nn.embedding_lookup(_char_embedd,
                    self.char_ids, name="char_embedd")
            # CharNN Architecture #
            if self.config.char_nn == 'cnn':
                char_embedd = char_cnn(char_embedd_lookup, dim_char=self.config.dim_char,
                        win_len=self.config.char_cnn_kernel_size, num_filters=self.config.char_cnn_nfilter,
                        keep_prob=self.config.char_keep_prob, nlayers=self.config.char_cnn_nlayer)
            elif self.config.char_nn == 'bilstm':
                char_embedd = char_bilstm(char_embedd_lookup, word_len=self.word_lengths,
                        dim_char=self.config.dim_char, cell_sz=self.config.char_rnn_cell_size,
                        nlayers=self.config.char_rnn_nlayer)
            else:
                raise ValueError("Character network %s NOT implemented!" % (self.config.char_nn))

            return char_embedd

        def get_elmo_embedd(self):
            """ELMo Representations"""
            bilm = BidirectionalLanguageModel(self.config.elmo_options, self.config.elmo_weights)
            elmo_embedd_op = bilm(self.char_ids)
            elmo_embedd = weight_layers('embedd', elmo_embedd_op, l2_coef=0.0)

            return elmo_embedd['weighted_op']

        # 0. gloabl embeddings list
        global_embedd_list = []

        # 1. character embeddings, can be reused
        with tf.variable_scope("char", reuse=tf.AUTO_REUSE):
            if self.config.charFLAG is True:
                char_embedd = get_char_embedd(self)
                global_embedd_list.append(char_embedd)

        # 2. word embeddings
        with tf.variable_scope(self.config.language+"-word", reuse=None):
            _word_embedd = tf.get_variable(
                    name="_word_embedd",
                    shape=[self.data.nemword, self.config.dim_wembedd],
                    initializer=tf.constant_initializer(self.data.embeddings),
                    dtype=tf.float32,
                    trainable=True)
            word_embedd = tf.nn.embedding_lookup(_word_embedd,
                    self.emword_ids, name="word_embedd")
            if self.config.reduce_pretrained:
                word_embedd = linear_projection(word_embedd,
                        self.config.reduce_pretrained_dim,
                        scope="word_embedd_reduce")
            global_embedd_list.append(word_embedd)

            if self.config.elmoFLAG is True:
                self.elmo_embedd = get_elmo_embedd(self)
                if self.config.emlo_position != 'output':
                    global_embedd_list.append(self.elmo_embedd)

        # 3. concat embeddings
        self.embeddings = tf.concat(global_embedd_list, axis=-1)

        # 4. dropout
        if self.config.embedd_dropout is 'global':
            self.embeddings = tf.nn.dropout(self.embeddings, self.config.embedd_keep_prob)
        elif self.config.embedd_dropout is 'local':
            embedd_dropout_list = []
            for _embedd in global_embedd_list:
                _embedd_dropout = tf.nn.dropout(_embedd, self.config.embedd_keep_prob)
                embedd_dropout_list.append(_embedd_dropout)
            self.embeddings = tf.concat(embedd_dropout_list, axis=-1)
        else:
            raise Warning("NO Dropout on Embeddings! You set %s." % (self.config.embedd_dropout))


    def add_encode_op(self):
        """Encoder
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.

        Define self.encoder_logits

        """
        reuse_encoder = None
        encoder_scope = self.config.language
        with tf.variable_scope(encoder_scope, reuse=reuse_encoder):
            if self.config.encoder == "birnn":
                # multi-layer birnn
                self.encoder_logits = birnn_multilayer(self.embeddings, self.sequence_lengths, scope="birnn_encoder",
                        nlayers=self.config.birnn_nlayer, rnn_cell=self.config.cell, cell_sz=self.config.cell_size,
                        rnn_dropout=True, vi_dropout=self.config.varianceFLAG, highway=self.config.highwayFLAG,
                        self_attention=self.config.local_attention, input_keep_prob=self.config.input_keep_prob,
                        output_keep_prob=self.config.output_keep_prob, state_keep_prob=self.config.state_keep_prob,
                        attention_mechanism=self.config.attention_type, attention_sz=self.config.attention_size
                        )
            if self.config.encoder == "transformer":
                # transformer or self-attentive encoder
                self.encoder_logits = self_attentive_encoder(self.embeddings, self.sequence_lengths,
                        scope="self_attentive_encoder",
                        nlayers=self.config.transformer_nlayer, n_head=self.config.nhead,
                        model_sz=self.config.model_size, key_sz=self.config.model_size, value_sz=self.config.model_size,
                        pos_hidden_sz=self.config.pos_hidden_size, relu_keep_prob=self.config.relu_keep_prob,
                        residual_keep_prob=self.config.residual_keep_prob, attn_keep_prob=self.config.attention_keep_prob
                        )
            if self.config.elmoFLAG and self.config.elmo_position != "input":
                # concat elmo-embeddings after encoder
                self.encoder_logits = tf.concat([self.encoder_logits, self.elmo_embedd], axis=-1)


    def add_score_op(self):
        """Affine and Scoring for Tagger
        Define self.logits.

        """
        with tf.variable_scope(self.config.language):
            logits = self.encoder_logits
            self.logits = linear_projection(logits, self.data.nupos, scope="scoring",
                    n_split=1, bias=True, activate_func=None, keep_prob=self.config.mlp_keep_prob)


    def decode_crf(self, logits, seq_lens, trans_params):
        """CRF-Decoders
        Args:
            logits: np.array returned from session, shape: [batch, max_step_dep, max_step_head]
            seq_lens: np.array returned from session, shape: [batch]
            trans_params: crf-params
        Returns:
            viterbi_sequences

        """
        viterbi_sequences = []
        for logit, sequence_length in zip(logits, seq_lens):
            logit = logit[:sequence_length] # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences


    def add_loss_op(self):
        """Loss Funtion
        Define self.loss

        """
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.upos_ids, self.sequence_lengths)
        self.trans_params = trans_params # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)

        if self.config.l2_regular:
            loss_penalty = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.add_n([self.loss] + loss_penalty)


class Tagger(object):
    """ Tagger Controller
    Manage building, training and predicting.

    """
    def __init__(self, config, data_collector):
        """
        Args:
            config: configure instance for a treebank
            data_collector: vocabs, n_vocabs, embedds, train or dev or test datasets

        """
        self.config = config
        self.data   = data_collector
        self.logger = get_logger(self.config.log)
        self.saver  = None


    def build(self):
        """Defines self.train_op that performs an update on a batch

        """
        # setup optimizer
        self.lr    = tf.placeholder(tf.float32, shape=[], name="lr")
        optim_name = self.config.optimizer.lower()
        if optim_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.config.beta1,
                    beta2=self.config.beta2, epsilon=self.config.epsilon)
        elif optim_name == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.lr)
        elif optim_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif optim_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9)
        elif optim_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(lr)
        else:
            raise NotImplementedError("Unknown method {}".format(optim_name))

        # build-up model
        model = TaggerBuilder(self.config, self.data)
        model.add_placeholders_op()
        model.add_embeddings_op()
        model.add_encode_op()
        model.add_score_op()
        model.add_loss_op()
        tf.add_to_collection("model", model)

        # compute gradients and update params
        grads, vs    = zip(*optimizer.compute_gradients(model.loss))
        grads, gnorm = tf.clip_by_global_norm(grads, self.config.clip)
        self.train_op = optimizer.apply_gradients(zip(grads, vs))


    def get_feed_dict(self, data_generator, model, lr=1e-3):
        """ Padding batch of data and building a feed dictionary
        Args:
            data_generator: generator of dataset.iter_batch(), return batch of data
                dictionary{sequence_lengths, word_lengths, word, emword, char,
                upos, xpos, head, deprel}
        Returns:
            feed_dict: feed dictionary for devices and learning rate
            data_dict_list: data-dict for each device
            data_nodes_list: data-nodes for each device

        """
        feed = {}
        # get batch of data dictionary and conll-nodes
        if self.config.seqlabFLAG:
            data_dict, data_nodes, seqlabs = next(data_generator)
            feed[model.seqlab_ids] = seqlabs
        else:
            data_dict, data_nodes = next(data_generator)
        # feed placeholders
        feed[model.sequence_lengths] = data_dict['sequence_lengths']
        feed[model.emword_ids] = data_dict['emword']
        feed[model.word_ids] = data_dict['word']
        feed[model.char_ids] = data_dict['char']
        feed[model.word_lengths] = data_dict['word_lengths']
        feed[model.upos_ids] = data_dict['upos']
        # feed learning rate globally
        feed[self.lr] = lr

        return feed, data_dict, data_nodes


    def train(self, sess, epoch):
        """Training the full-dataset by batches for a epoch
        Args:
            sess: Session managed by base-model
            epoch: current epoch index

        """
        # logger and progbar stuff for logging
        self.logger.info("Language-{:} -- Epoch {:} out of {:}".format(
            self.config.language, epoch+1, self.config.nepoch))
        nbatches = (self.data.train.nseq - 1) // self.config.batch_size + 1
        prog     = Progbar(target=nbatches)

        # reset learning rate
        lr =  self.config.lr / (1.0 + epoch*self.config.lr_decay)

        # iterate over the whole dataset
        model = tf.get_collection("model")[0]
        data_generator = self.data.train.iter_batch(self.config.batch_size, shuffle=False)
        for batch_idx in range(nbatches):
            feed_dict, _, _  = self.get_feed_dict(data_generator, model, lr)
            _, train_loss = sess.run([self.train_op, model.loss], feed_dict=feed_dict)
            prog.update(batch_idx+1, [("train loss", train_loss)])


    def predict(self, sess, dataset, save=False):
        """Predict the full-dataset by batches
        Args:
            sess: session managed by base model
        Returns:
            metrics: metrics dictionary

        """
        def _get_feed_dict(data_dict, model, seqlabs=None):
            feed={}
            feed[model.sequence_lengths] = data_dict['sequence_lengths']
            feed[model.emword_ids] = data_dict['emword']
            feed[model.word_ids] = data_dict['word']
            feed[model.char_ids] = data_dict['char']
            feed[model.word_lengths] = data_dict['word_lengths']
            feed[model.upos_ids] = data_dict['upos']
            if seqlabs is not None:
                feed[model.seqlab_ids] = seqlabs
            return feed

        preds_total  = []
        target_total = []
        seqlen_total = []

        model = tf.get_collection("model")[0]
        for batch_idx, batch_data in enumerate(dataset.iter_batch(self.config.batch_size, shuffle=False)):
            if self.config.seqlabFLAG:
                data_dict, data_nodes, seqlabs = batch_data
                feed_dict = _get_feed_dict(data_dict, model, seqlabs)
            else:
                data_dict, data_nodes = batch_data
                feed_dict = _get_feed_dict(data_dict, model)
            logits, trans_params = sess.run([model.logits, model.trans_params], feed_dict=feed_dict)
            slot_preds = model.decode_crf(logits, data_dict['sequence_lengths'], trans_params)

            preds_total  += slot_preds
            target_total += data_dict['upos']
            seqlen_total += data_dict['sequence_lengths']

        metrics = self.evaluate(preds_total, target_total, seqlen_total)

        return metrics


    def evaluate(self, preds, targets, sequence_lengths):
        """Evaluation
        Args:
            preds: total arc-predictions
            targets: total rel-predictions
            seq_lens: total tokens
        Return:
            metrics_dict: metrics dictionary

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0
        for (lab, lab_pred, length) in zip(targets, preds, sequence_lengths):
            lab      = lab[:length]
            lab_pred = lab_pred[:length]
            accs    += [a==b for (a, b) in zip(lab, lab_pred)]

        acc = np.mean(accs)

        metrics = {
                'F1':acc*100
                }

        return metrics
