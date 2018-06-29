import os
import numpy as np
import tensorflow as tf

from utils.logger import *
from models.base_model import BaseModel
from models.network import *
from models.decoder import non_projective_greedy, non_projective_edmonds, projective_eisner
from models.nn.rnn import birnn_multilayer
from models.nn.attention import self_attentive_encoder
from scripts.elmo.bilm import BidirectionalLanguageModel, weight_layers


class ParserBuilder(object):
    """ Parser Sub-Model for single device.
    Building model on a device.

    """
    def __init__(self, name, config, data_collector):
        """
        Args:
            config: configure instance for a treebank
            data_collector: vocabs, n_vocabs, embedds, train or dev or test datasets

        """
        self.name   = name
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

        # shape = (batch size, max length of sentence in batch)
        self.xpos_ids = tf.placeholder(tf.int32, shape=[None, None], name="xpos_ids")

        # shape = (batch size, max length of sentence in batch)
        self.head_ids = tf.placeholder(tf.int32, shape=[None, None], name="head_ids")

        # shape = (batch size, max length of sentence in batch)
        self.deprel_ids = tf.placeholder(tf.int32, shape=[None, None], name="deprel_ids")

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
            """ELMo embedding"""
            bilm = BidirectionalLanguageModel(self.config.elmo_options, self.config.elmo_weights)
            elmo_embedd_op = bilm(self.char_ids)
            elmo_embedd = weight_layers('embedd', elmo_embedd_op, l2_coef=0.0)

            return elmo_embedd['weighted_op']

        # 0. gloabl embeddings list
        global_embedd_list = []
        # 0* elmo embeddings
        if self.config.elmoFLAG is True:
            self.elmo_embedd = get_elmo_embedd(self)
            if self.config.reduce_elmo is True:
                self.elmo_embedd = linear_projection(
                        self.elmo_embedd,
                        self.config.reduce_elmo_dim,
                        scope="elmo_embedd_reduce")

        # 1. character embeddings, can be reused
        reuse_subtoken = None
        char_scope = "char"
        if self.config.reuse_subtoken is True or self.config.cross_domain is True:
            reuse_subtoken = tf.AUTO_REUSE
        if self.config.reuse_subtoken is not True and self.config.cross_lingual is True:
            char_scope = self.config.language+"_char"
        with tf.variable_scope(char_scope, reuse=reuse_subtoken):
            if self.config.charFLAG is True:
                char_embedd = get_char_embedd(self)
                global_embedd_list.append(char_embedd)

        # 2. word embeddings, can be reused for the same language
        reuse_token = None
        token_scope = "word"
        if self.config.reuse_token is True:
            reuse_token = tf.AUTO_REUSE
        if self.config.cross_lingual is True:
            token_scope = self.config.language+"_word"
        with tf.variable_scope(token_scope, reuse=reuse_token):
            # random initialized embedd
            if self.config.wngramFLAG is True:
                _word_ngram = tf.get_variable(
                        name="_word_ngram",
                        dtype=tf.float32,
                        shape=[self.data.nword, self.config.dim_wngram])
                word_ngram = tf.nn.embedding_lookup(_word_ngram,
                        self.word_ids, name="word_ngram")
                global_embedd_list.append(word_ngram)
            # pre-trained unsupervised embedd
            if self.config.wembeddFLAG is True:
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
            # elmo: pre-trained supervised embedd
            if self.config.elmoFLAG is True and self.config.elmo_position != 'output':
                global_embedd_list.append(self.elmo_embedd)

        # 3. tagging embeddings
        with tf.variable_scope(self.config.treebank+"_postag", reuse=None):
            if self.config.uposFLAG is True:
                _upos_embedd = tf.get_variable(
                        name="_upos_embedd",
                        dtype=tf.float32,
                        shape=[self.data.nupos, self.config.dim_postag])
                upos_embedd = tf.nn.embedding_lookup(_upos_embedd,
                        self.upos_ids, name="upos_embedd")
                global_embedd_list.append(upos_embedd)
            if self.config.xposFLAG is True:
                _xpos_embedd = tf.get_variable(
                        name="_xpos_embedd",
                        dtype=tf.float32,
                        shape=[self.data.nxpos, self.config.dim_postag])
                xpos_embedd = tf.nn.embedding_lookup(_xpos_embedd,
                        self.xpos_ids, name="xpos_embedd")
                global_embedd_list.append(xpos_embedd)

        # 4. concat embeddings
        self.embeddings = tf.concat(global_embedd_list, axis=-1)

        # 5. dropout
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
        encoder_scope = self.config.treebank+"encoder"
        if self.config.reuse_encoder is True:
            reuse_encoder = tf.AUTO_REUSE
            encoder_scope = "encoder_reused"
        with tf.variable_scope(encoder_scope, reuse=reuse_encoder):
            if self.config.encoder == "birnn":
                # multi-layer birnn
                self.encoder_logits, _ = birnn_multilayer(self.embeddings, self.sequence_lengths, scope="birnn_encoder",
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
        """Affine and Scoring for Parser
        Define self.arc_logits and self.rel_cond_logits.

        """
        with tf.variable_scope(self.config.treebank):
            logits = self.encoder_logits
            ## dim-reduction layer
            if self.config.reduce_encoder is True:
                if self.config.output_keep_prob < 1.0:
                    reduce_dim_keep_prob = 1.0
                else:
                    reduce_dim_keep_prob = self.config.mlp_keep_prob
                logits = linear_projection(self.encoder_logits, self.config.reduce_encoder_dim, scope="reduce_encoder_layer",
                        n_split=1, bias=True, activate_func=None, keep_prob=reduce_dim_keep_prob)

            ## arc & rel projection layer
            if self.config.share_arc_rel is True:
                (head_arc, dep_arc) = linear_projection(logits, self.config.share_arc_rel_size, scope="arc_rel_layer",
                        n_split=2, bias=True, activate_func=tf.nn.leaky_relu, keep_prob=self.config.mlp_keep_prob)
                head_rel = head_arc
                dep_rel  = dep_arc
            else:
                (head_arc, dep_arc) = linear_projection(logits, self.config.arc_size, scope="arc_layer",
                        n_split=2, bias=True, activate_func=tf.nn.leaky_relu, keep_prob=self.config.mlp_keep_prob)
                (head_rel, dep_rel) = linear_projection(logits, self.config.rel_size, scope="rel_layer",
                        n_split=2, bias=True, activate_func=tf.nn.leaky_relu, keep_prob=self.config.mlp_keep_prob)

            ## arc & rel scoring-biaffine layer
            # logits shape: [batch, max_step_dep, max_step_head]
            arc_logits = biaffine_projection(dep_arc, head_arc, output_sz=1, scope="arc_score",
                    bias1=True, bias2=False, activate_func=None, keep_prob=self.config.mlp_keep_prob)
            self.arc_logits = arc_logits
            self.arc_probs  = tf.nn.softmax(arc_logits)
            self.arc_preds  = tf.to_int32(tf.argmax(self.arc_probs, axis=-1))

            # logits shape: [batch, max_step_dep, ndeprel, max_step_head]
            rel_logits = biaffine_projection(dep_rel, head_rel, output_sz=self.data.ndeprel, scope="rel_score",
                    bias1=True, bias2=True, activate_func=None, keep_prob=self.config.mlp_keep_prob)
            # one-hot of arc-pred shape: [batch, max_step_dep, max_step_head, 1]
            arc_one_hot = tf.one_hot(self.arc_preds, tf.shape(arc_logits)[-1])
            arc_one_hot = tf.expand_dims(arc_one_hot, axis=-1)
            # conditional logits shape: [batch, max_step_dep, ndeprel]
            rel_cond_logits = tf.matmul(rel_logits, arc_one_hot)
            rel_cond_logits = tf.squeeze(rel_cond_logits, axis=-1)
            self.rel_logits = rel_logits
            self.rel_cond_logits = rel_cond_logits
            # score shape: [batch, max_step_dep, ndeprel, max_step_head]
            self.rel_probs = tf.nn.softmax(rel_logits, axis=2)
            # prediction shape: [batch, max_step_dep]
            self.rel_preds = tf.to_int32(tf.argmax(rel_cond_logits, axis=-1))


    def decode_mst(self, arc_probs, rel_probs, seq_lens):
        """Projective and NOn-projective MST-Decoders
        Args:
            arc_probs: np.array returned from session, shape: [batch, max_step_dep, max_step_head]
            rel_probs: shape: [batch, max_step_dep, ndeprel, max_step_head]
            seq_lens: np.array returned from session, shape: [batch]
        Returns:
            arc_preds: list of arc predictions, batch*seq_len
            rel_preds: list of rel predictions

        """
        arc_preds = []
        rel_preds = []
        for arc_prob, rel_prob, seq_len in zip(arc_probs, rel_probs, seq_lens):
            # arc-prediction with mst algorithm
            arc_prob = arc_prob[:seq_len][:,:seq_len]
            if self.config.decode == "projective-eisner":
                arc_pred = projective_eisner(arc_prob)
            elif self.config.decode == "non-projective-greedy":
                arc_pred = non_projective_greedy(arc_prob)
            else:
                arc_pred = non_projective_edmonds(arc_prob)
            arc_preds.append(arc_pred[1:seq_len])
            # rel-prediction condicted on arc-pred
            arc_pred_one_hot = np.zeros([rel_prob.shape[0], rel_prob.shape[2]])
            arc_pred_one_hot[np.arange(len(arc_pred)), arc_pred] = 1.
            rel_pred = np.argmax(np.einsum('nrb,nb->nr', rel_prob, arc_pred_one_hot), axis=1)
            rel_preds.append(rel_pred[1:seq_len])

        return arc_preds, rel_preds


    def add_loss_op(self):
        """Loss Funtion
        Define self.loss

        """
        if self.config.objective is 'hinge_loss':
            raise ValueError("Not implement yet! Please choose cross_entropy")
        elif self.config.objective is 'cross_entropy':
            arc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.arc_logits, labels=self.head_ids)
            rel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.rel_cond_logits, labels=self.deprel_ids)
            mask = tf.sequence_mask(self.sequence_lengths)
            arc_loss = tf.boolean_mask(arc_loss, mask)
            rel_loss = tf.boolean_mask(rel_loss, mask)
            self.loss = tf.reduce_mean(arc_loss) + tf.reduce_mean(rel_loss)
        else:
            raise ValueError("Not implement objective %s"%(self.config.objective))

        if self.config.l2_regular:
            loss_penalty = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.add_n([self.loss] + loss_penalty)


class Parser(object):
    """ Parser Controller
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


    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list is over
            individual gradients. The inner list is over the gradient calculation for each tower.
            shape like this: ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        Returns:
            List of pairs of averaged (gradient, variable).

        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            print(grad_and_vars)
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            print(grad)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So just return the first tower's pointer to the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads


    def build(self):
        """Defines self.train_op that performs an update on a batch

        """
        with tf.device('/cpu:0'):
            # Calculate the gradients for each model tower.
            tower_grads = []
            # Create a variable to count the number of train() calls.
            self.global_step = tf.get_variable("global_step", [], dtype=tf.float32,
                    initializer=tf.constant_initializer(0), trainable=False)

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

        # Build-up sub-models for each device.
        with tf.variable_scope(tf.get_variable_scope()):
            for device_idx in self.config.devices:
                with tf.device("/gpu:%d"%device_idx):
                    with tf.name_scope("tower_%d"%device_idx) as scope:
                        # build-up model
                        sub_model = ParserBuilder("base-parser-%d"%device_idx, self.config, self.data)
                        sub_model.add_placeholders_op()
                        sub_model.add_embeddings_op()
                        sub_model.add_encode_op()
                        sub_model.add_score_op()
                        sub_model.add_loss_op()
                        # Reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()
                        # Add model to collection
                        tf.add_to_collection("sub_models", sub_model)
                        # Compute grads for each device
                        grads, vs    = zip(*optimizer.compute_gradients(sub_model.loss))
                        if self.config.clip > 0.0:
                            grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
                        grad_and_var = zip(grads, vs)
                        tower_grads.append(grad_and_var)

        with tf.device('/cpu:0'):
            # average gradients synchronizely across all towers and setup train-op
            average_grads = self._average_gradients(tower_grads)
            self.train_op = optimizer.apply_gradients(average_grads, global_step=self.global_step)


    def get_feed_dict(self, data_generator, sub_models, lr=1e-3):
        """ Padding batch of data and building a feed dictionary
        Args:
            data_generator: generator of dataset.iter_batch(), return batch of data
                dictionary{sequence_lengths, word_lengths, word, emword, char,
                upos, xpos, head, deprel}
            sub_models: list of parser-builders
        Returns:
            feed_dict: feed dictionary for devices and learning rate
            data_dict_list: data-dict for each device
            data_nodes_list: data-nodes for each device

        """
        # collect built-up models on seperate devices
        feed = {}
        data_dicts_list = []
        data_nodes_list = []

        # build feed dictionary for models on different devices
        for model in sub_models:
            # get batch of data dictionary and conll-nodes
            if self.config.seqlabFLAG:
                data_dict, data_nodes, seqlabs = next(data_generator)
                feed[model.seqlab_ids] = seqlabs
            else:
                data_dict, data_nodes = next(data_generator)
            data_dicts_list.append(data_dict)
            data_nodes_list.append(data_nodes)
            # feed placeholders
            feed[model.sequence_lengths] = data_dict['sequence_lengths']
            feed[model.emword_ids] = data_dict['emword']
            feed[model.word_ids] = data_dict['word']
            feed[model.char_ids] = data_dict['char']
            feed[model.word_lengths] = data_dict['word_lengths']
            feed[model.upos_ids] = data_dict['upos']
            feed[model.xpos_ids] = data_dict['xpos']
            feed[model.head_ids] = data_dict['head']
            feed[model.deprel_ids] = data_dict['deprel']
        # feed learning rate globally
        feed[self.lr] = lr

        return feed, data_dicts_list, data_nodes_list


    def train(self, sess, epoch):
        """Training the full-dataset by batches for a epoch
        Args:
            sess: Session managed by base-model
            epoch: current epoch index

        """
        # logger and progbar stuff for logging
        self.logger.info("Language-{:} Treebank-{:} -- Epoch {:} out of {:}".format(
            self.config.language, self.config.treebank, epoch+1, self.config.nepoch))
        total_batch_sz = self.config.batch_size * len(self.config.devices)
        nbatches = (self.data.train.nseq - 1) // total_batch_sz + 1
        prog     = Progbar(target=nbatches)

        # reset learning rate
        lr =  self.config.lr / (1.0 + epoch*self.config.lr_decay)

        # iterate over the whole dataset
        data_generator = self.data.train.iter_batch(self.config.batch_size, shuffle=False)
        sub_models = tf.get_collection("sub_models")
        for batch_idx in range(nbatches):
            feed_dict, _, _  = self.get_feed_dict(data_generator, sub_models, lr)
            sub_models = tf.get_collection('sub_models')
            #losses       = tf.get_collection('losses')
            #total_loss   = tf.add_n(losses, name='total_losses')
            #average_loss = tf.float32(total_loss / len(self.config.devices))
            _, _, train_loss = sess.run([self.train_op, self.global_step,
                sub_models[0].loss], feed_dict=feed_dict)

            prog.update(batch_idx+1, [("train loss0", train_loss)])


    def predict(self, sess, dataset, save=False, mst=True):
        """Predict the full-dataset by batches
        Args:
            sess: session managed by base model
            save: whether save results in CoNLL format to the file.
            mst: whether use mst for decoding, else use argmax.
        Returns:
            metrics: metrics dictionary

        """
        def _get_feed_dict(data_dict, model, seqlabs=None):
            feed = {}
            feed[model.sequence_lengths] = data_dict['sequence_lengths']
            feed[model.emword_ids] = data_dict['emword']
            feed[model.word_ids] = data_dict['word']
            feed[model.char_ids] = data_dict['char']
            feed[model.word_lengths] = data_dict['word_lengths']
            feed[model.upos_ids] = data_dict['upos']
            feed[model.xpos_ids] = data_dict['xpos']
            feed[model.head_ids] = data_dict['head']
            feed[model.deprel_ids] = data_dict['deprel']
            if seqlabs is not None:
                feed[model.seqlab_ids] = seqlabs
            return feed

        arc_preds_total = []
        rel_preds_total = []
        arc_target_total = []
        rel_target_total = []
        nodes_pred_total = []
        nodes_total = []
        ntokens = 0

        sub_models = tf.get_collection("sub_models")
        model = sub_models[0]
        for batch_idx, batch_data in enumerate(dataset.iter_batch(self.config.batch_size, shuffle=False)):
            if self.config.seqlabFLAG:
                data_dict, data_nodes, seqlabs = batch_data
                feed_dict = _get_feed_dict(data_dict, model, seqlabs)
            else:
                data_dict, data_nodes = batch_data
                feed_dict = _get_feed_dict(data_dict, model)
            if mst is False:
                arc_preds_, rel_preds_ = sess.run([model.arc_preds, model.rel_preds], feed_dict=feed_dict)
                arc_preds_ = arc_preds_.tolist()
                rel_preds_ = rel_preds_.tolist()
                arc_preds = []
                rel_preds = []
                for arc_pred, rel_pred, seq_len in zip(arc_preds_, rel_preds_, data_dict['sequence_lengths']):
                    arc_preds.append(arc_pred[1:seq_len])
                    rel_preds.append(rel_pred[1:seq_len])
            else:
                arc_probs, rel_probs = sess.run([model.arc_probs, model.rel_probs], feed_dict=feed_dict)
                arc_preds, rel_preds = model.decode_mst(arc_probs, rel_probs, data_dict['sequence_lengths'])

            for arc_target, rel_target, seq_len in zip(data_dict['head'], data_dict['deprel'], data_dict['sequence_lengths']):
                arc_target_total.append(arc_target[1:seq_len])
                rel_target_total.append(rel_target[1:seq_len])

            arc_preds_total += arc_preds
            rel_preds_total += rel_preds
            ntokens += (np.sum(data_dict['sequence_lengths'])-len(data_dict['sequence_lengths']))

            if save is True:
                for nodes, heads, rels in zip(data_nodes, arc_preds, rel_preds):
                    nodes_pred = []
                    for node, head, rel in zip(nodes[1:], heads, rels):
                        rel = self.data.vocab_deprel['idx2str'][rel]
                        node.set_head_rel(head, rel)
                        nodes_pred.append(node)
                    nodes_pred_total.append(nodes_pred)
                nodes_total += data_nodes

        metrics = self.evaluate(arc_preds_total, arc_target_total, rel_preds_total, rel_target_total, ntokens)
        if save is True:
            dataset.write_dataset(self.config.pred_results, nodes_pred_total, pred=True)
            dataset.write_dataset(self.config.target_results, nodes_total, pred=False, have_root=True)

        return metrics


    def evaluate(self, arc_preds, arc_targets, rel_preds, rel_targets, ntokens):
        """Evaluation
        Args:
            arc_preds: total arc-predictions
            rel_preds: total rel-predictions
            ntokens: total tokens
        Return:
            metrics_dict: metrics dictionary

        """
        arc_corr = 0
        rel_corr = 0
        corr = 0
        for arc_pred, arc_target, rel_pred, rel_target in zip(arc_preds, arc_targets, rel_preds, rel_targets):
            arc_corr_array = np.equal(arc_pred, arc_target)
            rel_corr_array = np.equal(rel_pred, rel_target)
            corr_array = arc_corr_array & rel_corr_array
            arc_corr += np.sum(arc_corr_array)
            rel_corr += np.sum(arc_corr_array)
            corr += np.sum(corr_array)
        metrics = {
                'UAS':arc_corr/ntokens*100,
                'LAS':corr/ntokens*100
                }

        return metrics
