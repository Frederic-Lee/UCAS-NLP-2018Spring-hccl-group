import os
import tensorflow as tf

from utils.logger import get_logger


class BaseModel(object):
    """ Manager of Data-Collector and Sub-Graph-Model """

    def __init__(self, models=None):
        """
        Args:
            sess: session
            models: each model contain config, data_controller and sub-graph

        """
        self.sess = None
        self.models = models


    def init_session(self, enable_ckpt=False):
        """Init session and restore models params
        Args:
            enable_ckpt: enable checkpoint-restored, else create model with new params
            models: built-up sub-graph models

        """
        if enable_ckpt:
            wake_up_session_FLAG = True
            for model in self.models:
                ckpt = tf.train.get_checkpoint_state(model.config.checkpoint)
                if ckpt:
                    model.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    model.saver.restore(self.sess, ckpt.model_checkpoint_path)
                else:
                    model.logger.info("Created model with new parameters")
                    if wake_up_session_FLAG:
                        self.sess = tf.Session(config=tf.ConfigProto(
                            allow_soft_placement=True))
                        self.sess.run(tf.global_variables_initializer())
                        wake_up_session_FLAG = False
                    model.saver = tf.train.Saver(tf.global_variables())
        else:
            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=sess_config)
            self.sess.run(tf.global_variables_initializer())
            for model in self.models:
                model.logger.info("Created model with new parameters")
                model.saver = tf.train.Saver(tf.global_variables())


    def save_model(self, model):
        """Save Checkpoints"""
        if not os.path.exists(model.config.model_saved):
            os.makedirs(model.config.model_saved)
        model.saver.save(self.sess, model.config.model_saved)


    def load_model(self, model):
        """Load checkpoints into session"""
        model.logger.info("Reloading the latest trained model...")
        model.saver.restore(self.sess, model.config.model_saved)


    def get_scope_variables(scope_name):
        """Get collection of variables in the scope"""
        with tf.variable_scope(scope_name):
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name)


    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_summary(self, dir_output):
        """Defines variables for Tensorboard
        Args:
            dir_output: (string) where the results are written

        """
        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(dir_output, self.sess.graph)


    def train_models(self):
        """Performs training with early stopping and lr exponential decay
        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        n_models = len(self.models)
        trainable_model = [True]*n_models 
        initial_lr = [model.config.lr for model in self.models]
        num_epochs = [model.config.nepoch for model in self.models]
        max_nepoch = max(num_epochs)
        nepoch_no_imprv = [0]*n_models
        best_score = [0.0]*n_models
        # self.add_summary()
        
        # training epoches, each epoch train several models
        for epoch in range(max_nepoch):
            for i, model in enumerate(self.models):
                # stop training a model if early stopped
                if trainable_model[i] is False:
                    continue

                # training the model over full-dataset by batch
                model.train(self.sess, epoch)
                
                # validation on the whole dataset
                model.logger.info("\nEvaluation over the dev dataset...")
                metrics = {}
                metrics = model.predict(self.sess, model.data.dev)
                msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
                model.logger.info(msg)

                # early stopping and saving best parameters
                task='pos'
                if task is 'parse':
                    score = metrics['LAS']
                elif task is 'pos':
                    score = metrics['F1']
                else:
                    score = metrics['F1']
                if score > best_score[i]:
                    nepoch_no_imprv[i] = 0
                    self.save_model(model)
                    best_score[i] = score
                    model.logger.info("- new best score: {:}".format(score))
                else:
                    nepoch_no_imprv[i] += 1
                    if nepoch_no_imprv[i] >= model.config.early_stop_epoches:
                        model.logger.info("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                        trainable_model[i] = False
        self.sess.close()

    def evaluate_models(self):
        for model in self.models:
            self.load_model(model)
            metrics = {}
            metrics = model.predict(self.sess, model.data.test, save=True)
            msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
            model.logger.info(msg)


