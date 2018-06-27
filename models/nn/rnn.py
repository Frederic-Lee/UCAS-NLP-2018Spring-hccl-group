import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from models.nn.linear import affine, attention


class AttentionWrapper(rnn.RNNCell):
    def __init__(self, cell, memory, memory_len, 
            mechanism="additive", attention_sz=100, dot_norm=True):
        super(AttentionWrapper, self).__init__(_reuse=None)
        self._cell = cell
        self._memory = memory
        self._memory_len = memory_len
        self._mechanism = mechanism
        self._attention_sz = attention_sz
        self._dot_norm = dot_norm

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        inputs, _ = attention(memory=self._memory, query=inputs, memory_len=self._memory_len,
                attention_sz=self._attention_sz, dot_norm=self._dot_norm)
        output, new_state = self._cell(inputs, state, scope=scope)

        return output, new_state


def birnn_multilayer(inputs, seq_len, scope=None, rnn_cell="lstm", nlayers=3, cell_sz=300, 
        rnn_dropout=True, vi_dropout=True, highway=False, self_attention=False, 
        input_keep_prob=0.67, output_keep_prob=1.0, state_keep_prob=0.67,
        attention_mechanism="additive", attention_sz=100, dot_norm=True):
    """Bi-directional Reccurent Networks for Encoder
    Args:
        inputs: expected to be [batch, max_step, depth]
        seq_len: sequence length of inputs, [batch]
        rnn_cell: lstm is prefered, gru and sru are optional
        nlayers: rnn layer num
        cell_sz: size of the basic cell
        rnn_dropout: normal dropout on rnn-cell
        vi_dropout: variational inference dropout of the RNN-Cell
        highway: highway connetion
        self_attention: self-attention after each rnn-layer
        input_keep_prob: dropout inputs, e.g. 0.67 is better
        output_keep_prob: dropout outputs
        state_keep_prob: dropout hidden_state
        attention_mechanism: additive or dot_product
        attention_sz: additive attention
        dor_norm: dot-product attention
    Returns:
        concatonated embeddings:
            top_recur = [batch, max_step, 2*cell_sz]
            end_recur = [batch, 2*cell_sz]

    """
    def cell_wrapper():
        # forward and backward RNN-Cell
        if rnn_cell == "gru":
            cell = rnn.GRUCell(cell_sz, initializer=tf.orthogonal_initializer, 
                use_peepholes=False, cell_clip=None, proj_clip=None)
        elif rnn_cell == "sru":
            cell = rnn.SRUCell(cell_sz, initializer=tf.orthogonal_initializer, 
                use_peepholes=False, cell_clip=None, proj_clip=None) 
        else:
            cell = rnn.LSTMCell(cell_sz, initializer=tf.orthogonal_initializer, 
                use_peepholes=False, cell_clip=None, proj_clip=None)

        # dropout and highway
        if rnn_dropout is True:
            print ("dropout")
            cell = rnn.DropoutWrapper(cell, input_keep_prob=input_keep_prob,
                    output_keep_prob=output_keep_prob, state_keep_prob=state_keep_prob,
                    variational_recurrent=vi_dropout, input_size=inputs.shape[-1], dtype=tf.float32)
        if highway is True:
            print ("highway")
            cell = rnn.HighwayWrapper(cell, couple_carry_transform_gates=True, carry_bias_init=1.0)
        if self_attention is True:
            print ("attention!!!")
            cell = AttentionWrapper(cell, memory=inputs, memory_len=seq_len, 
                    mechanism=attention_mechanism, attention_sz=attention_sz, dot_norm=dot_norm)
        
        # single forward and backward cells
        return cell

    with tf.variable_scope(scope or "birnn_layers"):
        if nlayers > 1:
            fw_cells = [cell_wrapper() for _ in range(nlayers)]
            bw_cells = [cell_wrapper() for _ in range(nlayers)]
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(fw_cells)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(bw_cells)
            print (fw_cells, bw_cells)
        else:
            cell_fw = cell_wrapper()
            cell_bw = cell_wrapper()

        (output_fw, output_bw), ((_, state_fw), (_, state_bw)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=seq_len, dtype=tf.float32)
        top_recur_concat = tf.concat([output_fw, output_bw], axis=-1)
        end_recur_concat = tf.concat([state_fw, state_bw], axis=-1)

        return top_recur_concat, end_recur_concat
