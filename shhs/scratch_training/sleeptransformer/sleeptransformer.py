import tensorflow as tf
from nn_basic_layers import *
from config import Config
from transformer_encoder import Transformer_Encoder

class SleepTransformer(object):
    """
    SeqSleepNet implementation
    """

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config
        # self.config.frame_seq_len+1 because of CLS
        self.input_x = tf.placeholder(tf.float32,[None, self.config.epoch_seq_len, self.config.frame_seq_len, self.config.ndim, self.config.nchannel], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.epoch_seq_len, self.config.nclass], name="input_y")
        self.istraining = tf.placeholder(tf.bool, name='istraining') # idicate training for batch normmalization


        # input for frame-level transformer, self.config.frame_seq_len+1 because of CLS
        frm_trans_X = tf.reshape(self.input_x,[-1, self.config.frame_seq_len, self.config.ndim * self.config.nchannel])
        with tf.variable_scope("frame_transformer"):
            frm_trans_encoder = Transformer_Encoder(d_model=self.config.frm_d_model,
                                                    d_ff=self.config.frm_d_ff,
                                                    num_blocks=self.config.frm_num_blocks, # +1 because of CLS
                                                    num_heads=self.config.frm_num_heads,
                                                    maxlen=self.config.frm_maxlen,
                                                    fc_dropout_rate=self.config.frm_fc_dropout,
                                                    attention_dropout_rate=self.config.frm_attention_dropout,
                                                    smoothing=self.config.frm_smoothing)
            frm_trans_out = frm_trans_encoder.encode(frm_trans_X, training=self.istraining)
            print(frm_trans_out.get_shape())
            #[-1, frame_seq_len+1, d_model] [-1, 29, 128*3]


        with tf.variable_scope("frame_attention_layer"):
            self.attention_out, self.attention_weight = attention(frm_trans_out, self.config.frame_attention_size)
            print(self.attention_out.get_shape())
            # attention_output1 of shape (batchsize*epoch_step, nhidden1*2)

        # unfold the data for sequence processing
        seq_trans_X = tf.reshape(self.attention_out, [-1, self.config.epoch_seq_len, self.config.frm_d_model])
        with tf.variable_scope("seq_transformer"):
            seq_trans_encoder = Transformer_Encoder(d_model=self.config.seq_d_model,
                                                    d_ff=self.config.seq_d_ff,
                                                    num_blocks=self.config.seq_num_blocks,
                                                    num_heads=self.config.seq_num_heads,
                                                    maxlen=self.config.seq_maxlen,
                                                    fc_dropout_rate=self.config.seq_fc_dropout,
                                                    attention_dropout_rate=self.config.seq_attention_dropout,
                                                    smoothing=self.config.seq_smoothing)
            seq_trans_out = seq_trans_encoder.encode(seq_trans_X, training=self.istraining)
            print(seq_trans_out.get_shape())

        self.scores = []
        self.predictions = []
        with tf.variable_scope("output_layer"):
            seq_trans_out = tf.reshape(seq_trans_out, [-1, self.config.seq_d_model])
            fc1 = fc(seq_trans_out, self.config.seq_d_model, self.config.fc_hidden_size, name="fc1", relu=True)
            fc1 = tf.layers.dropout(fc1, rate=self.config.fc_dropout, training=self.istraining)
            fc2 = fc(fc1, self.config.fc_hidden_size, self.config.fc_hidden_size, name="fc2", relu=True)
            fc2 = tf.layers.dropout(fc2, rate=self.config.fc_dropout, training=self.istraining)
            score = fc(fc2, self.config.fc_hidden_size, self.config.nclass, name="output", relu=False)
            pred = tf.argmax(score, 1, name="pred")
            self.scores = tf.reshape(score, [-1, self.config.epoch_seq_len, self.config.nclass])
            self.predictions = tf.reshape(pred, [-1, self.config.epoch_seq_len])

        # calculate sequence cross-entropy output loss
        with tf.name_scope("output-loss"):
            self.output_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=score)
            self.output_loss = tf.reduce_sum(self.output_loss, axis=[0])
            self.output_loss /= self.config.epoch_seq_len # average over sequence length

            # add on regularization
        with tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss

        self.accuracy = []
        # Accuracy at each time index of the input sequence
        with tf.name_scope("accuracy"):
            for i in range(self.config.epoch_seq_len):
                correct_prediction_i = tf.equal(self.predictions[:,i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                accuracy_i = tf.reduce_mean(tf.cast(correct_prediction_i, "float"), name="accuracy-%s" % i)
                self.accuracy.append(accuracy_i)

