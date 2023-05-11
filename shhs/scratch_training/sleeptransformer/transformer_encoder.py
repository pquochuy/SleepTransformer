# https://github.com/Kyubyong/transformer

import tensorflow as tf
from modules import ff, positional_encoding, multihead_attention, noam_scheme

class Transformer_Encoder:
    def __init__(self, d_model, d_ff, num_blocks, num_heads, maxlen, fc_dropout_rate, attention_dropout_rate, smoothing):
        #self.hp = hp
        self.d_model= d_model #512 # hidden dimension of encoder/decoder
        self.d_ff= d_ff #1024 # hidden dimension of feedforward layer
        self.num_blocks= num_blocks# 6 # number of encoder/decoder blocks
        self.num_heads= num_heads#8 # number of attention heads
        self.maxlen= maxlen#20 # maximum sequence length
        self.fc_dropout_rate= fc_dropout_rate#0.3 # dropo#u#t
        self.attention_dropout_rate = attention_dropout_rate  # 0.3 # dropo#u#t
        self.smoothing= smoothing#0.1 # label mothing rate

    def encode(self, x, training=True):
        '''
        Returns
        memory: encoder outputs. (N, seq_len, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # x: (N, seq_len, nfeat)
            # embedding
            x *= self.d_model ** 0.5  # scale

            x += positional_encoding(x, self.maxlen)

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    x = multihead_attention(queries=x,
                                            keys=x,
                                            values=x,
                                            num_heads=self.num_heads,
                                            attention_dropout_rate=self.attention_dropout_rate,
                                            training=training)
                    # feed forward
                    x = ff(x,
                           num_units=[self.d_ff, self.d_model],
                           dropout_rate=self.fc_dropout_rate,
                           training=training)
        memory = x
        return memory
