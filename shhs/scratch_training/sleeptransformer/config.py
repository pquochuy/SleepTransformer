class Config(object):
    def __init__(self):
        # input spectral dimension, e.g. LTEs
        self.ndim = 128
        # number of spectral columns of one PSG epoch
        self.frame_seq_len = 29
        # sequence length
        self.epoch_seq_len = 20
        # number of channels
        self.nchannel = 1
        self.nclass = 5

        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.0001
        self.training_epoch = 10*self.epoch_seq_len
        self.batch_size = 32

        self.frame_attention_size = 64

        self.evaluate_every = 100
        self.checkpoint_every = 100
        self.early_stop_count = 200
        self.num_fold_training_data = 37 # the number of folds to parition the training subjects. To circumvent the memory
                                        # problem when the data is large, only one fold of the data is alternatively loaded at a time.
        self.num_fold_testing_data = 1

        self.frm_d_model = self.ndim*self.nchannel  # hidden dimension of encoder/decoder
        self.frm_d_ff = 1024  # hidden dimension of feedforward layer
        self.frm_num_blocks = 2  # number of encoder/decoder blocks
        self.frm_num_heads = 8  # number of attention heads
        self.frm_maxlen = 29  # maximum sequence length
        self.frm_fc_dropout = 0.1  # 0.3 dropout
        self.frm_attention_dropout = 0.1  # 0.3 dropout
        self.frm_smoothing = 0.0  # label smoothing rate
        #self.warmup_steps = 4000

        self.seq_d_model = self.ndim*self.nchannel  # hidden dimension of encoder/decoder
        self.seq_d_ff = 1024  # hidden dimension of feedforward layer
        self.seq_num_blocks = 2  # number of encoder/decoder blocks
        self.seq_num_heads = 8  # number of attention heads
        self.seq_maxlen = self.epoch_seq_len  # maximum sequence length
        self.seq_fc_dropout = 0.1  # 0.3 dropout
        self.seq_attention_dropout = 0.1  # 0.3 dropout
        self.seq_smoothing = 0.0  # label smoothing rate

        self.fc_hidden_size = 1024
        self.fc_dropout = 0.1
