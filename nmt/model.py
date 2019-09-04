import tensorflow as tf
import opennmt as onmt
from utils import constants
from opennmt.layers.common import embedding_lookup
from utils import optim


class NMT(object):
    """A sequence-to-sequence model."""

    def __init__(self, mode, params, src_vocab_size, tgt_vocab_size,
                 src_emb, tgt_emb, src_vocab_rev, tgt_vocab_rev, direction=''):
        self.name = constants.NMT_VAR_SCOPE + direction
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.params = params
        self.mode = mode
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_vocab_rev = src_vocab_rev
        self.tgt_vocab_rev = tgt_vocab_rev
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = self.params["learning_rate"]
        self.sampling_probability = self.params.get("sampling_probability", 1.0)

        self.input_ids = tf.placeholder(tf.int32, shape=(None, None), name=constants.INPUT_IDS)
        self.input_length = tf.placeholder(tf.int32, shape=(None,), name=constants.INPUT_LENGTH)
        self.target_ids_in = tf.placeholder(tf.int32, shape=(None, None), name=constants.LABEL_IDS_IN)
        self.target_ids_out = tf.placeholder(tf.int32, shape=(None, None), name=constants.LABEL_IDS_OUT)
        self.target_length = tf.placeholder(tf.int32, shape=(None,), name=constants.LABEL_LENGTH)
        self.target_length_in_or_out = self.target_length + 1
        self.reward = tf.placeholder(tf.float32, shape=(None,), name=constants.REWARD)

        encoder_decoder_type = self.params.get("encoder_decoder_type", "bilstm")
        print("Adopt {} as encoder and decoder".format(encoder_decoder_type))
        if encoder_decoder_type.lower() == "bilstm":
            self.encoder = onmt.encoders.BidirectionalRNNEncoder(params["n_layer"], params["encoder_units"])
            if params["decoder_units"] == params["encoder_units"]:
                print("RNN Decoder CopyBridge")
                self.decoder = onmt.decoders.AttentionalRNNDecoder(params["n_layer"], params["decoder_units"],
                                                                   bridge=onmt.layers.CopyBridge())
            else:
                print("RNN Decoder DenseBridge")
                self.decoder = onmt.decoders.AttentionalRNNDecoder(params["n_layer"], params["decoder_units"],
                                                                   bridge=onmt.layers.DenseBridge())
        elif encoder_decoder_type.lower() == "transformer":
            # Change to transformer: n_layer is 4 or 6, and encoder_units/decoder_units is 256 or 512
            self.encoder = onmt.encoders.SelfAttentionEncoder(params["n_layer"], params["encoder_units"])
            self.decoder = onmt.decoders.SelfAttentionDecoder(params["n_layer"], params["decoder_units"])
        else:
            raise ValueError("Unrecognized encoder_decoder_type: {}".format(encoder_decoder_type))

        self.logits, self.predictions = self.build()

        if mode != constants.INFER:
            self.loss_per_sequence, self.loss = self.compute_loss()
            self.lr_loss = self.compute_rl_loss()
            if mode == constants.TRAIN:
                with tf.variable_scope('train') as scope:
                    self.train_op = self.train(self.loss)
                    scope.reuse_variables()
                    self.retrain_op = self.train(self.lr_loss)

        # only save NMT vars when dual training
        var_list = [var for var in tf.global_variables() if self.name in var.name]
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=10)  # Must in the end of model define

    def get_variable_initializer(self):
        if self.params["initializer"] == "random_uniform":
            return tf.random_uniform_initializer(1, -1)
        elif self.params["initializer"] == "normal_unit_scaling":
            return tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_AVG", uniform=False)
        elif self.params["initializer"] == "uniform_unit_scaling":
            return tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_AVG", uniform=True)
        else:
            raise ValueError("Unrecognized initializer: {}".format(self.params["initializer"]))

    def build(self):
        with tf.variable_scope(self.name, initializer=self.get_variable_initializer()):
            encoder_outputs_tuple = self.encode()
            logits, predictions = self.decode(encoder_outputs_tuple)
        return logits, predictions

    def encode(self, reuse=None):
        input_ids, input_length = self.input_ids, self.input_length
        input_ = embedding_lookup(self.src_emb, input_ids)
        with tf.variable_scope("encoder", reuse=reuse):
            return self.encoder.encode(input_, sequence_length=input_length, mode=self.mode)

    def decode(self, encoder_outputs_tuple, output_layer=None, reuse=False):
        (encoder_outputs, encoder_state, encoder_sequence_length) = encoder_outputs_tuple
        self.encoder_outputs = encoder_outputs
        input_ids, target_ids_in, target_length_in = self.input_ids, self.target_ids_in, self.target_length_in_or_out

        with tf.variable_scope("decoder", reuse=reuse):
            if output_layer is None:
                output_layer = tf.layers.Dense(self.tgt_vocab_size)
            output_layer.build([None, encoder_outputs.get_shape()[-1]])

            predictions = None
            logits = None
            if self.mode != constants.INFER:
                target_in = embedding_lookup(self.tgt_emb, target_ids_in)
                logits, _, _ = self.decoder.decode(
                    target_in,
                    target_length_in,
                    vocab_size=self.tgt_vocab_size,
                    initial_state=encoder_state,
                    mode=self.mode,
                    memory=encoder_outputs,
                    memory_sequence_length=encoder_sequence_length,
                    output_layer=output_layer)
            else:
                batch_size = tf.shape(encoder_sequence_length)[0]
                maximum_iterations = self.params.get("maximum_iterations", 100)
                start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
                end_token = constants.END_OF_SENTENCE_ID

                decode_type = self.params.get("decode_type", constants.GREEDY)
                decode_width = self.params.get("decode_width", 1)
                if decode_type == constants.RANDOM:
                    print("random decode_width:", decode_width)
                    tile_start_tokens = tf.contrib.seq2seq.tile_batch(start_tokens, multiplier=decode_width)
                    tile_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=decode_width)
                    tile_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=decode_width)
                    tile_encoder_sequence_length = tf.contrib.seq2seq.tile_batch(encoder_sequence_length,
                                                                                 multiplier=decode_width)
                    sampled_ids, _, sampled_length, log_probs, alignment = self.decoder.dynamic_decode(
                        self.tgt_emb,
                        tile_start_tokens,
                        end_token,
                        vocab_size=self.tgt_vocab_size,
                        initial_state=tile_encoder_state,
                        output_layer=output_layer,
                        maximum_iterations=maximum_iterations,
                        mode=self.mode,
                        memory=tile_encoder_outputs,
                        memory_sequence_length=tile_encoder_sequence_length,
                        return_alignment_history=True,
                        sample_from=0,
                        # penalize_previous_words=True  # True for Transformer
                    )
                    sampled_ids = tf.reshape(sampled_ids, (batch_size, decode_width, -1))
                    sampled_length = tf.reshape(sampled_length, (batch_size, decode_width))
                    log_probs = tf.reshape(log_probs, (batch_size, decode_width))
                elif decode_type == constants.BEAM:
                    sampled_ids, _, sampled_length, log_probs, alignment = \
                        self.decoder.dynamic_decode_and_search(
                            self.tgt_emb,
                            start_tokens,
                            end_token,
                            vocab_size=self.tgt_vocab_size,
                            initial_state=encoder_state,
                            output_layer=output_layer,
                            beam_width=decode_width,
                            maximum_iterations=maximum_iterations,
                            mode=self.mode,
                            memory=encoder_outputs,
                            memory_sequence_length=encoder_sequence_length,
                            return_alignment_history=True)
                elif decode_type == constants.GREEDY or decode_width <= 1:
                    sampled_ids, _, sampled_length, log_probs, alignment = self.decoder.dynamic_decode(
                        self.tgt_emb,
                        start_tokens,
                        end_token,
                        vocab_size=self.tgt_vocab_size,
                        initial_state=encoder_state,
                        output_layer=output_layer,
                        maximum_iterations=maximum_iterations,
                        mode=self.mode,
                        memory=encoder_outputs,
                        memory_sequence_length=encoder_sequence_length,
                        return_alignment_history=True)

                target_tokens = self.tgt_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))
                predictions = {
                    "ids": sampled_ids,
                    "tokens": target_tokens,
                    "length": sampled_length,
                    "log_probs": log_probs}
        return logits, predictions

    def compute_loss(self):
        max_time = tf.shape(self.logits)[1]
        weights = tf.sequence_mask(self.target_length_in_or_out, maxlen=max_time, dtype=tf.float32)
        loss_per_token = tf.contrib.seq2seq.sequence_loss(self.logits,
                                                          self.target_ids_out,
                                                          weights,
                                                          average_across_timesteps=False,
                                                          average_across_batch=False)
        loss_per_sequence = tf.reduce_sum(loss_per_token, 1) / (tf.reduce_sum(weights + 1e-12, axis=1))
        mean_loss = tf.reduce_mean(loss_per_sequence)
        tf.summary.scalar("loss", mean_loss)
        return loss_per_sequence, mean_loss

    def train(self, loss):
        vars_list = [var for var in tf.trainable_variables() if self.name in var.name]
        params = self.params
        train_op = optim.optimize(loss, params, trainable_varaibles=vars_list)
        return train_op

    def eval(self):
        return self.compute_loss()

    def infer(self):
        return self.predictions

    def apply_gradients(self, grads, var_list, optimizer=None):
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.apply_gradients(zip(grads, var_list))
        return train_op

    def compute_rl_loss(self):
        rl_loss = self.loss_per_sequence * self.reward
        rl_loss = tf.reduce_mean(rl_loss)
        return rl_loss
