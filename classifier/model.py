import tensorflow as tf
from utils import constants


class TextCNN(object):

    def __init__(self, mode, params, vocab_size):
        emb_dim = params["emb_dim"]
        filter_sizes = [int(x) for x in params["filter_sizes"].split(",")]
        n_filters = params["n_filters"]

        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.learning_rate = params["learning_rate"]
        self.x = tf.placeholder(tf.int32, [None, None], name="x")   # batch_size * max_len
        self.y = tf.placeholder(tf.float32, [None], name="y")

        embedding = tf.get_variable("embedding", [vocab_size, emb_dim])
        x = tf.nn.embedding_lookup(embedding, self.x)
        self.logits = self.cnn(x, filter_sizes, n_filters, self.dropout, "cnn")
        self.probs = tf.sigmoid(self.logits)
        self.loss_per_sequence = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(self.loss_per_sequence)

        if mode == constants.TRAIN:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # only save CLS vars when dual training
        var_list = [var for var in tf.trainable_variables() if constants.CLS_VAR_SCOPE in var.name]
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=5)  # Must in the end of model define

    @staticmethod
    def cnn(inp, filter_sizes, n_filters, dropout, scope, reuse=False):
        dim = inp.get_shape().as_list()[-1]
        inp = tf.expand_dims(inp, -1)  # [batch, seq_length, dim, 1] == [batch, in_height, in_width, in_channels]

        def leaky_relu(x, alpha=0.01):
            return tf.maximum(alpha * x, x)

        with tf.variable_scope(scope) as vs:
            if reuse:
                vs.reuse_variables()

            outputs = []
            for size in filter_sizes:
                with tf.variable_scope("conv-maxpool-%s" % size):
                    W = tf.get_variable("W", [size, dim, 1, n_filters])
                    b = tf.get_variable("b", [n_filters])
                    conv = tf.nn.conv2d(inp, W,
                                        strides=[1, 1, 1, 1], padding="VALID")
                    h = leaky_relu(conv + b)
                    pooled = tf.reduce_max(h, reduction_indices=1)
                    pooled = tf.reshape(pooled, [-1, n_filters])
                    outputs.append(pooled)
            outputs = tf.concat(outputs, 1)
            outputs = tf.nn.dropout(outputs, dropout)

            with tf.variable_scope("output"):
                W = tf.get_variable("W", [n_filters * len(filter_sizes), 1])
                b = tf.get_variable("b", [1])
                logits = tf.reshape(tf.matmul(outputs, W) + b, [-1])
        return logits