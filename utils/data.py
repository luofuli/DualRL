import tensorflow as tf
from utils import constants

def load_paired_dataset(input_files,
                        translated_files,
                        input_vocab,
                        translated_vocab,
                        batch_size=32,
                        min_seq_len=5,
                        num_buckets=4):
    """Returns an iterator over the training data."""

    def _make_dataset(text_file, vocab):
        dataset = tf.data.TextLineDataset(text_file)
        dataset = dataset.map(lambda x: tf.string_split([x]).values)
        dataset = dataset.map(vocab.lookup)
        return dataset

    def _key_func(x):
        bucket_width = 6
        bucket_id = x["length"] // bucket_width
        bucket_id = tf.minimum(bucket_id, num_buckets)
        return tf.to_int64(bucket_id)

    def _reduce_func(unused_key, dataset):
        return dataset.padded_batch(batch_size,
                                    padded_shapes={
                                        "ids": [None],
                                        "length": [],
                                        "trans_ids": [None],
                                        "trans_ids_in": [None],
                                        "trans_ids_out": [None],
                                        "trans_length": []},
                                    )

    bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=tf.int64)
    eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=tf.int64)

    # Make a dataset from the input and translated file.
    input_dataset = _make_dataset(input_files, input_vocab)
    translated_dataset = _make_dataset(translated_files, translated_vocab)
    dataset = tf.data.Dataset.zip((input_dataset, translated_dataset))
    dataset = dataset.shuffle(100000)

    # Define the input format.
    dataset = dataset.map(lambda x, y: {
        "ids": x,
        "length": tf.shape(x)[0],
        "trans_ids": y,
        "trans_ids_in": tf.concat([bos, y], axis=0),
        "trans_ids_out": tf.concat([y, eos], axis=0),
        "trans_length": tf.shape(y)[0]})

    # Filter out invalid examples.
    dataset = dataset.filter(lambda x: tf.greater(x["length"], min_seq_len - 1))

    # Batch the dataset using a bucketing strategy.
    dataset = dataset.apply(tf.contrib.data.group_by_window(
        _key_func,
        _reduce_func,
        window_size=batch_size))
    return dataset.make_initializable_iterator()


def load_dataset(input_files,
                 input_vocab,
                 mode,
                 batch_size=32,
                 min_seq_len=5,
                 num_buckets=4):
    """Returns an iterator over the training data."""
    def _make_dataset(text_files, vocab):
        dataset = tf.data.TextLineDataset(text_files)
        dataset = dataset.map(lambda x: tf.string_split([x]).values)
        dataset = dataset.map(vocab.lookup)
        return dataset

    def _key_func(x):
        if mode == constants.TRAIN:
            bucket_width = 6
            bucket_id = x["length"] // bucket_width
            bucket_id = tf.minimum(bucket_id, num_buckets)
            return tf.to_int64(bucket_id)
        else:
            return 0

    def _reduce_func(unused_key, dataset):
        return dataset.padded_batch(batch_size,
                                    padded_shapes={
                                        "ids": [None],
                                        "ids_in": [None],
                                        "ids_out": [None],
                                        "ids_in_out": [None],
                                        "length": [], },
                                    )

    bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=tf.int64)
    eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=tf.int64)

    # Make a dataset from the input and translated file.
    input_dataset = _make_dataset(input_files, input_vocab)
    dataset = tf.data.Dataset.zip(input_dataset)
    if mode == constants.TRAIN:
        dataset = dataset.shuffle(200000)

    # Define the input format.
    dataset = dataset.map(lambda x: {
        "ids": x,
        "ids_in": tf.concat([bos, x], axis=0),
        "ids_out": tf.concat([x, eos], axis=0),
        "ids_in_out": tf.concat([bos, x, eos], axis=0),
        "length": tf.shape(x)[0]})

    # Filter out invalid examples.
    if mode == constants.TRAIN:
        dataset = dataset.filter(lambda x: tf.greater(x["length"], min_seq_len - 1))

    # Batch the dataset using a bucketing strategy.
    dataset = dataset.apply(tf.contrib.data.group_by_window(
        _key_func,
        _reduce_func,
        window_size=batch_size))
    return dataset.make_initializable_iterator()
