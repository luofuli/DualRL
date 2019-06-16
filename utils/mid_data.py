import numpy as np
from utils import constants


def process_mid_ids(ids_out, seq_length, min_length, vocab_size):
    if len(np.shape(seq_length)) > 1:
        batch_size, decode_width = np.shape(seq_length)
        ids_out = np.reshape(ids_out, (batch_size * decode_width, -1))  # [batch_size*beam_size, T_s]
        seq_length = np.reshape(seq_length, batch_size * decode_width)

    def padded_to_min_length(ids_out, seq_length, min_length, vocab_size):
        append_count = 0
        batch_ids = ids_out.tolist()
        for i in range(len(batch_ids)):
            end_index = seq_length[i] - 1
            if not isinstance(batch_ids[i], list):
                batch_ids[i] = batch_ids[i].tolist()
            max_j = len(batch_ids[i])
            for k in range(end_index + 1, max_j):  # generated result may have some char after </s>, change to pad_id
                batch_ids[i][k] = constants.PADDING_ID
            for k in range(max_j, min_length+1):  # to ensure len(seq remove </s>) >= min_length
                batch_ids[i].append(constants.PADDING_ID)

            if end_index < 3:  # 3 or 'min_length-2'
                append_count += 1
                for j in range(end_index, min_length):
                    if j < len(batch_ids[i]):
                        batch_ids[i][j] = np.random.choice(vocab_size)
                    else:
                        batch_ids[i].append(np.random.choice(vocab_size))
                batch_ids[i][min_length] = constants.END_OF_SENTENCE_ID
                seq_length[i] = min_length + 1
        return np.array(batch_ids), seq_length

    def add_or_remove_tag(ids_out, seq_length, add_start=False,
                          remove_end=False):  # batch_ids has shape: [batch_size, beam_size, T_s]
        batch_ids = ids_out.tolist()
        for i in range(len(batch_ids)):
            end_index = seq_length[i] - 1
            batch_ids[i] = batch_ids[i][:]
            if add_start:
                end_index += 1
                batch_ids[i] = [constants.START_OF_SENTENCE_ID] + batch_ids[i]  # add <s>
            if remove_end:
                batch_ids[i][end_index] = constants.PADDING_ID  # remove </s>
                batch_ids[i] = batch_ids[i][:-1]  # shorten sequence
        return np.array(batch_ids)

    ids_out, seq_length = padded_to_min_length(ids_out, seq_length, min_length=min_length, vocab_size=vocab_size)

    ids_in_out = add_or_remove_tag(ids_out, seq_length, add_start=True)
    ids_in = add_or_remove_tag(ids_out, seq_length, add_start=True, remove_end=True)
    ids = add_or_remove_tag(ids_out, seq_length, remove_end=True)
    ids_length = np.array(seq_length - 1)
    return ids, ids_in, ids_out, ids_in_out, ids_length
