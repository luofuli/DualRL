"""Define constant values used thoughout the project."""

PADDING_TOKEN = "<blank>"
START_OF_SENTENCE_TOKEN = "<s>"
END_OF_SENTENCE_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"

PADDING_ID = 0
START_OF_SENTENCE_ID = 1
END_OF_SENTENCE_ID = 2
NUM_OOV_BUCKETS = 1  # The number of additional unknown tokens <unk>.

INPUT_IDS = "input_ids"
INPUT_LENGTH = "input_length"
LABEL_IDS_IN = "label_ids_in"
LABEL_IDS_OUT = "label_ids_out"
LABEL_LENGTH = "label_length"

LM_VAR_SCOPE = "LM"
NMT_VAR_SCOPE = "NMT"
CLS_VAR_SCOPE = "CLS"

REWARD = "reward"

# Names of decode type
RANDOM = "random"
GREEDY = "greedy"
BEAM = "beam"

# Standard names for model modes (make sure same to tf.estimator.ModeKeys.TRAIN).
TRAIN = 'train'
DUAL_TRAIN = 'dual_train'
EVAL = 'eval'
INFER = 'infer'

