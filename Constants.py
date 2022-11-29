import collections
# constants
d_model         = 96
d_word          = 768
n_head          = 8
dropout         = 0.1
d_k             = d_model // n_head
len_c           = 384
len_q           = 384
max_seq_length = 384
doc_stride = 128
max_query_length = 64 # For Questions

# Hyper-parameters
train_batch_size = 4 # 12 
predict_batch_size = 4 # 8
learning_rate = 3e-5
# learning_rate = 1e-3
num_train_epochs = 10
warmup_proportion  = 0.1
n_best_size = 100
max_answer_length = 100
verbose_logging = False
seed = 42 # random seed for initialization
gradient_accumulation_steps = 1
do_lower_case = True
null_score_diff_threshold = 0.0
RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
PrelimPrediction = collections.namedtuple("PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])

class QAPair(object):
    """
    Question answer Pair
    """
    def __init__(self, qas_id, question_text, doc_tokens, orig_answer_text=None, start_position=None, end_position=None, is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        s = ""
        s += f"qas_id: {self.qas_id}"
        s += f", question_text: {self.question_text}"
        s += f", doc_tokens: [{' '.join(self.doc_tokens)}]"
        if self.start_position:
            s += f", start_position: {self.start_position}"
        if self.start_position:
            s += f", end_position: {self.end_position}"
        if self.start_position:
            s += f", is_impossible: {self.is_impossible}"
        return s

class QAFeatures(object):
    def __init__(self, unique_id, example_index, doc_span_index, tokens,
                 token_to_orig_map, token_is_max_context, input_ids, input_mask, 
                 segment_ids, start_position=None, end_position=None, is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def mask_logits(target, mask):
    return target * (1-mask) + mask * (-1e30)