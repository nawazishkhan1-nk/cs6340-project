import json
import glob
import os
import random
import collections
import math
import os
import queue
import shutil
import torch
from transformers import BasicTokenizer

def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def compute_softmax_probability(scores):
    if not scores:
        return []
    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score
    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x
    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index

def get_best_indexes(logits, n_best_size):
    """Get the best logits from a list """
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans by WordPiec tokenization that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)
def train_eval_split(path_dir):
    all_files = glob.glob(f'{path_dir}/*.story')
    unique_story_ids = set([fn.split('/')[-1].split('.')[0] for fn in all_files])
    random.seed(42)
    train_test_split = 0.85
    train_nums = int(train_test_split * len(unique_story_ids))
    print(f'{train_nums} Training Stories and {len(unique_story_ids) - train_nums} Eval Stories')
    
    train_stories = set(random.sample(unique_story_ids, train_nums))
    test_stories = unique_story_ids - train_stories
    par_dir = path_dir.split('/')
    par_dir = "/".join(par_dir[0:len(par_dir)-1])
    for s_id in train_stories:
        file_types = ['story', 'questions', 'answers']
        for ftype in file_types:
            src_path = f'{path_dir}/{s_id}.{ftype}'
            dest_path = f'{par_dir}/data/train/{s_id}.{ftype}'
            shutil.copyfile(src_path, dest_path)
    for s_id in test_stories:
        file_types = ['story', 'questions', 'answers']
        for ftype in file_types:
            src_path = f'{path_dir}/{s_id}.{ftype}'
            dest_path = f'{par_dir}/data/eval/{s_id}.{ftype}'
            shutil.copy2(src_path, dest_path)
                   
def get_story(story_path):
    with open(story_path) as f:
        all_lines = [line.strip() for line in f.readlines()]
        all_lines = list(filter(lambda x:len(x) > 0, all_lines))
    assert all_lines[0].startswith('HEADLINE: ')
    title = all_lines[0].split('HEADLINE: ')[-1]
    assert all_lines[3] == "TEXT:"
    context = " ".join(all_lines[4:])
    return title, context

def get_question_answer_pairs(path_dir, story_id, context, f_type, save_answers=True, training_mode=True):
    q_file = f'{path_dir}/{story_id}.questions'
    a_file = f'{path_dir}/{story_id}.answers'
    with open(q_file, 'r') as f:
        all_question_lines = [line.strip() for line in f.readlines()]
        
    difficulty_ar = list(map(lambda x: x.split('Difficulty: ')[-1], list(filter(lambda x:len(x) > 0 and x.startswith('Difficulty:'), all_question_lines))))
    question_ids =  list(map(lambda x: x.split('QuestionID: ')[-1], list(filter(lambda x:len(x) > 0 and x.startswith('QuestionID:'), all_question_lines))))
    questions =  list(map(lambda x: x.split('Question: ')[-1], list(filter(lambda x:len(x) > 0 and x.startswith('Question:'), all_question_lines))))
    if os.path.exists(a_file):
        with open(a_file, 'r') as f:
            all_answer_lines = [line.strip() for line in f.readlines()]
            answers =  list(map(lambda x: x.split('Answer: ')[-1], list(filter(lambda x:len(x) > 0 and x.startswith('Answer:'), all_answer_lines))))
    else:
        answers = [[]] * len(questions)
    assert len(answers) == len(question_ids)
    assert len(difficulty_ar) == len(question_ids) == len(questions)
    pairs_ar = []
    for i in range(len(questions)):
        cur_dict = {}
        cur_dict['question'] = questions[i]
        cur_dict['id'] = question_ids[i]
        # cur_dict['is_impossible'] = False if difficulty_ar[i].lower() in ['easy', 'moderate'] else True
        cur_dict['is_impossible'] = False
        ans_ar = []
        if len(answers[i]) > 0:
            all_answers = answers[i].split(' | ')
            for ans in all_answers:
                ans_dict = {}
                ans_dict['text'] = ans
                exact_find = context.lower().find(ans.strip().lower())
                if exact_find == -1:
                    # find in chunks of 4, 3, 2 and so on:
                    res = -1
                    for cs in range(10, 0, -1):
                        ans_temp_ar = ans.split()
                        sz = len(ans_temp_ar)
                        st = 0
                        while (st+cs) < sz:
                            check = " ".join(ans_temp_ar[st:(st+cs)])
                            if context.lower().find(check.lower()) != -1:
                                res = context.lower().find(check.lower())
                                break
                            st += 1
                        if res != -1:
                            break
                    exact_find = res
                ans_dict['answer_start'] = exact_find
                if exact_find == -1 and len(all_answers) > 1 and training_mode:
                    continue
                else:
                    ans_ar.append(ans_dict)
        # if f_type == 'train':
        cur_dict['answers'] = [ans_ar[0]]
        # else:
        #     cur_dict['answers'] = ans_ar
        if ans_ar[0]['answer_start'] == -1 and training_mode:
             # only 4 such cases, remove for sake of simplicity 521 -> 517 questions finally
            continue
        else:
            if not save_answers:
                cur_dict['answers'] = []
            pairs_ar.append(cur_dict)
    return pairs_ar

def create_data_json(path_dir, f_type='train'):
    all_files = glob.glob(f'{path_dir}/*.story')
    unique_story_ids = set([fn.split('/')[-1].split('.')[0] for fn in all_files])

    json_file = {}
    json_data = []
    max_context_words = -1
    for story_id in unique_story_ids:
        print(f'{story_id}')
        story_dict = {}
        story_name, context_line = get_story(f'{path_dir}/{story_id}.story')
        story_dict['title'] = story_name
        questions_dict = {}
        max_context_words = max(max_context_words, len(context_line.split()))
        questions_dict['qas'] = get_question_answer_pairs(path_dir, story_id, context_line, f_type)
        questions_dict['context'] = context_line
        story_dict['paragraphs'] = [questions_dict]
        json_data.append(story_dict)
    json_file['data'] = json_data
    par_dir = path_dir.split('/')
    par_dir = "/".join(par_dir[0:len(par_dir)-1])
    with open(f'{par_dir}/{f_type}.json', 'w') as f:
        json.dump(json_file, f)
    # print(f'Maximum number of words in context is {max_context_words}')

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""
    # Use`pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1
    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

class CheckpointSaver:
    # Ref. -  HuggingFace, used for saving model checkpoints whilst training
    """Class to save and load model checkpoints.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print('Saver will {}imize {}...'
                    .format('max' if maximize_metric else 'min', metric_name))

    def is_best(self, metric_val):
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       'step_{}.pth.tar'.format(step))
        torch.save(ckpt_dict, checkpoint_path)
        self._print('Saved checkpoint: {}'.format(checkpoint_path))

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print('New best checkpoint at step {}...'.format(step))

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print('Removed checkpoint: {}'.format(worst_ckpt))
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

def create_data_json(path_dir, f_type='train'):
    all_files = glob.glob(f'{path_dir}/*.story')
    unique_story_ids = set([fn.split('/')[-1].split('.')[0] for fn in all_files])
    
    json_file = {}
    json_data = []
    max_context_words = -1
    for story_id in unique_story_ids:
        # print(f'{story_id}')
        story_dict = {}
        story_name, context_line = get_story(f'{path_dir}/{story_id}.story')
        story_dict['title'] = story_name
        questions_dict = {}
        max_context_words = max(max_context_words, len(context_line.split()))
        questions_dict['qas'] = get_question_answer_pairs(path_dir, story_id, context_line, f_type)
        questions_dict['context'] = context_line
        story_dict['paragraphs'] = [questions_dict]
        json_data.append(story_dict)
    json_file['data'] = json_data
    par_dir = path_dir.split('/')
    par_dir = "/".join(par_dir[0:len(par_dir)-1])
    with open(f'{par_dir}/{f_type}.json', 'w') as f:
        json.dump(json_file, f)
    # print(f'Maximum number of words in context is {max_context_words}')

def create_data_without_answers(path_dir, f_type='train', story_ids=[]):
    all_files = glob.glob(f'{path_dir}/*.story')
    # unique_story_ids = set([fn.split('/')[-1].split('.')[0] for fn in all_files])
    unique_story_ids = story_ids
    json_file = {}
    json_data = []
    max_context_words = -1
    for story_id in unique_story_ids:
        # print(f'{story_id}')
        story_dict = {}
        story_name, context_line = get_story(f'{path_dir}/{story_id}.story')
        story_dict['title'] = story_name
        questions_dict = {}
        max_context_words = max(max_context_words, len(context_line.split()))
        questions_dict['qas'] = get_question_answer_pairs(path_dir, story_id, context_line, f_type, save_answers=False, training_mode=False)
        questions_dict['context'] = context_line
        story_dict['paragraphs'] = [questions_dict]
        json_data.append(story_dict)
    json_file['data'] = json_data
    par_dir = path_dir.split('/')
    par_dir = "/".join(par_dir[0:len(par_dir)-1])
    with open(f'{par_dir}/{f_type}.json', 'w') as f:
        json.dump(json_file, f)
    # print(f'Maximum number of words in context is {max_context_words}')

# train_eval_split('./devset-official')
# create_data_json('./data/train', 'train')
# create_data_without_answers('./data/train', 'devset_all_check')
# create_data_json('./data/eval', 'eval')
