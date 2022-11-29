import collections
import os
import random
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig, BasicTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
import transformers.utils
import utils
import evaluate as evaluate_helper
from model import *
import Constants
from Constants import *
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

VERBOSITY = True

if not VERBOSITY:
    transformers.utils.logging.set_verbosity_error()
    logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
    logging.disable(logging.INFO) 


eval_file='./data/eval.json'
DEVICE="cuda:0"
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
device = torch.device((DEVICE) if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

# f1_file = open('f_scores.txt', 'w')
# f1_file.write('F1 Scores \n')
# f1 = open('logits.txt', 'w')
# f1.write('logits \n')

def read_examples(input_file, is_training):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if utils.is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    if (answer_offset + answer_length - 1) > (len(char_to_word_offset)-1):
                        end_position = char_to_word_offset[-1]
                    else:
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(utils.whitespace_tokenize(orig_answer_text))
                    # if actual_text.find(cleaned_answer_text) == -1:
                    #     print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                    #     continue
                current_example = QAPair(qas_id=qas_id, question_text=question_text, doc_tokens=doc_tokens, orig_answer_text=orig_answer_text, start_position=start_position, end_position=end_position, is_impossible=is_impossible)
                examples.append(current_example)
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = utils.improve_answer_span(all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                                                                                example.orig_answer_text)

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3  # The -3 accounts for [CLS], [SEP] and [SEP]
        # Create chunks if context document exceeds doc_stride
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = utils.check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            start_position = None
            end_position = None
            if is_training:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            features.append(QAFeatures(unique_id=unique_id, example_index=example_index, doc_span_index=doc_span_index,
                                            tokens=tokens, token_to_orig_map=token_to_orig_map, token_is_max_context=token_is_max_context,
                                            input_ids=input_ids, input_mask=input_mask,segment_ids=segment_ids,
                                            start_position=start_position, end_position=end_position, is_impossible=example.is_impossible))
            unique_id += 1
    return features

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging, null_score_diff_threshold):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = utils.get_best_indexes(result.start_logits, n_best_size)
            end_indexes = utils.get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(PrelimPrediction(feature_index=feature_index, start_index=start_index,
                                                                end_index=end_index,
                                                                start_logit=result.start_logits[start_index],
                                                                end_logit=result.end_logits[end_index]))
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
        # nbest = prelim_predictions
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0: 
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)
                # De-tokenize
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = utils.get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                nbest.append(
                    NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit)) 
        
        assert len(nbest) >= 1
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            # output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)
        assert len(nbest_json) >= 1
        all_predictions[example.qas_id] = nbest_json[0]["text"]
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

def prep_eval_features( eval_file,tokenizer, max_seq_length, doc_stride,max_query_length):
    eval_examples = read_examples(input_file=eval_file, is_training=False)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

    return eval_examples,eval_features, eval_data

def evaluate(model, device, eval_examples,eval_features,eval_data,output_dir,ep,step,predict_batch_size=8):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=predict_batch_size)
    model.eval()
    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=(not VERBOSITY)):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))
    output_prediction_file = os.path.join(output_dir, str(ep)+"_"+str(step)+"predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")
    write_predictions(eval_examples, eval_features, all_results,
                      50, 50,True, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False, 0.0)
    return all_results

def run_model(train_file, predict_file=None, do_train=True, do_predict=False):
    output_dir = f'./output_dir/'
    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
        raise ValueError("For Training -- Output directory already exists and is not empty. Please empty it to save new model weights")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Hyper-parameters
    max_seq_length = Constants.max_seq_length
    doc_stride = Constants.doc_stride
    max_query_length = Constants.max_query_length # For Questions
    train_batch_size = Constants.train_batch_size 
    predict_batch_size = Constants.predict_batch_size 
    learning_rate = Constants.learning_rate
    num_train_epochs = Constants.num_train_epochs
    warmup_proportion  = Constants.warmup_proportion
    n_best_size = Constants.n_best_size
    max_answer_length = Constants.max_answer_length
    verbose_logging = Constants.verbose_logging
    seed = Constants.seed # random seed for initialization
    gradient_accumulation_steps = Constants.gradient_accumulation_steps # Number of updates steps to accumulate before performing a backward/update pass.
    do_lower_case = True
    null_score_diff_threshold = 0.0
    assert gradient_accumulation_steps >= 1
    train_batch_size = train_batch_size // gradient_accumulation_steps

    # SET SEED VALS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # BERT TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_examples = None
    num_train_optimization_steps = None
    if do_train:
        train_examples = read_examples(input_file=train_file, is_training=True)
        num_train_optimization_steps = int(len(train_examples)/train_batch_size/gradient_accumulation_steps) * num_train_epochs

    # MODEL
    model = ModelA()
    model.to(device)
    print('model initialized')
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=warmup_proportion, num_training_steps=num_train_optimization_steps)
    global_step = 0
    tbx = SummaryWriter(output_dir)
    if do_train:
        train_features = None
        train_features = convert_examples_to_features(examples=train_examples, tokenizer=tokenizer, max_seq_length=max_seq_length, doc_stride=doc_stride,
                                                        max_query_length=max_query_length, is_training=True)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).to(device)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).to(device)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).to(device)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long).to(device)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long).to(device)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
        eval_examples, eval_features, eval_data = prep_eval_features(eval_file, tokenizer, max_seq_length, doc_stride=doc_stride, max_query_length=max_query_length)
        saver = utils.CheckpointSaver(output_dir, max_checkpoints=3, metric_name='F1', maximize_metric=True, log=logger)
        model.train()
        for ep in trange(int(num_train_epochs), desc="Epoch"):
            progress_bar = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(progress_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                if n_gpu > 1:
                    loss = loss.mean() 
                if gradient_accumulation_steps > 1:
                    loss = loss /gradient_accumulation_steps
                loss_val = loss.item()
                progress_bar.set_postfix(Loss=loss_val)
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    max_grad_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    tbx.add_scalar('train/NLL', loss_val, step)
                    tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)
                    optimizer.zero_grad()
                    global_step += 1
                save_iter_num = 100 
                if step != 0 and step % save_iter_num ==0:
                    model.eval()
                    allresults = evaluate(model, device, eval_examples, eval_features, eval_data, output_dir,ep,step, predict_batch_size=predict_batch_size)
                    out_eval = evaluate_helper.main(eval_file, output_dir + '/'+str(ep)+"_"+str(step)+"predictions.json")

                    # Save the best models
                    saver.save(global_step, model, out_eval['f1'], device)
                    # Log to TensorBoard
                    for k, v in out_eval.items():
                        tbx.add_scalar('train/{}'.format(k), v, global_step)
                    model.train()
    if do_train:
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.bert_model.config.to_json_string())
        model = ModelA()
        model.load_state_dict(torch.load(output_model_file, map_location=device))
    else:
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        model = ModelA()
        model.load_state_dict(torch.load(output_model_file, map_location=device))

    model.to(device)

    if do_predict:
        eval_examples = read_examples(input_file=predict_file, is_training=False)
        eval_features = convert_examples_to_features(examples=eval_examples, tokenizer=tokenizer, max_seq_length=max_seq_length,
                                                    doc_stride=doc_stride, max_query_length=max_query_length, is_training=False)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long).to(device)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long).to(device)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long).to(device)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long).to(device)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=predict_batch_size)
        model.eval()
        all_results = []
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=(not VERBOSITY)):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
                # f1.write(f'start = {batch_start_logits}  end = {batch_end_logits} \n')
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                res = RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits)
                all_results.append(res)
        output_prediction_file = os.path.join(output_dir, "predictions.json")
        output_nbest_file = os.path.join(output_dir, "temp.json")
        output_null_log_odds_file = os.path.join(output_dir, "temp1.json")
        write_predictions(eval_examples, eval_features, all_results, n_best_size, max_answer_length, do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, verbose_logging, null_score_diff_threshold)
        # f1.close()
        # f1_file.close()