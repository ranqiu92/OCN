# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
import pickle
import json
import subprocess

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from model import OCN


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, guid, doc_token, question_text, options, answer=None):
        self.guid = guid
        self.doc_token = doc_token
        self.question_text = question_text
        self.options = options
        self.answer = answer


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id, query_len, opt_len, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.query_len = query_len
        self.opt_len = opt_len
        self.guid = guid


def read_race_examples(input_data):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for data in input_data:
        doc_id = data['id']
        doc = data["article"].replace('\\n', '\n')
        doc_token = []
        prev_is_whitespace = True
        for c in doc:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_token.append(c)
                else:
                    doc_token[-1] += c
                prev_is_whitespace = False

        for i, (question, options, answer) in enumerate(zip(data["questions"], data["options"], data["answers"])):
            example = InputExample(
                guid=doc_id + '-%d' % i,
                doc_token=doc_token,
                question_text=question,
                options=options,
                answer=answer)
            examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_doc_len, max_query_len, max_option_len):
    max_seq_len = max_doc_len + max_query_len + max_option_len + 4

    label_list = ["A", "B", "C", "D"]
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for(example_id, example) in enumerate(examples):
        doc_token = tokenizer.tokenize(' '.join(example.doc_token))

        query_token = tokenizer.tokenize(example.question_text)
        query_len = len(query_token)

        query_len_list, opt_len_list = [], []
        all_ids, all_mask, all_segment_ids = [], [], []
        for option_text in example.options:
            tokens = []
            segment_ids = []
            input_mask = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            input_mask.append(1)

            doc_len = min(max_doc_len, len(doc_token))
            tokens = tokens + doc_token[:doc_len] + ["[SEP]"] * (max_doc_len - doc_len + 1)
            segment_ids = segment_ids + [0] * (max_doc_len + 1)
            input_mask = input_mask + [1] * (doc_len + 1) + [0] * (max_doc_len - doc_len)

            option_token = tokenizer.tokenize(option_text)
            option_len = len(option_token)

            real_query_len = min(query_len, max_query_len)
            real_opt_len = min(option_len, max_option_len)
            query_option_token = query_token[: real_query_len] + ["[SEP]"] + option_token[: real_opt_len] + ["[SEP]"]

            real_query_option_len = len(query_option_token)
            tokens = tokens + query_option_token
            segment_ids = segment_ids + [1] * real_query_option_len
            input_mask = input_mask + [1] * real_query_option_len

            query_len_list.append(real_query_len)
            opt_len_list.append(real_opt_len)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_len, "%d v.s. %d" % (len(input_ids), max_seq_len)
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            all_ids.extend(input_ids)
            all_mask.extend(input_mask)
            all_segment_ids.extend(segment_ids)

        label_id = label_map[example.answer]

        features.append(
                InputFeatures(
                        input_ids=all_ids,
                        input_mask=all_mask,
                        segment_ids=all_segment_ids,
                        label_id=label_id,
                        query_len=query_len_list,
                        opt_len=opt_len_list,
                        guid=example.guid))
    return features


def load_data(data_dir, tokenizer, max_doc_len, max_query_len, max_option_len, is_training):
    if is_training:
        subset_list = ['train']
    else:
        subset_list = ['dev', 'test']

    examples, features = {}, {}
    for subset in subset_list:
        level_example_dict = {"high": None, "middle": None}
        level_features_dict = {"high": None, "middle": None}

        for level in ['high', 'middle']:
            subset_dir = os.path.join(data_dir, subset, level)
            file_list = os.listdir(subset_dir)
            file_list = [file for file in file_list if file.endswith('txt')]
            file_list = sorted(file_list)

            alldata = []
            for file in file_list:
                data = json.load(open(os.path.join(subset_dir, file)))
                alldata.append(data)

            level_example_dict[level] = read_race_examples(alldata)
            level_features_dict[level] = convert_examples_to_features(
                level_example_dict[level], tokenizer, max_doc_len, max_query_len, max_option_len)

        examples[subset] = level_example_dict
        features[subset] = level_features_dict

    return examples, features


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)


def evaluation(model, features, batch_size, device, local_rank):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_query_len = torch.tensor([f.query_len for f in features], dtype=torch.long)
    all_opt_len = torch.tensor([f.opt_len for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, \
                                all_query_len, all_opt_len, all_label_ids, all_example_index)
    if local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    prediction = dict()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, query_len, opt_len, label_ids, example_indexes = batch

        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, query_len, opt_len, label_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

        for i, example_idx in enumerate(example_indexes):
            cur_feature = features[example_idx.item()]
            guid = cur_feature.guid
            prediction[guid] = int(np.argmax(logits[i, :]))

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    return eval_loss, eval_accuracy, prediction


def evaluation_on_RACE(model, features, batch_size, device, local_rank):
    eval_acc = {'dev': dict(), 'test': dict()}
    eval_pred = {'dev': dict(), 'test': dict()}
    example_num = {'dev': dict(), 'test': dict()}

    for subset in ['dev', 'test']:
        for level in ["high", "middle"]:
            eval_features = features[subset][level]
            example_num[subset][level] = len(eval_features)

            logger.info("***** Running evaluation on %s level %s set*****" % (level, subset))
            logger.info("  Num examples = %d", len(eval_features))
            logger.info("  Batch size = %d", batch_size)

            _, cur_subset_acc, _ = evaluation(model, eval_features, batch_size, device, local_rank)
            eval_acc[subset][level] = cur_subset_acc

    eval_acc['dev']['all'] = float((eval_acc['dev']['high'] * example_num['dev']['high'] \
            + eval_acc['dev']['middle'] * example_num['dev']['middle'])) \
            / (example_num['dev']['high'] + example_num['dev']['middle'])

    eval_acc['test']['all'] = float((eval_acc['test']['high'] * example_num['test']['high'] \
            + eval_acc['test']['middle'] * example_num['test']['middle'])) \
            / (example_num['test']['high'] + example_num['test']['middle'])

    return eval_acc


def make_metric_log(eval_acc, epoch=None, batch=None, global_step=None):
    metrics = {
        'dev_acc': {
            'RACE-M': eval_acc['dev']['middle'],
            'RACE-H': eval_acc['dev']['high'],
            'RACE': eval_acc['dev']['all']
        },
        'test_acc': {
            'RACE-M': eval_acc['test']['middle'],
            'RACE-H': eval_acc['test']['high'],
            'RACE': eval_acc['test']['all']
        }
    }
    if epoch is not None:
        metrics['epoch'] = epoch
    if batch is not None:
        metrics['batch'] = batch
    if global_step is not None:
        metrics['global_step'] = global_step

    return json.dumps(metrics)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--race_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory of RACE data")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_dir",
                        default="",
                        required=True,
                        type=str,
                        help="The directory of the initial model checkpoint")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_doc_len",
                        default=400,
                        type=int,
                        help="The maximum length of the document")
    parser.add_argument("--max_query_len",
                        default=30,
                        type=int,
                        help="The maximum length of the question")
    parser.add_argument("--max_option_len",
                        default=16,
                        type=int,
                        help="The maximum length of the option")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--train_batch_size",
                        default=12,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=12,
                        type=int,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=None,
                        help="Random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--log_period',
                        type=int,
                        default=50,
                        help="The batch number between two training loss records")
    parser.add_argument('--eval_period',
                        type=int,
                        default=1000,
                        help="The batch number between two evaluations")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    total_batch_size = int(args.train_batch_size)
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    if args.seed is None:
        seed = random.Random(None).randint(1, 100000)
    else:
        seed = args.seed
    logger.info("seed: {} ".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    output_dir = args.output_dir
    if output_dir:
        if os.path.exists(output_dir) and os.listdir(output_dir) and args.do_train:
            raise ValueError("Output directory () already exists and is not empty.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

    tokenizer = BertTokenizer.from_pretrained(args.model_dir, do_lower_case=args.do_lower_case)

    # Load and preprocess data
    if args.do_train:
        train_examples, train_features = load_data(
            args.race_dir, tokenizer, args.max_doc_len, args.max_query_len, args.max_option_len, is_training=True)
        train_examples = train_examples['train']['high'] + train_examples['train']['middle']
        train_features = train_features['train']['high'] + train_features['train']['middle']

        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    if args.do_eval:
        eval_examples, eval_features = load_data(
            args.race_dir, tokenizer, args.max_doc_len, args.max_query_len, args.max_option_len, is_training=False)

    # Prepare model
    model = OCN.from_pretrained(args.model_dir,
                num_labels=4,
                max_doc_len=args.max_doc_len,
                max_query_len=args.max_query_len,
                max_option_len=args.max_option_len)

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        if args.optimize_on_cpu:
            param_optimizer = [param.clone().detach().to('cpu').requires_grad_() \
                                for param in model.parameters()]
        else:
            param_optimizer = list(model.parameters())

        optimizer = BertAdam(param_optimizer,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps,
                             weight_decay=0.01)

    global_step = 0
    best_accuracy = None

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_query_len = torch.tensor([f.query_len for f in train_features], dtype=torch.long)
        all_opt_len = torch.tensor([f.opt_len for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_query_len, all_opt_len, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for epoch in range(int(args.num_train_epochs)):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, query_len, opt_len, label_ids = batch
                loss, _ = model(input_ids, segment_ids, input_mask, query_len, opt_len, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    if global_step % args.log_period == 0:
                        logger.info("Epoch=%-2d batch=%-4d step=%-6d loss=%f" % (epoch, step + 1, global_step, loss.item()))

                    if args.do_eval and global_step % args.eval_period == 0:
                        model.eval()
                        eval_acc = evaluation_on_RACE(model, eval_features, args.eval_batch_size, device, args.local_rank)
                        metric_log = make_metric_log(eval_acc, epoch, step + 1, global_step)
                        logger.info(metric_log)

                        if best_accuracy is None or eval_acc['dev']['all'] > best_accuracy['dev']['all']:
                            best_accuracy = eval_acc.copy()
                            if output_dir:
                                model_to_save = model.module if hasattr(model, 'module') else model
                                torch.save(model_to_save.state_dict(), output_model_file)
                                model_to_save.config.to_json_file(output_config_file)
                                tokenizer.save_vocabulary(output_dir)

                        model.train()

    if args.do_eval:
        model.eval()
        eval_acc = evaluation_on_RACE(model, eval_features, args.eval_batch_size, device, args.local_rank)
        if args.do_train:
            metric_log = make_metric_log(eval_acc, epoch, step + 1, global_step)
            logger.info(metric_log)

        if best_accuracy is None or eval_acc['dev']['all'] > best_accuracy['dev']['all']:
            best_accuracy = eval_acc.copy()
            if args.do_train and output_dir:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), output_model_file)

        final_metric_log = make_metric_log(best_accuracy)
        logger.info('Final evaluation results: %s' % final_metric_log)


if __name__ == "__main__":
    main()

