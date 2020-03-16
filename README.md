# Option Comparison Network for Multi-choice Reading Comprehension

This is the implementation of [Option Comparison Network for Multi-choice Reading Comprehension](https://arxiv.org/abs/1903.03033), and the code is mainly based on the [PyTorch implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

## Introduction

Multiple-choice reading comprehension (MCRC) is the task of selecting the correct answer from multiple options given a question and an article. Existing MCRC models typically either read each option independently or compute a fixed-length representation for each option before comparing them. However, humans typically compare the options at multiple-granularity level before reading the article in detail to make reasoning more efficient.

Mimicking humans, we propose an option comparison network (OCN) for MCRC which compares options at word-level to better identify their correlations to help reasoning. Specially, each option is encoded into a vector sequence using a skimmer to retain fine-grained information as much as possible. An attention mechanism is leveraged to compare these sequences vector-by-vector to identify more subtle correlations between options, which is potentially valuable for reasoning.

## Prerequisite

To use this source code, you need Python3.6+, a few python3 packages, RACE data and pretrained BERT models. The python dependencies can be installed as follows:

```
pip install -r requirements.txt
```

## Usage

### Training

To train the model, you can use the following command. Here, `[RACE_DIR]` is the directory of the original RACE data you download, whose structure is as follows: 

```
RACE
|--train
   |--high
      |--file1.txt
      |  ...
   |--middle
      |--file2.txt
      |  ...
|--dev
   |--high
      |--file3.txt
      |  ...
   |--middle
      |--file4.txt
      |  ...
|--test
   |--high
      |--file5.txt
      |  ...
   |--middle
      |--file6.txt
      |  ...
```

`MODEL_DIR` is the directory of model files including the BERT parameters, the vocabulary of the tokenizer and the model configuration file, and `[OUTPUT_DIR]` is the directory where you want to save the checkpoint. `[GRAD_ACCUM_NUM]` is an integer which the batch at each step will be divided by, and the gradient will be  accumulated over `[GRAD_ACCUM_NUM]` steps. When determining the value of `[GRAD_ACCUM_NUM]`, the batch size, GPU number and the memory each GPU has should be considered. For 4 NVIDIA Tesla P40 GPUs each of which has 24GB memory, we set `[GRAD_ACCUM_NUM]` to 1 and 6 when using BERT-Base and BERT-Large respectively.

```
# when using BERT-Base
python run.py \
  --do_train \
  --do_eval \
  --do_lower_case \
  --race_dir [RACE_DIR] \
  --model_dir [MODEL_DIR] \
  --max_doc_len 400 \
  --max_query_len 30 \
  --max_option_len 16 \
  --train_batch_size 12 \
  --eval_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps [GRAD_ACCUM_NUM] \
  --output_dir [OUTPUT_DIR]

# when using BERT-Large
python run.py \
  --do_train \
  --do_eval \
  --do_lower_case \
  --race_dir [RACE_DIR] \
  --model_dir [MODEL_DIR] \
  --max_doc_len 400 \
  --max_query_len 30 \
  --max_option_len 16 \
  --train_batch_size 24 \
  --eval_batch_size 24 \
  --learning_rate 1.5e-5 \
  --num_train_epochs 5 \
  --gradient_accumulation_steps [GRAD_ACCUM_NUM] \
  --output_dir [OUTPUT_DIR]
```

### Evaluation

To evaluate the model, the following command can be used. Here, `[MODEL_DIR]` is the directory of the checkpoint you saved when training.

```
python run.py \
  --do_eval \
  --do_lower_case \
  --race_dir [RACE_DIR] \
  --model_dir [MODEL_DIR] \
  --max_doc_len 400 \
  --max_query_len 30 \
  --max_option_len 16 \
  --eval_batch_size 24
```
