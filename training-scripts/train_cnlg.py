"""
Author: Yarik Menchaca Resendiz

MIT License

Copyright (c) 2023 Yarik Menchaca Resendiz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Training script for MetaAI's Bart or Google's T5 natural language
generation models using one of three conditional configurations ( Emotions,
Appraisals, and Emotions and Appraisals)
"""


import argparse
import logging
import random
from pathlib import Path
import pandas as pd
from datasets import Dataset

from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq

from transformers import create_optimizer
import tensorflow as tf

# Replace "0" with the GPU availability
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# running example:
# python train-t5-bart.py -path envent_emotion.csv

def prompt_and_label(text: str, cut_off: int) -> tuple[str, str]:
    """
    Split the text into prompt and labels. The splitting position is given the
    cut_off  (e.g. for a text = "I like to play" with a cut_off = 2. The
    prompt "I like" and the label = "to play")
    """
    words = text.split()
    prompt = text
    label = text
    if cut_off < len(words):
        prompt = ' '.join(words[:cut_off])
        label = ' '.join(words[cut_off:])
    return prompt, label


def emotion_format_data(dataset: pd.DataFrame) -> tuple[list, list]:
    """
    Generate the emotion conditional string “generate [emotion]: [prompt]” for
    every record in the input dataframe
    """
    labels = []
    prefixes = []
    for row in dataset.itertuples():
        prefix = f'generate {row.emotion}: '
        cut_offs = random.sample(range(1, 10), random.randint(2, 5))
        for cut_off in cut_offs:
            prompt, label = prompt_and_label(row.generated_text, cut_off)
            prefixes.append(prefix + prompt)
            labels.append(label)
    return prefixes, labels


def appraisals_format_data(dataset: pd.DataFrame) -> tuple[list, list]:
    """
    Generate the Appraisal conditional string “generate [appraisal]: [prompt]”
    for every record in the input dataframe
    """
    labels = []
    prefixes = []
    for row in dataset.itertuples():
        print(type.row)
        appraisal_set = isear_vector(row)
        prefix = "generate {apprsl}: ".format(apprsl=appraisal_set)
        cut_offs = random.sample(range(1, 10), random.randint(2, 5))
        for cut_off in cut_offs:
            promt, label = prompt_and_label(row.generated_text, cut_off)
            prefixes.append(prefix+promt)
            labels.append(label)
    return prefixes, labels


def emotion_appraisals_format_data(dataset: pd.DataFrame) -> tuple[list, list]:
    """
    Generate the Emotion and Appraisal conditional string
    “generate [emotion] [appraisals]: [prompt]” for every record in the input
    dataframe
    """
    labels = []
    prefixes = []
    for row in dataset.itertuples():
        appraisal_set = isear_vector(row)

        prefix = "generate {emotion} {apprsl}: ".format(emotion=row.emotion,
                                                         apprsl=appraisal_set)

        cut_offs = random.sample(range(1, 10), random.randint(2, 5))
        for cut_off in cut_offs:
            prompt, label = prompt_and_label(row.generated_text, cut_off)
            prefixes.append(prefix + prompt)
            labels.append(label)
    return prefixes, labels


def isear_vector(row: pd.DataFrame.iterrows) -> str:
    """
    Build the appraisal string vector of the form  “{Attention, NoATTE}
    {Responsibility, NoRESP} {Control, NoCONT} {Circumstance, NoCIRC}
    {Pleasantness, NoPLEA} {Effort, NoEffort} {Certanty, NoCERT}”
    """
    vector = ['NoATTE', 'NoRESP', 'NoCONT', 'NoCIRC',
              'NoPLEA', 'NoEFFO', 'NoCERT']
    if row.attention > 3:
        vector[0] = 'Attention'
    if row.self_responsblt > 3:
        vector[1] = 'Responsibility'
    if row.self_control > 3:
        vector[2] = 'Control'
    if row.chance_control > 3:
        vector[3] = 'Circumstance'
    if row.pleasantness > 3:
        vector[4] = 'Pleasantness'
    if row.effort > 3:
        vector[5] = 'Effort'
    if row.predict_conseq > 3:
        vector[6] = 'Certainty'
    return ' '.join(vector)


def preprocess_function(examples, max_input_length=30,
                        max_target_length=512) -> AutoTokenizer:
    """
    Converts a token string (or a sequence of tokens) in a single integer id
    (or a sequence of ids), using the vocabulary.
    """
    model_inputs = tokenizer(examples["prompt"],
                             max_length=max_input_length,
                             truncation=True)
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["generation"],
                           max_length=max_target_length,
                           truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == '__main__':
    # Create log file:
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Fine-tuning T5 & BART')
    # arguments
    parser.add_argument("--path", required=True, help='path to training data',
                        dest="path", type=str)
    parser.add_argument("--model", required=True, type=str,
                        dest="model_checkpoint")
    parser.add_argument("--task", required=True, type=str, dest='task')
    parser.add_argument("--epochs", type=int, dest="epochs", default=10)
    parser.add_argument("--output", type=str, dest="output_dir", default='.')
    parser.add_argument('--entities', type=bool, dest='ent', default=False)

    args = parser.parse_args()

    assert args.task in ['emotions', 'emotions_appraisals', 'appraisals'], 'not supported task '


    logging.info('Reading file')
    dataset = pd.read_csv(Path(args.path))
    add_entities = args.ent

    logging.info('Pre-processing file')

    if args.task == 'emotions':
        input_text, target_text = emotion_format_data(dataset, add_entities)
    elif args.task == 'emotions_appraisals':
        input_text, target_text = emotion_appraisals_format_data(dataset, add_entities)
    elif args.task == 'appraisals':
        input_text, target_text = appraisals_format_data(dataset, add_entities)
    else:
        raise EOFError('no supported task')

    for x, y in zip(input_text[1100:1105], target_text[1100:1105]):
        print(x, y)
    # hugging face dataset
    hg_dataset = Dataset.from_dict({'prompt': input_text,
                                    'generation': target_text})

    # Load tokenizer & tokenize data
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # Add special tokens
    esp_tokens = ['NoATTE', 'NoRESP', 'NoCONT', 'NoCIRC', 'NoPLEA', 'NoEFFO',
                  'NoCERT', 'ATTE', 'RESP', 'CONT', 'CIRC', 'PLEA', 'EFFO',
                  'CERT']
    tokenizer.add_special_tokens({'additional_special_tokens': esp_tokens})

    tokenized_datasets = hg_dataset.map(preprocess_function, batched=True)

    # Load pretrain model
    logging.info("loading model_checkpoint")
    model = TFAutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    # update tokenizer
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,
                                           return_tensors="tf")
    # Transform HF to TF datasets
    tf_train_dataset = tokenized_datasets.to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=4)

    # set hyper parameters
    num_train_steps = len(tf_train_dataset) * args.epochs
    optimizer, schedule = create_optimizer(init_lr=5.6e-5,
                                           num_warmup_steps=0,
                                           num_train_steps=num_train_steps,
                                           weight_decay_rate=0.01)
    try:
        logging.info('compiling model')
        model.compile(optimizer=optimizer)
    except Exception as e:
        logging.error(e)

    # Train in mixed-precision float16
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # training
    try:
        logging.info('training model')
        model.fit(tf_train_dataset,
                  epochs=args.epochs)
    except Exception as e:
        logging.error(e)

    # save model
    tf_save_directory = Path(args.output_dir).joinpath \
        (f"{args.model_checkpoint}")
    tokenizer.save_pretrained(tf_save_directory)
    model.save_pretrained(tf_save_directory)
