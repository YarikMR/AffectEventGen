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
"""


import argparse
from pathlib import Path
import os
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import TFAutoModelForSequenceClassification, \
    create_optimizer, AutoTokenizer, DataCollatorWithPadding
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np

"""
Train any Model For Sequence Classifications available on the Hugging face
repository.
running example:

python training-scripts/train_classifiers.py \
 --path data/classifiers.csv \
 --model roberta-base \
 --output models/classifiers\
 --class-columns emotion_code effort \
 --text-column generated_text
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def tokenize_str(datadict: DatasetDict, column_name: str = 'Sentence'):
    """Tokenize all str in a given column, by default is Sentences"""
    return tokenizer(datadict[column_name], truncation=True)


def compute_metrics(classifier: TFAutoModelForSequenceClassification,
                    test_dataset):
    """
    Compute the macro average accuracy, Recall and F1 of a given models over
    tha test dataset.
    """
    predictions = classifier.predict(test_dataset)['logits']
    predictions = np.argmax(predictions, axis=1)
    y_true = list(np.concatenate([y for x, y in tf_test_dataset], axis=0))
    temp_metrics = classification_report(y_true, predictions, output_dict=True)
    return list(temp_metrics['macro avg'].values())


if __name__ == '__main__':
    # Create log file:
    parser = argparse.ArgumentParser(description='Train any transformer '
                                                 'classifier using hugging face'
                                                 ' library')
    # arguments
    parser.add_argument("--path", required=True, help='path to training data',
                        dest="path", type=str)
    parser.add_argument("--model", type=str, dest="model_checkpoint",
                        required=True, help='model checkpoint name (e.g '
                                            'roberta-base)')
    parser.add_argument("--class-columns", required=True, type=str, nargs='+',
                        dest='columns', help='class column names on the '
                                             'training file')
    parser.add_argument("--text-column", required=True, type=str, dest='text',
                        help='Text colum to be classified')
    parser.add_argument("--epochs", type=int, dest="epochs", default=10)
    parser.add_argument("--output", type=str, dest="output_dir", default='.',
                        help='Output directory for the classifiers')
    parser.add_argument('--batch', type=int, default=5, dest='batch_size')
    parser.add_argument('--test_size', type=float, default=0.25,
                        dest='test_size')

    args = parser.parse_args()
    DATASET_PATH = str(Path(args.path))
    output = Path(args.output_dir)

    dataset = load_dataset('csv', data_files=DATASET_PATH, delimiter=',')

    # Rename text column to Sentence
    dataset = dataset.rename_column(args.text, "Sentence")

    train_test = dataset['train'].train_test_split(test_size=args.test_size)
    train_test_valid_dataset = DatasetDict({'train': train_test['train'],
                                            'test': train_test['test']})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # tokenize sentences
    pre_tokenizer_columns = set(train_test_valid_dataset["train"].features)
    encoded_dataset = train_test_valid_dataset.map(tokenize_str, batched=True)
    tokenizer_columns = list(
        set(encoded_dataset["train"].features) - pre_tokenizer_columns)
    print("Columns added by tokenizer:", tokenizer_columns)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            return_tensors="tf")

    # Train classifiers
    metrics = []

    for column in args.columns:
        print('---------------------------')
        print(f'Training {column} classifier')

        tf_train_dataset = encoded_dataset["train"].to_tf_dataset(
            columns=tokenizer_columns,
            label_cols=[column],
            shuffle=True,
            batch_size=16,
            collate_fn=data_collator)

        tf_test_dataset = encoded_dataset["test"].to_tf_dataset(
            columns=tokenizer_columns,
            label_cols=[column],
            shuffle=False,
            batch_size=16,
            collate_fn=data_collator)

        print(type(tf_test_dataset))

        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        num_labels = len(set(dataset['train'][column]))

        model = TFAutoModelForSequenceClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=num_labels)

        batches_per_epoch = len(encoded_dataset["train"]) // args.batch_size
        total_train_steps = int(batches_per_epoch * args.epochs)

        optimizer, schedule = create_optimizer(init_lr=2e-5,
                                               num_train_steps=total_train_steps,
                                               num_warmup_steps=0)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=tf.metrics.SparseCategoricalAccuracy())

        model.fit(tf_train_dataset, epochs=args.epochs)

        # Save model
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output.joinpath(f'{column}'))
        tokenizer.save_pretrained(output.joinpath(f'{column}'))

        # compute metrics
        metrics_model = compute_metrics(model, tf_test_dataset)
        metrics.append([args.model_checkpoint, column] + metrics_model)

        del model
    df_metrics = pd.DataFrame(metrics, columns=['Base model', 'model name',
                                                'Precision', 'Recall',
                                                'f1-score', 'support'])

    df_metrics.to_csv(output.joinpath('metrics.csv'), index=False)
