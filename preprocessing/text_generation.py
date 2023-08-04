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
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MAX_INPUT_LENGTH = 30
MAX_TARGET_LENGTH = 512

"""
Generate n text for each prompt using a pre-train language generation model.

running example:
python train-scripts/generate-text-multiples.py \
 --model Models/ISEAR-NLG-WORDS/bart-base \
 --prompt data-preprocessing/ISEAR-prompts/test.xlsx \
 --overwrite True \
 --prompt_col Prompt \
 --n_examples 5
"""


def create_full_prompts(prompt: str, prompt_text: list[str]) -> list[str]:
    """
    Remove the conditional variables from each the prompt. Further, the prompt
    is added at the beginning to the generated text by the models
    """
    full_prompts = []
    clean_prompt = prompt.split(': ')[1]
    for g_text in prompt_text:
        full_prompts.append(f'{clean_prompt} {g_text}')
    return full_prompts


def generate_texts(text: str, model: TFAutoModelForSeq2SeqLM,
                   tokenizer: AutoTokenizer, n_examples: int) -> list[str]:
    """
    Generate the top n sentences from the given text or prompt
    """

    text_ids = tokenizer(text, return_tensors="tf").input_ids
    predictions = model.generate(text_ids,
                                 temperature=0.4,
                                 num_beams=30,
                                 no_repeat_ngram_size=2,
                                 top_k=5,
                                 top_p=0.8,
                                 num_return_sequences=n_examples)

    decode_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    full_prompt = create_full_prompts(text, decode_text)
    return full_prompt


if __name__ == '__main__':
    # Create log file:
    parser = argparse.ArgumentParser(description='Prompt Generation Bart & T5')
    # arguments
    parser.add_argument("--model", required=True, help='Bart or T5 path',
                        dest="model_checkpoint", type=str)
    parser.add_argument("--prompt", required=True, type=str,
                        dest="prompt_path")
    parser.add_argument("--col_prefix", type=str,dest="col_prefix",
                        default='')
    parser.add_argument('--overwrite', type=bool, dest='over_write',
                        default=False)
    parser.add_argument("--prompt_col", required=True, type=str, dest="p_col")
    parser.add_argument("--n_examples", type=int, default=5, dest="n_examples")

    args = parser.parse_args()
    model_checkpoint = Path(args.model_checkpoint)

    data = pd.read_excel(args.prompt_path)

    # Validate index
    if 'sentence_id' not in data.columns:
        data.insert(0, 'sentence_id', data.index)

    # duplicate each row if they are not already duplicate
    no_duplicated_rows = [id_ for id_, id_df in data.groupby('sentence_id')
                          if len(id_df) != args.n_examples]
    if no_duplicated_rows:
        data = data.loc[data.index.repeat(args.n_examples)]

    unique_prompts = data.drop_duplicates(subset=['sentence_id'])[args.p_col].to_list()

    # load model
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    text_accumulator = []
    for text in tqdm(unique_prompts):
        generated_text = generate_texts(text, model, tokenizer, args.n_examples)
        text_accumulator += generated_text

    data[f"{args.col_prefix}{model_checkpoint.name}"] = text_accumulator

    if args.over_write:
        data.to_excel(args.prompt_path, index=False)
    else:
        data.to_excel('output_multiples.xlsx', index=False)
