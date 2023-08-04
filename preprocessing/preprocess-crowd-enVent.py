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
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

"""
Filters the records and columns of the crowd-enVent dataset  that have one of 
the 7 emotions with at least one of the 7 appraisals. The subset is then split 
into the classifiers.csv and cnlg.csv to train the classifiers and CNLG models.

running example:
python preprocessing/preprocess-crowd-enVent.py \
    --enVent_path data/crowd-enVent2022/crowd-enVent_generation.tsv \
    --output data
"""


def appraisal_and_emotion_subset(envent_df: pd.DataFrame) -> pd.DataFrame:
    # crowd enVent appraisals that map the EnISEAR's appraisals and Emotions
    enISEAR_equivalent = ['attention', 'self_responsblt', 'self_control',
                          'chance_control', 'pleasantness', 'effort',
                          'predict_conseq']

    isear_emo_encode = {'joy': 0, 'no-emotion': 1, 'shame': 2, 'fear': 3,
                        'disgust': 4, 'anger': 5, 'sadness': 6, 'guilt': 7}

    subset_cols = ['generated_text', 'emotion'] + enISEAR_equivalent
    app_subset = envent_df[subset_cols]
    emo_app = app_subset[app_subset['emotion'].isin(isear_emo_encode)]
    emo_app['emotion_code'] = emo_app['emotion'].replace(isear_emo_encode)
    return emo_app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Filters the records and 
    columns of the crowd-enVent dataset  that have one of the 7 emotions with 
    at least one of the 7 appraisals. The subset is then split into the 
    classifiers.csv and cnlg.csv to train the classifiers and CNLG models.""")

    # arguments
    parser.add_argument("--enVent_path", required=True, type=str,
                        help='crowd-enVent path', dest="envent_path")

    parser.add_argument("--output", required=True, type=str, dest="output_path")

    parser.add_argument("--cnlg_size", type=float, default=0.8, dest='cnlg_size',
                        help="dataset size to train the CNLG models, the  "
                             "rest is to train the emotion and appraisal "
                             "classifiers")

    args = parser.parse_args()
    print('hola')

    data = pd.read_csv(args.envent_path, sep='\t')
    emo_app_subset = appraisal_and_emotion_subset(data)

    cnlg, classifiers = train_test_split(emo_app_subset,
                                         train_size=args.cnlg_size,
                                         random_state=42)

    # Save datasets
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    cnlg.to_csv(output_path.joinpath('cnlg.csv'), index=False)
    classifiers.to_csv(output_path.joinpath('classifiers.csv'),
                       index=False)
