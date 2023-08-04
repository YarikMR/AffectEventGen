
# Affective Natural Language Generation of Event Descriptions through Fine-grained Appraisal Conditions

## Information
This repository contains code and data for the fine-tuning of T5/Bart models to generate emotional event descriptions,
as presented in the [paper](https://arxiv.org/abs/2307.14004). These models are conditioned on Emotions and/or Appraisal Theories.

| Condition              | Trigger-Phrase |  Generated text |
|------------------------|----------------|--------------------|
| Joy                    | I got          | I got a job I really wanted.|
| Guilt & Responsability | I saw          | I saw a homeless person who needed medical attention because I couldn’t afford it |


## Abstract.
Models for affective text generation have shown a remarkable
  progress, but they commonly rely only on basic emotion theories or
  valance/arousal values as conditions. This is appropriate when the
  goal is to create explicit emotion statements ("The kid is
  happy."). Emotions are, however, commonly communicated
  implicitly. For instance, the emotional interpretation of an event
  ("Their dog died.") does often not require an explicit emotion
  statement. In psychology, appraisal theories explain the link
  between a cognitive evaluation of an event and the potentially
  developed emotion. They put the assessment of the situation on the
  spot, for instance regarding the own control or the responsibility
  for what happens. We hypothesize and subsequently show that
  including appraisal variables as conditions in a generation
  framework comes with two advantages. (1) The generation model is
  informed in greater detail about what makes a specific emotion and
  what properties it has. This leads to text
  generation that better fulfills the condition. (2) The variables of
  appraisal allow a user to perform a more fine-grained control of the
  generated text, by stating properties of a situation instead of only
  providing the emotion category.  Our Bart and T5-based experiments
  with 7 emotions (Anger, Disgust, Fear, Guilt, Joy, Sadness, Shame),
  and 7 appraisals (Attention, Responsibility, Control, Circumstance,
  Pleasantness, Effort, Certainty) show that (1) adding appraisals
  during training improves the accurateness of the generated texts by
  10\,pp in F1. Further, (2) the texts with appraisal
  variables are longer and contain more details. This exemplifies the greater
  control for users.


## Quick Start

### Install and activate conda environment  

```shell
conda env create -f env.yml
conda activate Appraisal-CNLG
```

### Dataset & Pre-processing

The crowd-enVent dataset can be downloaded from 
[here](https://www.ims.uni-stuttgart.de/data/appraisalemotion). The dataset 
is preprocessed, sub-selected, and split using the scrip preprocess-crowd-enVent.py

```bash
python preprocessing/preprocess-crowd-enVent.py \
  --enVent_path path_to/crowd-enVent_generation.tsv \
  --output data
```

Extra arguments and script documentation:
```bash
python preprocessing/preprocess-crowd-enVent.py --help
```

The **preprocess-crowd-enVent.py** script generates two csv files: **classifiers.csv** 
and **cnlg.csv** with the following format:

```csv
generated_text,emotion,attention,self_responsblt,self_control,chance_control,pleasantness,effort,predict_conseq,emotion_code
[text],[emotion],[attention_score],[self_responsblt_score],[self_control_score],[chance_control_score],[pleasantness_score],[effort_score], [predict_conseq_score], [emotion_code]
```

### Train Conditional Natural Language Generation Models

The six conditional natural language generation configurations are train with
the scrip **train-cnlg.py** by changing the arguments **--model** ("t5-base" or 
"facebook/bart-base") and **--task** ("emotions", "emotions_appraisals", or 
"appraisals")

```bash
python training-scripts/train_cnlg.py \
  --path="path_to/cnlg.csv" \
  --model="t5-base" \
  --epochs=10 \
  --output="path_to_output_directory" \
  --task='emotions'
```

Extra arguments and script documentation:
```bash
python training-scripts/train_cnlg.py --help
```

### Appraisals and Emotion Classifiers

To train the seven binary appraisal classifiers and the multi-label emotion
classifier use the scrip **train-classifiers.py**. All classifiers can be 
trained in a single time by adding all the column class names to the argument
**--class-columns**.

```bash
python training-scripts/train_classifiers.py \
  --path path_to/train_classifiers.py \
  --model roberta-base \
  --output Models/ISEAR-Emotions \
  --class-columns emotion_code effort \
  --text-column generated_text
```

Extra arguments and script documentation:
```bash
python training-scripts/train_classifiers.py
```

## Conditional Text Generation

The script **text-generation.py** is expecting an Excel file with at least one
column, containing the conditional variables and prompts described on the paper.

Example of the Emotion Prompt set (EP) format:

```Excel
Prompt
generate NoATTE NoRESP NoCONT NoCIRC NoPLEA NoEFFO NoCONS: I felt
generate Attention NoRESP NoCONT NoCIRC NoPLEA NoEFFO NoCONS: When I
```

```bash
python preprocessing/text_generation.py \
  --model path_to_cnlg_model/t5 \
  --prompt path_to_dataset/prompts.xlsx \
  --overwrite True \
  --prompt_col Prompt \
  --n_examples 5 \
  --col_prefix "EA-"
```

Extra arguments and script documentation:
```bash
python preprocessing/text_generation.py
```

## Citation

Yarik Menchaca Resendiz and Roman Klinger. Affective natural language generation of event descriptions through
fine-grained appraisal conditions. In Proceedings of the 16th International Conference on Natural Language Generation,
Prague, Czech Republic, September 2023. Association for Computational Linguistics. accepted.

### BibText

```bibtex
@inproceedings{MenchacaResendiz2023,
  title = {Affective Natural Language Generation of Event
                  Descriptions through Fine-grained Appraisal
                  Conditions},
  author = {Menchaca Resendiz, Yarik and Klinger, Roman},
  booktitle = {Proceedings of the 16th International Conference on
                  Natural Language Generation},
  month = sep,
  year = {2023},
  address = {Prague, Czech Republic},
  publisher = {Association for Computational Linguistics},
  note = {accepted},
  url = {https://arxiv.org/abs/2307.14004},
  pdf = {https://www.romanklinger.de/publications/MenchacaResendiz_Klinger_INLG2023.pdf}
}
```