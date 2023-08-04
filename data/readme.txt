# Introduction

This directory contains the automatically generated text (AP.csv, EfA.csv, EnAP.csv,
and EP.csv, ATG.csv) and the results from the human evaluation.

## Automatic Generated text.

Each file contains the generated text from the compatible model configurations 
(EA, A and E) from the two model architectures (T5 and Bart).

### Variables:
-   Prompt: Prompt set used to generate the text
-   EA-bart-base: Generated text using Bart conditioned on Emotion and Appraisals.
-   A-bart-base: Generated text using Bart conditioned on Appraisals.
-   E-bart-base: Generated text using Bart conditioned on Emotion.
-   EA-t5-base: Generated text using T5 conditioned on Emotion and Appraisals.
-   A-t5-base: Generated text using T5 conditioned on Appraisals.
-   E-t5-base: Generated text using T5 conditioned on Emotions.



## Human Evaluation.

The human evaluation is  performed on 330 sentences, 30 human-generated sentences 
from the crow-enVent dataset, and 100 sentences randomly selected from each of 
the following model configuration and prompt sets: EA with EP, E with EP, and
EA with EmfAP.



### Variables

Each text has been evaluated by three different annotators. Each one was
reate in a five-level Likerscale.

-   Annotator 0-2: Annotator id for each sentence.
-   Text : Text to evaluate.
-   PromptSet: Prompt set used to generate the text (EP, and EmFAP) 
	and the enVent (human-generated text).
-   Architecture: Model used to generate the text (EA and E) and Human
	(human-generated text).
-   Emotion: The emotion used to condition the generated text.
-   Attention, Responsibility, Control, Circumstance, Pleasantness, Effort
	Certainty 0,1: Appraisals used to condition the generated text.
-   a_Attention, a_Responsibility, a_Control, a_Circumstance, a_Pleasantness, 
	a_Effort, a_Certainty  0-5: Appraisal score given by the annotator.
-   a_Anger, a_Disgust, a_Fear, a_Guilty, a_Joy, a_Sadness, a_Shame 0-5: Emotion
	score given by the annotator.
-   Fluent, Grammatical_error, Native_speaker, Coherence, Really_happened,
	 Written_AI, Written_Human 0-5: Text quality score given by the annotator.

