# Data

This folder contains different datasets. For each one there is the source version and the preprocessed one (where the differences in data annotation have been removed).

## ATIS

Charles T Hemphill, John J Godfrey, and George R Doddington. 1990. "The ATIS spoken language systems pilot corpus". _In Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley, Pennsylvania, June 24-27, 1990_.

The dataset ATIS contains intent and slots annotations for single-turned dialogs. The corpus has been retrieved from https://github.com/yvchen/JointSLU/tree/master/data.
Since the split is done with spaces, and the default tokenization with SpaCy is a bit different, this dataset needs the tokenizer `space` to align the word embeddings retrieval inside the tensorflow `py_func`.

## KVRET

Mihail Eric and Christopher D Manning. 2017. "Key-Value Retrieval Networks for Task-Oriented Dialogue". _In Proceedings of the 18th Annual SIGdial MeetingDiscourse and Dialogue_. Association for Computational Linguistics, Saarbrücken, Germany, 37–49.

The dataset KVRET contains intent and slots annotations for multi-turn dialogs. The corpus has been retrieved from https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/.

## Nlu-benchmark

The dataset contains intent and slots annotations for single-turn dialogs. The corpus has been retrieved from https://github.com/snipsco/nlu-benchmark.
