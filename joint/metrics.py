import itertools
import os
import numpy as np
import numpy.ma as ma
from sklearn.metrics import f1_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import data

def get_data_from_sequence_batch(true_batch, pred_batch, eos_token):
    """Extract data from a batch of sequences：
    [[3,1,2,0,0,0],[5,2,1,4,0,0]] -> [3,1,2,5,2,1,4]"""
    true_ma = []
    pred_ma = []
    for idx, true in enumerate(true_batch):
        where = true.tolist()
        lentgth = where.index(eos_token)
        true = true[:lentgth]
        pred = pred_batch[idx][:lentgth]
        true_ma.extend(true.tolist())
        pred_ma.extend(pred.tolist())
    return true_ma, pred_ma

def precision_recall_f1_for_sequence(true_batch, pred_batch, average='micro', eos_token='<EOS>'):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, eos_token)
    labels = list(set(true))
    values = precision_recall_fscore_support(true, pred, labels=labels, average=average)
    return {'precision': values[0], 'recall': values[1], 'f1': values[2]}

def precision_recall_f1_for_intents(true, pred, average='micro'):
    values = precision_recall_fscore_support(true, pred, average=average)
    return {'precision': values[0], 'recall': values[1], 'f1': values[2]}

def precision_recall_f1_spans(true_slots_batch, pred_slots_batch):
    """compute f1 from lists of slots, unordered. As what is said in 'What is left to be understood in ATIS'"""
    true_positives_count = true_slots_count = found_slots_count = 0
    for true_slots, pred_slots in zip(true_slots_batch, pred_slots_batch):
        #print(true_slots)
        #print(pred_slots)
        for ts in true_slots:
            if ts in pred_slots:
                # tuple comparison each slot is a tuple type, begin, end
                true_positives_count += 1
        true_slots_count += len(true_slots)
        found_slots_count += len(pred_slots)
    try:
        recall = true_positives_count / true_slots_count
        precision = true_positives_count / found_slots_count
        f1 = (2 * recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1 = 0
        precision = 0
        recall = 0
    return {'precision': precision, 'recall': recall, 'f1': f1}

def precision_recall_f1_spans_conditioned_intent(true_slots_batch, pred_slots_batch, true_intents_batch, pred_intents_batch):
    """compute f1 from lists of slots, unordered. As what is said in 'What is left to be understood in ATIS'"""
    true_positives_count = true_slots_count = found_slots_count = 0
    for true_slots, pred_slots, true_intent, pred_intent in zip(true_slots_batch, pred_slots_batch, true_intents_batch, pred_intents_batch):
        for ts in true_slots:
            if ts in pred_slots and true_intent == pred_intent:
                true_positives_count += 1
        true_slots_count += len(true_slots)
        found_slots_count += len(pred_slots)
    try:
        recall = true_positives_count / true_slots_count
        precision = true_positives_count / found_slots_count
        f1 = (2 * recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1 = 0
        precision = 0
        recall = 0
    return {'precision': precision, 'recall': recall, 'f1': f1}

def plot_history(file_name, history):
    plt.clf()
    for scores in history.values():
        plt.plot(scores)
    
    plt.legend(list(history.keys()), loc='lower right')
    plt.title('model f1')
    plt.ylabel('f1')
    plt.xlabel('epochs')
    plt.grid()
    print(file_name)
    plt.savefig(file_name)

def clean_predictions(decoder_prediction_batch, intent_prediction_batch, true_batch, intent_attentions_batch, bd_attentions_batch, ac_attentions_batch, slots_attentions_batch):
    """Given the raw outputs of the network, provides back a usable representation, with the same structure as the training samples"""

    samples = [{
        'words': gold['words'][:gold['length']],
        'intent_pred': intent_prediction,
        'intent_true': gold['intent'],
        'length': gold['length'],
        'slots_pred': [slot if (slot != '<EOS>' and slot != '<PAD>') else 'O' for slot in decoder_prediction[:gold['length']]],
        'slots_true': gold['slots'][:gold['length']],
        'file': gold.get('file', None),
        'start_token_id': gold.get('start_token_id', None),
        'end_token_id': gold.get('end_token_id', None),
        'id': gold.get('id', None),
        'lexical_unit_ids': gold.get('lexical_unit_ids', []),
        'intent_attentions': [score for score in intent_attentions[:gold['length']]],
        'bd_attentions': [[score for score in line[:gold['length']]] for line in bd_attentions[:gold['length']]],
        'ac_attentions': [[score for score in line[:gold['length']]] for line in ac_attentions[:gold['length']]],
        'slots_attentions': [[score for score in line[:gold['length']]] for line in slots_attentions[:gold['length']]]
    } for (gold, decoder_prediction, intent_prediction, intent_attentions, bd_attentions, ac_attentions, slots_attentions) in zip(true_batch, decoder_prediction_batch, intent_prediction_batch, intent_attentions_batch, bd_attentions_batch, ac_attentions_batch, slots_attentions_batch)]

    return samples

def evaluate_epoch(epoch_data):

    intent_pred, intent_true, slots_pred, slots_true = zip(*[(s['intent_pred'], s['intent_true'], s['slots_pred'], s['slots_true']) for s in epoch_data])

    intent_measures = precision_recall_f1_for_intents(intent_true, intent_pred)

    # argument-based
    slots_arguments_true = data.sequence_iob_to_ents(slots_true)
    slots_arguments_pred = data.sequence_iob_to_ents(slots_pred)
    slots_measures = precision_recall_f1_spans(slots_arguments_true, slots_arguments_pred)
    slots_cond_old_measures = precision_recall_f1_spans_conditioned_intent(slots_arguments_true, slots_arguments_pred, intent_true, intent_pred)

    # BD argument-based
    bd_true = [data.slots_to_iob_only(s) for s in slots_true]
    bd_pred = [data.slots_to_iob_only(s) for s in slots_pred]
    bd_arguments_true = data.sequence_iob_to_ents(bd_true)
    bd_arguments_pred = data.sequence_iob_to_ents(bd_pred)
    bd_measures = precision_recall_f1_spans(bd_arguments_pred, bd_arguments_true)
    bd_measures_cond_old = precision_recall_f1_spans_conditioned_intent(bd_arguments_true, bd_arguments_pred, intent_true, intent_pred)

    # AC argument-based
    ac_true = [data.slots_to_types_only(s) for s in slots_true]
    ac_pred = [data.slots_to_types_only(s) for s in slots_pred]
    # remove consecutive duplicates and 'O'
    # TODO keep idx for considering the right order of entities? but what if a preceeding ent is missing, then counting 0 also for the following ones
    #ac_no_dup_true = [key for key, group in itertools.groupby(ac_true) if key != 'O']
    #ac_no_dup_pred = [key for key, group in itertools.groupby(ac_pred) if key != 'O']
    #ac_measures = precision_recall_f1_spans(ac_no_dup_true, ac_no_dup_pred)
    #ac_measures_cond_old = precision_recall_f1_spans_conditioned_intent(ac_no_dup_true, ac_no_dup_pred, intent_true, intent_pred)



    # The correct way of calculating conditioned probability
    # AI_P_cond | action detection = Tp(argument identification | all Tps from AD) / Tp(argument identification | all Tps from AD) + Fp (argument identification | all Tps from AD)
    # AC_P_cond | argument identification = Tp(argument classification | all Tps from AI) / Tp(argument classification | all Tps from AI) + Fp (argument classification | all Tps from AI)

    # filtered means that the prediction of previous stage was correct: if the AD is wrong, the whole sentence is removed for evaluating next steps
    bd_true_filtered = []
    bd_pred_filtered = []
    ac_true_filtered = []
    ac_pred_filtered = []
    for bd_true_val,  bd_pred_val, ac_true_val, ac_pred_val, intent_true_val, intent_pred_val in zip(bd_arguments_true, bd_arguments_pred, slots_arguments_true, slots_arguments_pred, intent_true, intent_pred):
        if intent_true_val == intent_pred_val:
            bd_true_filtered.append(bd_true_val)
            bd_pred_filtered.append(bd_pred_val)
            ac_true_filtered.append(ac_true_val)
            ac_pred_filtered.append(ac_pred_val)
    bd_measures_cond = precision_recall_f1_spans(bd_pred_filtered, bd_true_filtered)

    # for the AC filter again on the correct bd
    # span-based means that if BD failed on a true span, AC that overlap with that span are be removed
    ac_true_filtered_span_based = []
    ac_pred_filtered_span_based = []
    # sentence_based means that if BD failed on a sentence, the whole AC in the sentence are be removed
    ac_true_filtered_sentence_based = []
    ac_pred_filtered_sentence_based = []
    for ac_arguments_true_val, ac_arguments_pred_val, bd_arguments_true_val, bd_arguments_pred_val in zip(ac_true_filtered, ac_pred_filtered, bd_arguments_true, bd_arguments_pred):
        if sorted(bd_arguments_true_val) == sorted(bd_arguments_pred_val):
            # all right
            ac_true_filtered_sentence_based.append(ac_arguments_true_val)
            ac_pred_filtered_sentence_based.append(ac_arguments_pred_val)
        # forbid all the spans that are in true but not in pred
        wrong_spans = [(bd[1], bd[2]) for bd in bd_arguments_true_val if bd not in bd_arguments_pred_val]
        # and also the ones that are in pred but not in true
        wrong_spans += [(bd[1], bd[2]) for bd in bd_arguments_pred_val if bd not in bd_arguments_true_val]
        overlaps = lambda a, b: (a[1] >= b[0] and a[0] <= b[1])
        ac_true_filtered_span_based.append([ac for ac in ac_arguments_true_val if not any([overlaps((ac[1], ac[2]), s) for s in wrong_spans])])
        ac_pred_filtered_span_based.append([ac for ac in ac_arguments_pred_val if not any([overlaps((ac[1], ac[2]), s) for s in wrong_spans])])
        #print(bd_arguments_true_val, bd_arguments_pred_val, wrong_spans, sorted(bd_arguments_true_val) == sorted(bd_arguments_pred_val), [ac for ac in ac_arguments_true_val if not any([overlaps((ac[1], ac[2]), s) for s in wrong_spans])])

    ac_measures_cond_sent = precision_recall_f1_spans(ac_pred_filtered_sentence_based, ac_true_filtered_sentence_based)
    ac_measures_cond_span = precision_recall_f1_spans(ac_pred_filtered_span_based, ac_true_filtered_span_based)


    # token-based
    #all_slots_true = [x for y in slots_true for x in y]
    #all_slots_pred = [x for y in slots_pred for x in y]
    # TODO change the name of the function
    #token_slots_measures = precision_recall_f1_for_intents(all_slots_true, all_slots_pred)
    # BD token-based
    #all_bd_true = data.slots_to_iob_only(all_slots_true)
    #all_bd_pred = data.slots_to_iob_only(all_slots_pred)
    #token_bd_measures = precision_recall_f1_for_intents(all_bd_true, all_bd_pred)
    # AC token-based
    #all_ac_true = data.slots_to_types_only(all_slots_true)
    #all_ac_pred = data.slots_to_types_only(all_slots_pred)
    #token_ac_measures = precision_recall_f1_for_intents(all_ac_true, all_ac_pred)

    return {
        'intent': intent_measures,
        'slots': slots_measures,
        'slots_cond_old': slots_cond_old_measures,
        #'token_based_slots': token_slots_measures,
        'bd': bd_measures,
        'bd_cond_old': bd_measures_cond_old,
        'bd_cond': bd_measures_cond,
        #'token_based_bd': token_bd_measures,
        #'ac': ac_measures,
        #'ac_cond_old': ac_measures_cond_old,
        #'token_based_ac': token_ac_measures
        'ac_cond_sent': ac_measures_cond_sent,
        'ac_cond_span': ac_measures_cond_span,
        # some counts
        '#sentences': len(intent_true),
        '#bd_spans': sum([len(bd) for bd in bd_arguments_true]),
        '#bd_spans_cond': sum([len(bd) for bd in bd_true_filtered]),
        '#sentences_cond_intent': len(bd_true_filtered),
        '#ac_spans_cond_sent': sum([len(ac) for ac in ac_true_filtered_sentence_based]),
        '#ac_spans_cond_bd': sum([len(ac) for ac in ac_true_filtered_span_based]),
        '#sentences_cond_bd': len(ac_true_filtered_sentence_based)
    }
