import numpy as np
import numpy.ma as ma
from sklearn.metrics import f1_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def precision_recall_f1_slots(true_slots_batch, pred_slots_batch):
    """compute f1 from lists of slots, unordered. As what is said in 'What is left to be understood in ATIS'"""
    true_positives_count = true_slots_count = found_slots_count = 0
    for true_slots, pred_slots in zip(true_slots_batch, pred_slots_batch):
        #print(true_slots)
        #print(pred_slots)
        for ts in true_slots:
            if ts in pred_slots:
                true_positives_count += 1
        true_slots_count += len(true_slots)
        found_slots_count += len(pred_slots)
    try:
        recall = true_positives_count / true_slots_count
        precision = true_positives_count / found_slots_count
        f1 = (2 * recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1 = 0
    return {'precision': precision, 'recall': recall, 'f1': f1}

def precision_recall_f1_slots_conditioned_intent(true_slots_batch, pred_slots_batch, true_intents_batch, pred_intents_batch):
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
