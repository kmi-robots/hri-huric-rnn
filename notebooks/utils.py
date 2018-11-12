import json

import math
import numpy as np
from collections import defaultdict
from itertools import groupby, product
from pathlib import Path
import xml.etree.ElementTree as ET
from IPython.display import HTML, display
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support
from textstat.textstat import textstat

import nltk
from nltk.tag import pos_tag, map_tag

from spacy.gold import tags_to_entities, iob_to_biluo

from nlunetwork import data

def load_json(folder, epoch=99):
    if epoch == None:
        # loading directly the preprocessed files
        json_location = Path(folder) / 'all_samples.json'
    else:
        json_location = Path(folder) / 'json' / 'epoch_{}'.format(epoch) / 'prediction_fold_full.json'
    with open(json_location) as f:
        content = json.load(f)
    samples = content.get('samples', None)
    if not samples:
        samples = content['data']
    return sorted(samples, key=lambda el: el['file'] if el.get('file', None) else 0)

def get_intent_attention_arrays(sentence_data):
    """returns true (lexical units) and pred attentions"""
    lexical_units = [0] * sentence_data['length']
    more = [0] * sentence_data['length']
    for lu in sentence_data['lexical_unit_ids']:
        lexical_units[lu - sentence_data['start_token_id']] = 1
    for lu in sentence_data['lexical_unit_ids_more']:
        index = lu - sentence_data['start_token_id']
        if not lexical_units[index]:
            more[index] = 1
    return lexical_units, more, sentence_data.get('intent_attentions', [])

#def get_bd_attention_arrays(sentence_data):
#    return sentence_data['words'], sentence_data['slots_pred']

def get_color(value):
    """Colors a cell by the value provided [0,1] if value is number, otherwise white"""

    if isinstance(value, str):
        return 'rgb(255,255,255)'
        print('str', value)
    else:
        v = "%.4f" %  (255 - value * 255)
        return 'rgb({}, 255,{})'.format(v, v)

def float_to_str(value):
    if isinstance(value, str):
        return value
    else:
        return "%.4f" % float(value)

def display_sequences(row_names, sequences):
    html_str = '<table><tr>{}</tr></table>'.format(
        '</tr><tr>'.join(
            '<td><b>{}</b></td>'.format(row_name) +
            ''.join(['<td style="background-color: {};">{}</td>'.format(get_color(value), float_to_str(value)) for value in row])
            for row_name, row in zip(row_names, sequences)
        )
    )
    display(HTML(html_str))

def display_align(row_labels, col_labels, values):
    html_str = '<table><tr>{}</tr><tr>{}</tr></table>'.format(
        # the headers
        ''.join(['<td>{}</td>'.format(t) for t in [''] + row_labels]),
        # the body
        '</tr><tr>'.join(
            '<td>{}</td>'.format(col_labels[row_idx]) +
            ''.join(['<td style="background-color: {};">{}</td>'.format(get_color(value), float_to_str(value)) for value in row])
            for row_idx, row in enumerate(values)
        )
    )
    display(HTML(html_str))

def display_all(samples, display_bd=True, display_ac=True, display_slots=True, max_show=None):
    for idx, sample in enumerate(samples):
        print('true:', sample['intent_true'], 'pred:', sample['intent_pred'])
        attentions_true, disc, attentions_pred = get_intent_attention_arrays(sample)
        display_sequences(['words', 'lexical_unit', 'disc','attention_intent', 'slots_true', 'slots_pred'],
                          [sample['words'], attentions_true, disc, attentions_pred, sample['slots_true'], sample['slots_pred']])
        bd = data.slots_to_iob_only(sample['slots_pred'])
        ac = data.slots_to_types_only(sample['slots_pred'])
        if display_bd:
            display_align(sample['words'], bd, sample['bd_attentions'])
        bd_and_words = ['{}+{}'.format(b, w) for b, w in zip(bd, sample['words'])]
        if display_ac:
            display_align(bd_and_words, ac, sample['ac_attentions'])
        if display_slots:
            display_align(sample['words'], sample['slots_pred'], sample['slots_attentions'])
        # don't make the browser crash with all those tables
        if max_show and idx > max_show:
            return

def ad_align_accuracy_argmax(samples, which='lu'):
    """Returns a measure of how many times the highest value of attention is captured by the Lexical Unit
    which can be lu or more"""
    if which == 'lu':
        key = 'lexical_unit_ids'
    elif which == 'more':
        key = 'lexical_unit_ids_more'
    else:
        raise ValueError('invalid which' + str(which))
    total = 0
    for sample in samples:
        true_lus = [v -sample['start_token_id'] for v in sample[key]]
        n_lus = len(true_lus)
        pred_lus = np.argpartition(sample['intent_attentions'], -n_lus)[-n_lus:]
        compared = [1 if t in pred_lus else 0 for t in true_lus]
        total += sum(compared) / len (compared)
        #pred_lus = [np.argmax(s['intent_attentions']) for s in samples]
    #true_lexical_units = [s[key][0] - s['start_token_id'] for s in samples]
    #pred_lexical_units = [np.argmax(s['intent_attentions']) for s in samples]
    #compared = [1 if t == p else 0 for t, p in zip(true_lexical_units, pred_lexical_units)]
    return total / len(samples)

def ad_average_attention(samples, which='lu'):
    """Returns the average value of attention on the Lexical Units"""
    total = 0.
    if which == 'lu':
        key = 'lexical_unit_ids'
    elif which == 'more':
        key = 'lexical_unit_ids_more'
    else:
        raise ValueError('invalid which' + str(which))
    for s in samples:
        # get the indexes of the lexical units in the sentence
        true_lexical_units = [val - s['start_token_id'] for val in s[key]]
        # get their value of attention (summing them if more than one LU is there)
        attn_val = np.sum(np.array(s['intent_attentions'])[true_lexical_units])
        total += attn_val

    return total / len(samples)

def align_score(samples, max_len=50):
    true_lexical_units = []
    for s in samples:
        true_s = [1 if idx + 1 in s['lexical_unit_ids'] else 0 for idx in range(s['length'])]
        #print('t', true_s)
        true_lexical_units.extend(true_s)
    #print(np.shape(true_lexical_units))

    precisions_vs_k = {}
    recalls_vs_k = {}
    f1_vs_k = {}

    for how_many in range(1, 10):
        pred_lexical_units = []
        for s in samples:
            padded_intent_att = np.pad(s['intent_attentions'], (0, max_len - s['length']), 'constant')
            pred_k_indexes = np.argpartition(padded_intent_att, -how_many)[-how_many:]
            #print(pred_k_indexes)
            pred_k = [1 if idx in pred_k_indexes else 0 for idx in range(s['length'])]
            #print(pred_k)
            pred_lexical_units.extend(pred_k)
        #print('true', true_lexical_units[:30])
        #print('pred', pred_lexical_units[:30])
        p, r, f1, support = precision_recall_fscore_support(true_lexical_units, pred_lexical_units, average='binary')
        #print('k=', how_many, 'p=', p, 'r=', r, 'f1=', f1, 'support=', support)
        precisions_vs_k[how_many] = p # with average='binary' the scores are for the 1 label, the interesting one
        recalls_vs_k[how_many] = r
        f1_vs_k[how_many] = f1

    display_alignment_vs_k(precisions_vs_k, recalls_vs_k, f1_vs_k)
    return precisions_vs_k, recalls_vs_k, f1_vs_k

def display_alignment_vs_k(precisions, recalls, f1s):
    plt.clf()
    # TODO for ROC computation should have TPR and FPR, not precision and recall!!
    # tprs = recalls
    # fprs =
    plt.plot(precisions.keys(), precisions.values(), label='precision')
    plt.plot(recalls.keys(), recalls.values(), label='recall')
    plt.xlabel('k-top')
    plt.legend(['precision', 'recall'], loc='lower right')
    plt.show()

def group_samples_by_frame(samples):
    groups = {k: list(v) for k,v in groupby(sorted(samples, key=lambda s: s['intent_true']), key=lambda s: s['intent_true'])}
    return groups

def get_words_by_attention(samples):
    """Given a set of samples, returns a list of (word, average_attention) sorted by decreasing attention"""
    bow_cumulative = defaultdict(lambda: 0)
    for s in samples:
        for w_idx, w in enumerate(s['words']):
            bow_cumulative[w] += s['intent_attentions'][w_idx]
        #print(s['intent_attentions'])
    bow_cumulative = {k: score / len(samples) for k, score in bow_cumulative.items()}
    result = sorted([(w, w_sum) for (w, w_sum) in bow_cumulative.items()], key=lambda x: x[1], reverse=True)
    return result

def load_xmls(folder):
    path = Path(folder)
    files_list = [el for el in path.iterdir() if el.is_file()]
    file_contents = []
    for file in sorted(files_list):
        with open(file) as file_in:
            tree = ET.parse(file_in)
        root = tree.getroot()
        file_contents.append(root)
    return file_contents

def get_additional_discriminators_idxs(samples, gold_missing):
    """Return a map that says for each sample what are the words non-LU that can help as discriminators for Frame Classification"""
    additional_per_lemma = {
        # those values have been identified by looking at the results of the function get_lemma_invoker and cleaning them up
        'be': {'Being_in_category': [], 'Being_located': ['on','in']},
        'move': {'Motion': ['near', 'to', 'towards', 'along', 'away', 'from'], 'Bringing': ['from']},
        'put': {'Placing': ['in', 'on', 'under', 'into', 'down'], 'Closure': ['all', 'the', 'way', 'down'], 'Change_operational_state': ['on']},
        'get': {'Taking': ['from', 'to', 'in'], 'Bringing': ['me']},
        'take': {'Bringing': ['to', 'into', 'out', 'me'], 'Taking': ['in', 'near', 'on', 'from']},
        'come': {'Following': ['with'], 'Motion': ['to']},
        'open': {'Change_operational_state': [], 'Closure': []},
        'look': {'Perception_active': ['at'], 'Searching': ['for']}
    }

    results = {}

    for s in samples:
        xml_file_name = s['file']
        # ambiguous LU are made of only one word
        lu_idx = s['lexical_unit_ids'][0] - 1
        lu_lemma = gold_missing[s['file']]['lemmas'][lu_idx]
        additional = additional_per_lemma.get(lu_lemma, None)
        interesting = []
        if 'intent_true' in s:
            key_intent = 'intent_true'
        else:
            key_intent = 'intent'
        if additional:
            #this is an ambiguous LU lemma
            interesting = [idx + s['start_token_id'] for idx, w in enumerate(s['words']) if w in additional[s[key_intent]]]

        results[s['id']] = interesting

    return results

def add_discriminators(samples, gold_missing):
    additional = get_additional_discriminators_idxs(samples, gold_missing)
    for s in samples:
        s['lexical_unit_ids_more'] = s['lexical_unit_ids'] + additional[s['id']]

def get_lu_pos(root, verbose=False):
    """Returns the POS for all the LU in the current document"""
    w_id_to_pos = {t.attrib['id']: t.attrib['pos'] for t in root.findall('tokens/token')}
    #print(w_id_to_pos)
    lu_idxs = [lu.attrib['id'] for lu in root.findall('semantics/frameSemantics/frame/lexicalUnit/token')]
    #print(lu_idxs)
    lu_pos = [w_id_to_pos[id] for id in lu_idxs]
    # get the simplified universal tags
    lu_pos = [map_tag('en-ptb', 'universal', t) for t in lu_pos]
    if verbose:
        print(root.attrib['id'], lu_pos)
    return lu_pos

def get_lu_are_roots(root, verbose=False):
    """Returns whether the lexicalUnits are the roots in dependencies"""
    roots_id = [d.attrib['to'] for d in root.findall('dependencies/dep') if d.attrib['type'] == 'root']
    lu_idxs = [lu.attrib['id'] for lu in root.findall('semantics/frameSemantics/frame/lexicalUnit/token')]
    lu_are_roots = [str(l in roots_id) for l in lu_idxs]
    if verbose:
        print(root.attrib['id'], lu_are_roots)
    return lu_are_roots

def get_lengths(root, verbose=False):
    """Returns the length of the command"""
    # TODO shouldn't be frame-based? find min and max token id and do difference
    return [len(root.findall('tokens/token'))]

def get_lu_depths(root, verbose=False):
    """Returns the depths in the dependency tree of the lexicalUnits"""
    # TODO the depth should be relative to the frame
    edges = [(d.attrib['from'], d.attrib['to'], d.attrib['type']) for d in root.findall('dependencies/dep')]
    #print(edges)
    to_father = {e[1]: e[0] for e in edges}
    depths = {}
    for el, f in to_father.items():
        current_el = el
        depth = 0
        # the root has id == '0', the second condition is only to avoid infinite looping
        while f != '0' and depth < len(edges):
            depth += 1
            current_el = f
            f = to_father[current_el]
        if f != '0':
            # broken annotations
            depth = -1
        depths[el] = depth
    lu_idxs = [lu.attrib['id'] for lu in root.findall('semantics/frameSemantics/frame/lexicalUnit/token')]
    lu_depths = [depths[l] for l in lu_idxs]
    if verbose:
        print(root.attrib['id'], lu_depths)
    return lu_depths

def get_lu_positions(root, verbose=False):
    """Get the position of lexicalUnits in the command"""
    # TODO the position should be relative to the frame
    lu_idxs = [lu.attrib['id'] for lu in root.findall('semantics/frameSemantics/frame/lexicalUnit/token')]
    lu_positions = [int(l) for l in lu_idxs]
    return lu_positions

def get_corpus_complexity_statistics(dataset_location):
    """Loads the dataset in xml format HuRIC 1.2 and computes some measures"""
    xml_docs = load_xmls(dataset_location)
    lu_pos_all = defaultdict(lambda: 0)
    lu_are_roots_all = defaultdict(lambda: 0)
    lengths_all = defaultdict(lambda: 0)
    lu_depths_all = defaultdict(lambda: 0)
    lu_positions_all = defaultdict(lambda: 0)
    flesch_ease_all = defaultdict(lambda: 0)
    fog_all = defaultdict(lambda: 0)
    smog_all = defaultdict(lambda: 0)
    automated_all = defaultdict(lambda: 0)

    frame_stats = defaultdict(lambda: 0)

    for doc in xml_docs:
        frame_names = get_frame_names(doc)
        for fn in frame_names:
            frame_stats[fn] += 1
        lu_pos = get_lu_pos(doc)
        for p in lu_pos:
            lu_pos_all[p] += 1
        lu_are_roots = get_lu_are_roots(doc)
        for r in lu_are_roots:
            lu_are_roots_all[r] += 1
        lengths = get_lengths(doc)
        for l in lengths:
            lengths_all[l] += 1
        lu_depths = get_lu_depths(doc)
        for d in lu_depths:
            lu_depths_all[d] += 1
        lu_positions = get_lu_positions(doc)
        for p in lu_positions:
            lu_positions_all[p] += 1
        # readability scores
        text = doc.find('sentence').text
        flesch = textstat.flesch_reading_ease(text)
        school_level = flesch_to_school_level(flesch)
        flesch_ease_all[school_level] += 1
        fog = math.floor(textstat.gunning_fog(text))
        fog_all[fog] += 1
        smog = math.floor(textstat.smog_index(text))
        smog_all[smog] += 1
        automated = math.floor(textstat.automated_readability_index(text))
        automated_all[automated] += 1



    result = {
        'lu_pos': lu_pos_all,
        'lu_are_roots': lu_are_roots_all,
        'lengths': lengths_all,
        'lu_depths': lu_depths_all,
        'lu_positions': lu_positions_all,
        'flesch_reading_ease': flesch_ease_all,
        'gunning_fog_grade': fog_all,
        'smog_grade': smog_all,
        'automated_readability_index_grade': automated_all
    }

    return result, frame_stats

def get_frame_names(doc):
    frames = doc.findall('semantics/frameSemantics/frame')
    return [f.attrib['name'] for f in frames]

def flesch_to_school_level(p):
    if p <= 30:
        return 'College graduate'
    elif p > 30 and p <= 50:
        return 'College'
    elif p > 50 and p <= 60:
        return '10th to 12th grade'
    elif p > 60 and p <= 70:
        return '08th to 9th grade'
    elif p > 70 and p <= 80:
        return '07th grade'
    elif p > 80 and p <= 90:
        return '06th grade'
    elif p > 90:
        return '05th grade'

def plot_measures(datasets_dict, show=True, path=None):
    """Given a dict {conf_name: measures} produces a set of plots, one for each measure, with all the configurations inside"""
    # reverse the nested dict to {measure_name: {conf_name: {label: value}}
    stats_by_measure = defaultdict(dict)
    for conf_name, measures in datasets_dict.items():
        for measure_name, measure in measures.items():
            stats_by_measure[measure_name][conf_name] = measure
    # adjust the width of the columns
    width = 0.5 / len(datasets_dict.keys())
    #print('stats_by_measure', stats_by_measure)
    for measure_name, confs_measures in stats_by_measure.items():
        fig, ax = plt.subplots(figsize=(16, 5))
        # merge all the measures indexes, that can have different values among the configurations
        # first get the set of all unique indexes
        all_indexes = sorted(set([m for measure in confs_measures.values() for m in measure.keys()]))
        #print('all_idexes', all_indexes)
        # then build a map {index_value: position_in_all_indexes}
        all_indexes = {val: idx for idx, val in enumerate(all_indexes)}
        #print('all_indexes', all_indexes)
        bars = []
        for idx, (conf_name, measure) in enumerate(confs_measures.items()):
            corresponding_x_indexes = [all_indexes[m] for m in measure]
            total_n_samples = sum(measure.values())
            bar = ax.bar(np.array(corresponding_x_indexes) + (idx * width), np.array([m for m in measure.values()]) / total_n_samples, width)
            bars.append(bar)
        ax.set_ylabel('%')
        ax.set_title(measure_name)
        ax.set_xticks(np.arange(len(all_indexes)) + (width / 2 * (len(datasets_dict) - 1)))
        ax.set_xticklabels(list(all_indexes.keys()))
        ax.legend(bars, confs_measures.keys())
        if path:
            _ = plt.savefig('{}_{}.png'.format(path, measure_name))
        if show:
            plt.show()
        plt.close(fig)

def get_frame_elements_span(samples):
    """Returns a list of spans that contain gold frame elements"""
    result = []
    for s in samples:
        biluo = iob_to_biluo(s['slots_true'])
        entities = tags_to_entities(biluo)
        #print(entities)
        for e in entities:
            result.append({
                'sample_id': s['id'],
                'type': e[0],
                'start': e[1],
                'end': e[2]
            })
    return result

def get_attention_average_from_word_indexes(sample, observer_indexes, observed_indexes, task='bd'):
    """For n x n tasks ('bd' and 'ac'), returns the average value of the attentions from the indexes 'observer_indexes' on the 'observed_indexes'"""
    if task == 'bd':
        attention_key = 'bd_attentions'
    elif task == 'ac':
        attention_key = 'ac_attentions'
    else:
        raise ValueError(task)
    attention_matrix = np.array(sample[attention_key])
    mean_value = sum([np.mean([attention_matrix[obs, obd] for obs in observer_indexes]) for obd in observed_indexes])
    return mean_value

def get_attention_argmax_percentage_from_word_indexes(sample, observer_indexes, observed_indexes, task='bd'):
    """For n x n tasks ('bd' and 'ac'), returns a percentage of how many times the 'observed_indexes' get the attention from the 'observer_indexes'"""
    if task == 'bd':
        attention_key = 'bd_attentions'
    elif task == 'ac':
        attention_key = 'ac_attentions'
    else:
        raise ValueError(task)
    attention_matrix = np.array(sample[attention_key])
    # get the length of the observed
    observed_length = len(observed_indexes)
    if not observed_length:
        # sometimes we are looking for nouns on spans that have no nouns, so add a shortcut
        return 0.
    # based on K, the length of the observed, get the K-top highest values for each row of the matrix
    # - argsort returns the indices that would sort the matrix rows
    # - the [:,::-1] is to reverse (want high values) each row
    # - the [:,:K] is to get the K-top for each row
    top_k_for_rows = attention_matrix.argsort()[:,::-1][:,:observed_length]
    total_count = 0
    # iterate on the rows of the observers
    for row in top_k_for_rows[observer_indexes]:
        # count how many observed are K-top for the row
        present = sum([el in observed_indexes for el in row])
        total_count += present
    # and normalize the score
    #print(observed_length, len(observer_indexes))
    return total_count / (observed_length * len(observer_indexes))

def get_attention_score_on_task(samples, gold_missing, task):
    spans = get_frame_elements_span(samples)
    samples_by_id = {s['id']: s for s in samples}
    overall_totals = defaultdict(lambda: 0)
    for span in spans:
        sample = samples_by_id[span['sample_id']]
        sample_missing = gold_missing[sample['file']]
        sample_pos_list = sample_missing['pos'][sample['start_token_id']:sample['end_token_id']+1]
        # define the interesting things
        interesting = {
            'first_word_of_span': [span['start']], # the first word in the span is usually a preposition, very important for Boundary Detection
            'lexical_unit': [el - sample['start_token_id'] for el in sample['lexical_unit_ids']],
            'nouns': [i for i, value in enumerate(sample_pos_list) if value == 'NOUN']
        }
        for why, observed in interesting.items():
            #print(span, why, sample)
            average_value = get_attention_average_from_word_indexes(sample, range(span['start'], span['end'] + 1), observed, task)
            argmax_value = get_attention_argmax_percentage_from_word_indexes(sample, range(span['start'], span['end'] + 1), observed, task)
            #print(s, value)
            overall_totals[why + '_average'] += average_value
            overall_totals[why + '_argmax'] += argmax_value
    result = {k: v / len(spans) for k,v in overall_totals.items()}

    return result

def get_samples_pos_and_lemmas_and_deps(gold_location):
    """Returns a map {file_name: {'pos': [POS_LIST], 'deps': [{'from':FROM, 'to':TO, 'type':TYPE}]}}
    all these informations are missing in the json files"""
    docs = load_xmls(gold_location)
    result = {
        doc.attrib['id'] + '.xml': {
            'pos': [map_tag('en-ptb', 'universal', t.attrib['pos']) for t in doc.findall('tokens/token')],
            'lemmas': [t.attrib['lemma'] for t in doc.findall('tokens/token')],
            'deps': [d for d in doc.findall('dependencies/dep')]
        }
    for doc in docs}

    return result

def get_lemma_invoker(xmls):
    """Returns a {lemma: [{frameName: count}]}"""

    result = defaultdict(lambda: defaultdict(lambda: []))

    for x in xmls:
        tokens = {t.attrib['id']: t.attrib for t in x.findall('tokens/token')}
        for sf in x.findall('semantics/frameSemantics/frame'):
            f_name = sf.attrib['name']
            lus = [t.attrib['id'] for t in sf.findall('lexicalUnit/token')]
            lus_tokens = [tokens[l] for l in lus]
            lus_lemmas = [l['lemma'] for l in lus_tokens]

            lus_lemma_stringified = ' '.join(lus_lemmas)
            result[lus_lemma_stringified][f_name].append(x.find('sentence').text.strip())

    return result

def get_attention_scores(samples, gold_missing):
    add_discriminators(samples, gold_missing)
    result = {
        'ad': {
            'argmax': ad_align_accuracy_argmax(samples),
            'average': ad_average_attention(samples)
        },
        'ad_more': {
            'argmax': ad_align_accuracy_argmax(samples, 'more'),
            'average': ad_average_attention(samples, 'more')
        },
        'ai': get_attention_score_on_task(samples, gold_missing, 'bd'),
        'ac': get_attention_score_on_task(samples, gold_missing, 'ac')
    }
    return result

def write_to_tsv(samples, out_path):
    lines = [['id', 'huric_source_file', 'frame', 'sentence', 'lu', 'disc', 'lu+disc']]
    for s in samples:
        lus, discs, _ = get_intent_attention_arrays(s)
        lus_and_discs = [1 if (lu or disc) else 0 for lu,disc in zip(lus, discs)]
        int_array_to_str_array = lambda arr: [str(el) for el in arr]
        lus = int_array_to_str_array(lus)
        discs = int_array_to_str_array(discs)
        lus_and_discs = int_array_to_str_array(lus_and_discs)
        lines.append([str(s['id']), s['file'], s['intent'], ','.join(s['words']), ','.join(lus), ','.join(discs), ','.join(lus_and_discs)])
    with open(out_path, 'w') as f:
        f.write('\n'.join(['\t'.join(l) for l in lines]))


if __name__ == '__main__':
    import numpy as np

    # put there the path to the results you want to show

    HURIC_JSON_LOCATION='data/huric/modern_right/preprocessed'
    HURIC_XML_LOCATION = 'data/huric/modern/source'
    samples = load_json(HURIC_JSON_LOCATION, None)
    gold_missing = get_samples_pos_and_lemmas_and_deps(HURIC_XML_LOCATION)
    add_discriminators(samples, gold_missing)

    write_to_tsv(samples, 'data/huric/modern_right/lu_and_disc.tsv')
