"""
This module preprocesses the datasets in equivalent formats
"""
import json
import os
import re
import numpy as np
import spacy
import csv
from itertools import groupby
from collections import defaultdict
from spacy.gold import biluo_tags_from_offsets
from sklearn.model_selection import StratifiedKFold
import xml.etree.ElementTree as ET 

def huric_preprocess_old(path, nlp):
    path_source = path + '/source'

    tree = ET.parse('{}/example.xml'.format(path_source))
    root = tree.getroot()

    samples = []
    intent_types = set()
    slot_types = set()
    for command in root:
        tokens = [t.attrib['surface'] for t in command.find('tokens').findall('token')]
        frame_name = command.find('semantics/frameSemantics/frame').attrib['name']
        frame_type = command.find('semantics/frameSemantics/frame/frameElement').attrib['type']
        intent = '{}-{}'.format(frame_name,frame_type)
        print(tokens)
        print('intent: ' + intent)
        slots = ['O'] * len(tokens)
        for role in command.find('semantics/spatialSemantics').findall('spatialRelation/spatialRole'):
            slot_type = role.attrib['type']
            token_ids = [int(t.attrib['id']) - 1 for t in role.findall('token')]
            for count, token_id in enumerate(token_ids):
                prefix = 'B' if not count else 'I'
                slots[token_id] = '{}-{}'.format(prefix, slot_type)
        print(slots)
        samples.append({'words': tokens,
                        'intent': intent,
                        'length': len(tokens),
                        'slots':slots})

        intent_types.add(intent)
        slot_types.update(slots)

    dataset = np.array(samples)
    # do the stratified split on 5 folds, fixing the random seed
    slot_types = list(sorted(slot_types))
    intent_types = list(sorted(intent_types))
    # the value of intent for each sample, necessary to perform the stratified split (keeping distribution of intents in splits)
    intent_values = [s['intent'] for s in samples]
    return
    # TODO this stuff is broken, see https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=dataset.size)
    folds_indexes = []
    for train_idx, test_idx in sss.split(np.zeros(len(intent_values)), intent_values):
        #print('train idx', train_idx, 'test idx', test_idx)
        folds_indexes.append(test_idx.tolist())
        #print(train_idx, test_idx)
    
    train, dev, final_test = (dataset[folds_indexes[0] + folds_indexes[1] + folds_indexes[2]], dataset[folds_indexes[3]],dataset[folds_indexes[4]])

    meta = {
        'tokenizer': 'whitespace',
        'language': 'en',
        'intent_types': intent_types,
        'slot_types': slot_types
    }

    if not os.path.exists('{}/preprocessed'.format(path)):
        os.makedirs('{}/preprocessed'.format(path))

    with open('{}/preprocessed/fold_train.json'.format(path), 'w') as outfile:
        json.dump({
            'data': train.tolist(),
            'meta': meta
        }, outfile)
    
    with open('{}/preprocessed/fold_test.json'.format(path), 'w') as outfile:
        json.dump({
            'data': dev.tolist(),
            'meta': meta
        }, outfile)
    
    with open('{}/preprocessed/final_test.json'.format(path), 'w') as outfile:
        json.dump({
            'data': final_test.tolist(),
            'meta': meta
        }, outfile)

def huric_preprocess_eb(path, nlp):
    """Preprocess the huric dataset, passed by EB"""
    def get_int_id(string_id):
        #print(string_id)
        return int(string_id.split('.')[3])


    path_source = path + '/source'

    samples = []
    intent_types = set()
    # a resume of how sentences are split into frames
    splitting_resume = {}
    slot_types = set()
    for filename in sorted(os.listdir(path_source)):
        tree = ET.parse('{}/{}'.format(path_source, filename))
        root = tree.getroot()
        # the main object used is xdg
        xdg = root.find('PARAGRAPHS/P/XDGS/XDG')

        # read the tokens, saving in a map indexed by the serializerID
        tokens_map = {sc.attrib['serializerID']: sc.attrib['surface'] for sc in xdg.findall('CSTS/SC')}

        # where the last frame is beginning
        start_of_frame = 0

        splitting_resume[filename] = {'all_words': ' '.join([value for (key, value) in tokens_map.items()]), 'frames':[]}
        # multiple frames possible for each sentence. Frames are represented by items in the interpretation list
        for item in xdg.findall('interpretations/interpretationList/item'):
            intent = item.attrib['name']
            #print('intent: ' + intent)

            # accumulator for all the tokens mentioned in the current frame
            frame_tokens_mentioned = item.find('constituentList').text
            frame_tokens_mentioned = frame_tokens_mentioned.split(' ') if frame_tokens_mentioned else []
            slots_map = {}
            for arg in item.findall('ARGS/sem_arg'):
                slot_type = arg.attrib['argumentType']
                token_ids = arg.find('constituentList').text
                token_ids = token_ids.split(' ') if token_ids else token_ids
                frame_tokens_mentioned.extend(token_ids)
                #words += [tokens_map[id] for id in token_ids]
                for count, token_id in enumerate(token_ids):
                    prefix = 'B' if not count else 'I'
                    iob_label = '{}-{}'.format(prefix, slot_type)
                    slots_map[token_id] = iob_label
                    #print('found a slot', iob_label, token_id)
                    #slots.append('{}-{}'.format(prefix, slot_type))
            
            # now find the part of the sentence that is related to the specific frame
            #print(frame_tokens_mentioned)
            frame_tokens_mentioned = [get_int_id(id) for id in frame_tokens_mentioned]
            min_token_id, max_token_id = min(frame_tokens_mentioned), max(frame_tokens_mentioned)
            #print('min max token id', min_token_id, max_token_id)
            # the old division with min max
            #frame_tokens = {key: value for (key, value) in tokens_map.items() if get_int_id(key) >= min_token_id and get_int_id(key) <= max_token_id}
            # the correct division [0:max1],[max1:max2]. 'and' words between are part of the new frame. 'robot can you' words are part of the first frame
            frame_tokens = {key: value for (key, value) in tokens_map.items() if get_int_id(key) >= start_of_frame and get_int_id(key) <= max_token_id}
            words = [value for (key, value) in frame_tokens.items()]
            slots = [slots_map.get(key, 'O') for (key, value) in frame_tokens.items()]
            start_of_frame = max_token_id + 1
            if not len(words):
                print('len 0 for', frame_tokens_mentioned, tokens_map, frame_tokens, filename)
            #print(words)
            #print(slots)
            sample = {'words': words,
                            'intent': intent,
                            'length': len(words),
                            'slots':slots,
                            'file': filename}

            splitting_resume[filename]['frames'].append({'name': intent, 'words': ' '.join(words), 'slots': ' '.join(slots)})

            samples.append(sample)
            intent_types.add(intent)
            slot_types.update(slots)
        #print(tokens_map)
        print('new file')

    with open('{}/resume.json'.format(path), 'w') as outfile:
        json.dump(splitting_resume, outfile, indent=2)

    # do the stratified split on 5 folds, fixing the random seed
    slot_types = list(sorted(slot_types))
    intent_types = list(sorted(intent_types))
    # the value of intent for each sample, necessary to perform the stratified split (keeping distribution of intents in splits)
    intent_values = [s['intent'] for s in samples]
    count_by_intent = {key:len(list(group)) for (key,group) in groupby(sorted(intent_values))}
    # remove intents that have less members than the number of splits to make StratifiedKFold work
    intent_remove = [key for (key, value) in count_by_intent.items() if value < 5]
    print('removing samples with the following intents:', intent_remove)
    samples = [s for s in samples if not s['intent'] in intent_remove]
    intent_values = [s['intent'] for s in samples]
    dataset = np.array(samples)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=dataset.size)
    folds_indexes = []
    for train_idx, test_idx in skf.split(np.zeros(len(intent_values)), intent_values):
        #print('train idx', train_idx, 'test idx', test_idx)
        folds_indexes.append(test_idx.tolist())
        #print(train_idx, test_idx)
    
    # TODO really store the 5 folds separately, not only train,dev,test
    train, dev, final_test = (dataset[folds_indexes[0] + folds_indexes[1] + folds_indexes[2]], dataset[folds_indexes[3]],dataset[folds_indexes[4]])

    meta = {
        'tokenizer': 'whitespace',
        'language': 'en',
        'intent_types': intent_types,
        'slot_types': slot_types
    }

    if not os.path.exists('{}/preprocessed'.format(path)):
        os.makedirs('{}/preprocessed'.format(path))

    with open('{}/preprocessed/fold_train.json'.format(path), 'w') as outfile:
        json.dump({
            'data': train.tolist(),
            'meta': meta
        }, outfile)
    
    with open('{}/preprocessed/fold_test.json'.format(path), 'w') as outfile:
        json.dump({
            'data': dev.tolist(),
            'meta': meta
        }, outfile)
    
    with open('{}/preprocessed/final_test.json'.format(path), 'w') as outfile:
        json.dump({
            'data': final_test.tolist(),
            'meta': meta
        }, outfile)


def huric_preprocess(path, invoke_frame_slot=False):
    """Preprocess the huric dataset, provided by Danilo Croce"""

    def overlap(a, b):
        return max(0, min(a['max'], b['max']) - max(a['min'], b['min']))

    def covers(a, b):
        return (a['max'] >= b['max']) and (a['min'] <= b['min'])

    path_source = path + '/source'

    samples = []
    # a resume of how sentences are split into frames
    splitting_resume = {}
    slot_types = set()
    file_locations = {}

    # retrieve all the possible xml files in the three subfolders
    for subfolder in ['GrammarGenerated', 'S4R_Experiment', 'Robocup']:
        for filename in os.listdir('{}/{}/xml'.format(path_source, subfolder)):
            file_locations[filename] = subfolder

    # TODO remove this
    file_locations.pop('2307.xml')
    file_locations.pop('2353.xml')
    file_locations.pop('2374.xml')
    file_locations.pop('2377.xml')
    file_locations.pop('2411.xml')
    file_locations.pop('3395.xml')
    print('#files: ', len(file_locations))

    for filename, subfolder in sorted(file_locations.items()):
        tree = ET.parse('{}/{}/xml/{}'.format(path_source, subfolder, filename))
        root = tree.getroot()

        # read the tokens, saving in a map indexed by the serializerID
        tokens_map = {int(sc.attrib['id']): sc.attrib['surface'] for sc in root.findall('tokens/token')}


        splitting_resume[filename] = {'all_words': ' '.join([value for (key, value) in tokens_map.items()]), 'semantic_frames':[]}
        # multiple frames possible for each sentence. Frames are represented by items in the interpretation list

        """
        frames = {}
        # first step look at each frame and its frame elements which portion of text it covers (min and max)
        for frame in root.findall('semantics/frameSemantics/frame'):
            frame_name = frame.attrib['name']
            # accumulator for all the tokens mentioned in the current frame
            frame_tokens_mentioned = [int(id.attrib['id']) for id in frame.findall('lexicalUnit/token')]
            elements = {}
            # compute min and max for each frame element
            for frame_element in frame.findall('frameElement'):
                element_name = frame_element.attrib['type']
                element_tokens = [int(id.attrib['id']) for id inframe_element.findall('token')]
                min_token_id, max_token_id = min(element_tokens), max(element_tokens)
                frame_tokens_mentioned.extend(element_tokens)
                elements[element_name] = {'min': min_token_id, 'max': max_token_id, 'name': element_name}
            # now at frame level
            min_token_id, max_token_id = min(frame_tokens_mentioned), max(frame_tokens_mentioned)
            frames[frame_name] = {'min': min_token_id, 'max': max_token_id, 'elements': elements, 'name': frame_name}


        # second step remove the elements that are expanded by other frames and select top level frame (tree flattening)
        # the flattened frames do not overlap
        flattened_frames = []
        current_frame = None
        for frame_name, frame in frames.items():
            overlapping = False
            # check for overlap with saved frames
            for f2_name, f2 in flattened_frames:
                if overlap(frame, f2):
                    overlapping = True
                    if covers(frame, f2):
                        smaller, bigger = f2, frame
                        flattened_frames.remove(f2)
                        flattened_frames.append(frame)
                    else:
                        smaller, bigger = frame, f2
                    bigger['top_level'] = True
                    smaller['top_level'] = False


            if not overlapping:
                flattened_frames.append(frame)
                
        
        # third step iterate over the sorted frames (2353 and 2374 were annotated in swapped order)
        """
        # where the last frame is beginning
        start_of_frame = 0
        for frame in root.findall('semantics/frameSemantics/frame'):
            intent = frame.attrib['name']
            #print('intent: ' + intent)

            # accumulator for all the tokens mentioned in the current frame
            frame_tokens_mentioned = frame.findall('lexicalUnit/token')
            slots_map = {}
            if invoke_frame_slot:
                slot_type = 'invoke_{}'.format(intent)
                for count, token in enumerate(frame_tokens_mentioned):
                    prefix = 'B' if not count else 'I'
                    iob_label = '{}-{}'.format(prefix, slot_type)
                    slots_map[int(token.attrib['id'])] = iob_label
            for frame_element in frame.findall('frameElement'):
                slot_type = frame_element.attrib['type']
                element_tokens = frame_element.findall('token')
                frame_tokens_mentioned.extend(element_tokens)
                #words += [tokens_map[id] for id in token_ids]
                for count, token in enumerate(element_tokens):
                    prefix = 'B' if not count else 'I'
                    iob_label = '{}-{}'.format(prefix, slot_type)
                    # TODO re-enable IOB
                    #iob_label = slot_type
                    slots_map[int(token.attrib['id'])] = iob_label
                    
                    #print('found a slot', iob_label, token_id)
                    #slots.append('{}-{}'.format(prefix, slot_type))
            
            # now find the part of the sentence that is related to the specific frame
            #print(frame_tokens_mentioned)
            frame_tokens_mentioned = [int(id.attrib['id']) for id in frame_tokens_mentioned]
            min_token_id, max_token_id = min(frame_tokens_mentioned), max(frame_tokens_mentioned)
            #print('min max token id', min_token_id, max_token_id)
            # the old division with min max
            #frame_tokens = {key: value for (key, value) in tokens_map.items() if int(key) >= min_token_id and int(key) <= max_token_id}
            # the correct division [0:max1],[max1:max2]. 'and' words between are part of the new frame. 'robot can you' words are part of the first frame
            frame_tokens = {key: value for (key, value) in tokens_map.items() if int(key) >= start_of_frame and int(key) <= max_token_id}
            words = [value for (key, value) in frame_tokens.items()]
            slots = [slots_map.get(key, 'O') for (key, value) in frame_tokens.items()]
            start_of_frame = max_token_id + 1
            if not len(words):
                print('len 0 for', frame_tokens_mentioned, tokens_map, frame_tokens, filename)
            #print(words)
            #print(slots)
            sample = {'words': words,
                            'intent': intent,
                            'length': len(words),
                            'slots':slots,
                            'file': filename}

            splitting_resume[filename]['semantic_frames'].append({'name': intent, 'words': ' '.join(words), 'slots': ' '.join(slots)})

            samples.append(sample)
            slot_types.update(slots)
        #print(tokens_map)
        #print('new file')

    with open('{}/resume.json'.format(path), 'w') as outfile:
        json.dump(splitting_resume, outfile, indent=2)

    # remove samples with empty word list
    samples = [s for s in samples if len(s['words'])]

    # save all data together, for alexa
    intent_values = [s['intent'] for s in samples]
    meta = {
        'tokenizer': 'whitespace',
        'language': 'en',
        'intent_types': sorted(set(intent_values)),
        'slot_types': sorted(slot_types)
    }
    with open('{}/preprocessed/all_samples.json'.format(path), 'w') as outfile:
        json.dump({
            'data': samples,
            'meta': meta
        }, outfile)

    # do the stratified split on 5 folds, fixing the random seed
    # the value of intent for each sample, necessary to perform the stratified split (keeping distribution of intents in splits)
    count_by_intent = {key:len(list(group)) for (key,group) in groupby(sorted(intent_values))}
    # remove intents that have less members than the number of splits to make StratifiedKFold work
    intent_remove = [key for (key, value) in count_by_intent.items() if value < 5]
    print('removing samples with the following intents:', intent_remove)
    samples = [s for s in samples if not s['intent'] in intent_remove]
    intent_values = [s['intent'] for s in samples]
    dataset = np.array(samples)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=dataset.size)
   
    meta = {
        'tokenizer': 'whitespace',
        'language': 'en',
        'intent_types': sorted(set(intent_values)),
        'slot_types': sorted(slot_types)
    }
    if not os.path.exists('{}/preprocessed'.format(path)):
        os.makedirs('{}/preprocessed'.format(path))

    for i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(intent_values)), intent_values)):
        #print(i, train_idx, test_idx)
        fold_data = dataset[test_idx]
        with open('{}/preprocessed/fold_{}.json'.format(path, i + 1), 'w') as outfile:
            json.dump({
                'data': fold_data.tolist(),
                'meta': meta
            }, outfile)

def alexa_prepare(path, invocation_name):
    """Creates the interaction model schema from the annotation scheme"""     
    result = { # https://developer.amazon.com/docs/smapi/interaction-model-schema.html
        'interactionModel': { # Conversational primitives for the skill
            'languageModel': {
                'invocationName': invocation_name,
                'intents': [
                    #{ 'name': 'AMAZON.FallbackIntent', 'samples': [] },
                    { 'name': 'AMAZON.CancelIntent', 'samples': [] },
                    { 'name': 'AMAZON.StopIntent', 'samples': [] },
                    { 'name': 'AMAZON.HelpIntent', 'samples': [] },
                    # {name,slots{name,type,samples},samples}
                ],
                'types': [] # custom types {name,values}
            }
            #'dialog': [], # Rules for conducting a multi-turn dialog with the user
            #'prompts': [] # Cues to the user on behalf of the skill for eliciting data or providing feedback
        }
    }

    preprocessed_location = '{}/preprocessed'.format(path)
    fold_files = os.listdir(preprocessed_location)
    # change here between fold_ or all_samples
    fold_files = sorted([f for f in fold_files if f.startswith('all_samples')])

    folds = []
    for file_name in fold_files:
        with open('{}/{}'.format(preprocessed_location, file_name)) as json_file:
            folds.append(json.load(json_file))

    all_samples = [s for fold in folds for s in fold['data']]
    meta_data = folds[0]['meta']

    sort_property = lambda sample: sample['intent']
    samples_by_intent = {intent_name: list(group) for intent_name, group in groupby(sorted(all_samples, key=sort_property), key=sort_property)}
    
    # shortcuts
    languageModel = result['interactionModel']['languageModel']
    intents = languageModel['intents']
    types = languageModel['types']

    total_slot_substitutions = defaultdict(set)

    for intent_type in meta_data['intent_types']:
        intent_slots = set()
        sample_sentences = set()
        for sample in samples_by_intent[intent_type]:
            slot_types = get_slot_types(sample['slots'])
            intent_slots.update(slot_types)
            templated_sentence, substitutions = get_templated_sentence(sample['words'], sample['slots'])
            sample_sentences.add(templated_sentence)
            for key, values in substitutions.items():
                total_slot_substitutions[key].update(values)

        
        intent_slots.discard('O')
        intent = {'name': intent_type,
            'slots':[{'name': slot, 'type': slot} for slot in intent_slots],
            'samples': list(sample_sentences)
        }
        intents.append(intent)
    
    for key, values in total_slot_substitutions.items():
        type = {'name': key, 'values': [{'name': {'value': value}} for value in values]}
        types.append(type)

    with open('{}/alexa/interactionModel.json'.format(path), 'w') as out_file:
        json.dump(result, out_file)
    

def get_slot_types(iob_list):
    result = []
    for iob_label in iob_list:
        parts = iob_label.split('-')
        if len(parts) > 1:
            # 'B-something' or 'I-something
            result.append(parts[1])
        else:
            # 'O'
            result.append(parts[0])
    return result

def get_templated_sentence(words, iob_list):
    """Returns a sentence with 'slot values' replaced by '{types}' and a dict {type:['slot values']}"""
    last_slot = None
    result_sentence = ''
    last_slot_surface = None
    result_dict = defaultdict(list)
    for idx, iob in enumerate(iob_list):
        iob_parts = iob.split('-')
        slot_type = iob_parts[1] if len(iob_parts) > 1 else None
        if iob == 'O':
            result_sentence += ' ' + words[idx]
            if last_slot_surface:
                result_dict[last_slot].append(sentence_fix(last_slot_surface))
                last_slot_surface = None
        elif iob.startswith('B-'):
            result_sentence += ' {' + slot_type + '}'
            if last_slot_surface:
                result_dict[last_slot].append(sentence_fix(last_slot_surface))
            last_slot_surface = words[idx]
        else:
            last_slot_surface += ' ' + words[idx]

        last_slot = slot_type

    if last_slot_surface:
        result_dict[last_slot].append(sentence_fix(last_slot_surface))

    return sentence_fix(result_sentence), result_dict

def sentence_fix(sentence):
    sentence = re.sub('\'s', 'is', sentence)
    sentence =re.sub('^\s+', '', sentence)
    return sentence



def main():
    #nlp_en = load_nlp()
    #nlp_it = load_nlp('it')
    which = os.environ.get('DATASET', None)
    print(which)
    
    if which == 'huric_eb':
        huric_preprocess_eb('huric_eb', None)
    elif which == 'huric':
        huric_preprocess('huric', False)

    alexa_prepare('huric', 'office robot')


if __name__ == '__main__':
    main()