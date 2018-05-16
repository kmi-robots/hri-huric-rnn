"""
This module preprocesses the datasets in equivalent formats
"""
import json
import os
import re
import numpy as np
import spacy
import csv
import zipfile
from itertools import groupby
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import xml.etree.ElementTree as ET

n_folds = 3
ALEXA_FILE_NAME = 'alexaInteractionModel.json'
LEX_FILE_NAME = 'lexBot.json'
LEX_ZIP_NAME = 'lexBot.zip'


def huric_preprocess(path, subfolder, invoke_frame_slot=False):
    """Preprocess the huric dataset, provided by Danilo Croce
    invoke_frame_slot adds a slot fot the lexical unit of invocation of the frame
    """

    def overlap(a, b):
        return max(0, min(a['max'], b['max']) - max(a['min'], b['min']))

    def covers(a, b):
        return (a['max'] >= b['max']) and (a['min'] <= b['min'])

    path_source = path + '/source/' + subfolder

    samples = []
    spatial_samples = []
    # a resume of how sentences are split into frames
    splitting_resume = {}
    slot_types = set()
    spatial_slot_types = set()

    """ subfolder is a parameter of this function
    # retrieve all the possible xml files in the three subfolders
    for subfolder in ['GrammarGenerated', 'S4R_Experiment', 'Robocup']:
        for filename in os.listdir('{}/{}/xml'.format(path_source, subfolder)):
            file_locations[filename] = subfolder
    """
    files_list = os.listdir('{}/xml'.format(path_source))

    # TODO remove this
    for forbidden in ['2307.xml', '2353.xml', '2374.xml', '2377.xml', '2411.xml', '3395.xml', '3281.xml', '3283.xml', '3284.xml']:
        if forbidden in files_list:
            files_list.remove(forbidden)
    print('#files: ', len(files_list))

    for filename in sorted(files_list):
        tree = ET.parse('{}/xml/{}'.format(path_source, filename))
        root = tree.getroot()

        # read the tokens, saving in a map indexed by the id
        tokens_map = {int(sc.attrib['id']): sc.attrib['surface']
                      for sc in root.findall('tokens/token')}

        splitting_resume[filename] = {'all_words': ' '.join(
            [value for (key, value) in tokens_map.items()]), 'semantic_frames': []}

        # remember where the last semantic frame is beginning
        start_of_frame = 0
        
        slots_map = {}

        # first loop: over the semantic frames
        for frame in root.findall('semantics/frameSemantics/frame'):
            intent = frame.attrib['name']

            # accumulator for all the tokens mentioned in the current semantic frame
            frame_tokens_mentioned = frame.findall('lexicalUnit/token')
            if invoke_frame_slot:
                slot_type = 'invoke_{}'.format(intent)
                for count, token in enumerate(frame_tokens_mentioned):
                    prefix = 'B' if not count else 'I'
                    iob_label = '{}-{}'.format(prefix, slot_type)
                    slots_map[int(token.attrib['id'])] = {'iob_label': iob_label, 'slot_type': slot_type}
            for frame_element in frame.findall('frameElement'):
                slot_type = frame_element.attrib['type']
                element_tokens = frame_element.findall('token')
                frame_tokens_mentioned.extend(element_tokens)
                for count, token in enumerate(element_tokens):
                    prefix = 'B' if not count else 'I'
                    iob_label = '{}-{}'.format(prefix, slot_type)
                    slots_map[int(token.attrib['id'])] = {'iob_label': iob_label, 'slot_type': slot_type}

            # now find the part of the sentence that is related to the specific frame
            frame_tokens_mentioned = [int(id.attrib['id']) for id in frame_tokens_mentioned]
            min_token_id, max_token_id = min(frame_tokens_mentioned), max(frame_tokens_mentioned)
            # the correct division [0:max1],[max1:max2]. 'and' words between are part of the new frame. 'robot can you' words are part of the first frame
            frame_tokens = {key: value for (key, value) in tokens_map.items() if int(key) >= start_of_frame and int(key) <= max_token_id}
            words = [value for (key, value) in frame_tokens.items()]
            slots_objects = [slots_map.get(key, {'iob_label': 'O'}) for (key, value) in frame_tokens.items()]
            slots = [slot['iob_label'] for slot in slots_objects]
            start_of_frame = max_token_id + 1
            if not len(words):
                print('WARNING: len 0 for', frame_tokens_mentioned, tokens_map, frame_tokens, filename)
            sample = {'words': words,
                      'intent': intent,
                      'length': len(words),
                      'slots': slots,
                      'file': filename}

            splitting_resume[filename]['semantic_frames'].append(
                {'name': intent, 'words': ' '.join(words), 'slots': ' '.join(slots)})

            samples.append(sample)
            slot_types.update(slots)
        
        tokenId_by_slotType = defaultdict(list)
        for id, slot in slots_map.items():
            tokenId_by_slotType[slot['slot_type']].append(id)

        spatial_slots = {}
        # second loop: over the spatial frames
        for frame in root.findall('semantics/spatialSemantics/spatialRelation'):
            spatial_frame_name = frame.attrib['name']
            frame_tokens_mentioned = []

            # accumulator for all the tokens mentioned in the current spatial frame
            frame_tokens = []
            for frame_element in frame.findall('spatialRole'):
                slot_type = frame_element.attrib['type']
                element_tokens = frame_element.findall('token')
                frame_tokens_mentioned.extend(element_tokens)
                for count, token in enumerate(element_tokens):
                    prefix = 'B' if not count else 'I'
                    iob_label = '{}-{}'.format(prefix, slot_type)
                    # to remove IOB prefix, set iob_label = slot_type
                    spatial_slots[int(token.attrib['id'])] = iob_label

            # now find the part of the sentence that is related to the specific frame
            frame_tokens_mentioned = [int(id.attrib['id']) for id in frame_tokens_mentioned]
            min_token_id, max_token_id = min(frame_tokens_mentioned), max(frame_tokens_mentioned)
            frame_element_min, frame_element_max = slots_map[min_token_id]['slot_type'], slots_map[max_token_id]['slot_type']
            if frame_element_min != frame_element_max:
                print('Something is wrong: spatial frame is not contained in a single frame element: ', frame_tokens_mentioned, filename)
            
            parent_start, parent_end = tokenId_by_slotType[frame_element_min][0], tokenId_by_slotType[frame_element_max][-1]
            frame_tokens = {key: value for (key, value) in tokens_map.items() if int(key) >= parent_start and int(key) <= parent_end}
            words = [value for (key, value) in frame_tokens.items()]
            slots = [spatial_slots.get(key, 'O') for (key, value) in frame_tokens.items()]
            if not len(words):
                print('WARNING: len 0 for', frame_tokens_mentioned, tokens_map, frame_tokens, filename)
            sample = {'words': words,
                      'intent': spatial_frame_name,
                      'length': len(words),
                      'slots': slots,
                      'file': filename}


            spatial_samples.append(sample)
            spatial_slot_types.update(slots)

    write_json(path, 'resume.json', splitting_resume)

    # remove samples with empty word list
    samples = [s for s in samples if len(s['words'])]

    out_path = '{}/{}'.format(path, subfolder)
    out_path_preprocessed = '{}/preprocessed'.format(out_path)

    # save all data together, for alexa
    intent_values = [s['intent'] for s in samples]
    meta = {
        'tokenizer': 'whitespace',
        'language': 'en',
        'intent_types': sorted(set(intent_values)),
        'slot_types': sorted(slot_types)
    }
    result_all = {
            'data': samples,
            'meta': meta
        }
    
    write_json(out_path_preprocessed, 'all_samples.json', result_all)

    # save also spatial submodel
    spatial_out_path = '{}/spatial/preprocessed'.format(out_path)
    spatial_intent_values = [s['intent'] for s in spatial_samples]
    spatial_meta = {
        'tokenizer': 'whitespace',
        'language': 'en',
        'intent_types': sorted(set(spatial_intent_values)),
        'slot_types': sorted(spatial_slot_types)
    }
    spatial_result = {
        'data': spatial_samples,
        'meta': spatial_meta
    }
    write_json(spatial_out_path, 'all_samples.json', spatial_result)

    # do the stratified split on k folds, fixing the random seed
    # the value of intent for each sample, necessary to perform the stratified
    # split (keeping distribution of intents in splits)
    count_by_intent = {key: len(list(group)) for (
        key, group) in groupby(sorted(intent_values))}
    # remove intents that have less members than the number of splits to make
    # StratifiedKFold work
    intent_remove = [
        key for (key, value) in count_by_intent.items() if value < n_folds]
    print('removing samples with the following intents:', intent_remove)
    samples = [s for s in samples if not s['intent'] in intent_remove]
    intent_values = [s['intent'] for s in samples]
    dataset = np.array(samples)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=dataset.size)

    meta = {
        'tokenizer': 'whitespace',
        'language': 'en',
        'intent_types': sorted(set(intent_values)),
        'slot_types': sorted(slot_types)
    }

    for i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(intent_values)), intent_values)):
        #print(i, train_idx, test_idx)
        fold_data = dataset[test_idx]
        content = {
            'data': fold_data.tolist(),
            'meta': meta
        }
        write_json(out_path_preprocessed, 'fold_{}.json'.format(i + 1), content)

    return result_all, spatial_result


def alexa_prepare(path, invocation_name):
    """Creates the interaction model schema from the annotation scheme"""
    result = {  # https://developer.amazon.com/docs/smapi/interaction-model-schema.html
        'interactionModel': {  # Conversational primitives for the skill
            'languageModel': {
                'invocationName': invocation_name,
                'intents': [
                    #{ 'name': 'AMAZON.FallbackIntent', 'samples': [] },
                    {'name': 'AMAZON.CancelIntent', 'samples': []},
                    {'name': 'AMAZON.StopIntent', 'samples': []},
                    {'name': 'AMAZON.HelpIntent', 'samples': []},
                    # {name,slots{name,type,samples},samples}
                ],
                'types': []  # custom types {name,values}
            }
            #'dialog': [], # Rules for conducting a multi-turn dialog with the user
            #'prompts': [] # Cues to the user on behalf of the skill for eliciting data or providing feedback
        }
    }

    preprocessed_location = '{}/preprocessed'.format(path)
    with open('{}/{}'.format(preprocessed_location, 'all_samples.json')) as json_file:
        file_content = json.load(json_file)

    all_samples = file_content['data']
    meta_data = file_content['meta']

    #print(all_samples)

    def sort_property(sample): return sample['intent']
    samples_by_intent = {intent_name: list(group) for intent_name, group in groupby(
        sorted(all_samples, key=sort_property), key=sort_property)}

    # shortcuts
    languageModel = result['interactionModel']['languageModel']
    intents = []
    types = []

    total_slot_substitutions = defaultdict(set)

    for intent_type in meta_data['intent_types']:
        intent_slots = set()
        sample_sentences = set()
        for sample in samples_by_intent[intent_type]:
            slot_types = get_slot_types(sample['slots'])
            intent_slots.update(slot_types)
            templated_sentence, substitutions = get_templated_sentence(
                sample['words'], sample['slots'])
            sample_sentences.add(templated_sentence)
            for key, values in substitutions.items():
                total_slot_substitutions[key].update(values)

        intent_slots.discard('O')
        intent = {'name': intent_type,
                  'slots': sorted([{'name': slot, 'type': slot} for slot in intent_slots], key= lambda k: k['name']),
                  'samples': sorted(sample_sentences)
                  }
        intents.append(intent)

    for key, values in total_slot_substitutions.items():
        type = {'name': key, 'values': sorted([
            {'name': {'value': value}} for value in values], key=lambda k: k['name']['value'])}
        types.append(type)

    languageModel['intents'] = sorted(intents, key=lambda k: k['name'])
    languageModel['types'] = sorted(types, key=lambda k: k['name'])

    write_json(path, ALEXA_FILE_NAME, result)


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
    sentence = re.sub('^\s+', '', sentence)
    return sentence


def lex_from_alexa(path, bot_name):
    """Builds the .zip file to be uploaded directly on lex portal"""
    
    with open('{}/{}'.format(path, ALEXA_FILE_NAME)) as json_file:
        alexa_content = json.load(json_file)

    intents = alexa_content['interactionModel']['languageModel']['intents']
    intents = [intent for intent in intents if not intent['name'].startswith('AMAZON.')]
    # the 'look {Phenomenon}' appears both in 'Perception_active' and in 'Searching' intent
    intents = [intent for intent in intents if intent['name'] != 'Perception_active']
    for intent in intents:
        intent['version'] = '$LATEST'
        intent['fulfillmentActivity'] = {'type': 'ReturnIntent'}
        intent['sampleUtterances'] = intent.pop('samples')
        for slot in intent['slots']:
            slot['slotConstraint'] = 'Optional'
            slot['slotType'] = slot.pop('type')

    types = alexa_content['interactionModel']['languageModel']['types']
    for slotType in types:
        slotType['enumerationValues'] = [value['name'] for value in slotType.pop('values')]
    
    lex_content = {
        'metadata': {
            'schemaVersion': '1.0',
            'importType': 'LEX',
            'importFormat': 'JSON'
        },
        'resource': {
            'name': bot_name,  # ^([A-Za-z]_?)+$ , no spaces
            'version': '$LATEST',
            'intents': intents,  # list of intents
            'slotTypes': types,  # list of slots
            'voiceId': 'Salli',  # which voice, for TTS service
            'childDirected': False,  # targeted to child use
            'locale': 'en-US',
            # The maximum time in seconds that Amazon Lex retains the data
            # gathered in a conversation
            'idleSessionTTLInSeconds': 300,
            #'description': 'bot description',
            'clarificationPrompt': {
                'messages': [
                    {
                        'contentType': 'PlainText',
                        'content': 'Sorry, can you please rephrase that?'
                    }
                ],
                'maxAttempts': 5  # max number of reprompts
            },
            'abortStatement': {
                'messages': [
                    {
                        'contentType': 'PlainText',
                        'content': 'Sorry, I could not understand. Goodbye.'
                    }
                ]
            }
        }
    }

    write_json(path, LEX_FILE_NAME, lex_content)

    # create zip
    zf = zipfile.ZipFile('{}/{}'.format(path, LEX_ZIP_NAME), "w")
    zf.write('{}/{}'.format(path, LEX_FILE_NAME), LEX_FILE_NAME)
    zf.close()


def combine_and_save(results, path):
    if not results:
        raise ValueError('no results passed')
    
    samples = []
    slot_types = []
    intent_types = []
    meta = results[0]['meta']
    for result in results:
        samples += list(result['data'])
        slot_types += list(result['meta']['slot_types'])
        intent_types += list(result['meta']['intent_types'])
    merged = {
        'data': list({' '.join(s['words']): s for s in samples}.values()),
        'meta': {
            'tokenizer': meta['tokenizer'],
            'language': meta['language'],
            'intent_types': sorted(set(intent_types)),
            'slot_types': sorted(set(slot_types))
        }
    }

    out_path = '{}/preprocessed'.format(path)
    print(out_path)

    write_json(out_path, 'all_samples.json', merged)

def write_json(out_path, file_name, serializable_content):
    """Creates a file with `filename` on the `path` (checked for existence or created)
    with the following json content"""
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open('{}/{}'.format(out_path, file_name), 'w') as outfile:
        json.dump(serializable_content, outfile, indent=2)

def main():
    #nlp_en = load_nlp()
    #nlp_it = load_nlp('it')
    which = os.environ.get('DATASET', None)
    print(which)

    if which == 'huric':
        subfolder_results = []
        subfolder_spatial_results = []
        for subfolder in ['GrammarGenerated', 'S4R_Experiment', 'Robocup']:
            res, spatial_res = huric_preprocess('huric', subfolder, False)
            subfolder_results.append(res)
            subfolder_spatial_results.append(spatial_res)
            alexa_prepare('huric/{}'.format(subfolder), 'office robot {}'.format(subfolder))
            alexa_prepare('huric/{}/spatial'.format(subfolder), 'office robot spatial {}'.format(subfolder))
            lex_from_alexa('huric/{}/'.format(subfolder), 'kmi_{}'.format(subfolder))
            lex_from_alexa('huric/{}/spatial'.format(subfolder), 'spatial_{}'.format(subfolder))
        
        combine_and_save(subfolder_results, 'huric/combined')
        combine_and_save(subfolder_spatial_results, 'huric/combined/spatial')
        alexa_prepare('huric/combined', 'office robot')
        alexa_prepare('huric/combined/spatial', 'office robot spatial')
        lex_from_alexa('huric/combined', 'kmi')
        lex_from_alexa('huric/combined/spatial', 'spatial')


if __name__ == '__main__':
    main()
