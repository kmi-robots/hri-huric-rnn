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
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import xml.etree.ElementTree as ET
from xml.dom import minidom

import spacy

from nlunetwork.embeddings import WhitespaceTokenizer

nlp = spacy.load('en')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

n_folds = 5
#ALEXA_FILE_NAME = 'alexaInteractionModel.json'
#LEX_FILE_NAME = 'lexBot.json'
#LEX_ZIP_NAME = 'lexBot.zip'


def huric_preprocess(path, trim='right', invoke_frame_slot=False):
    """Preprocess the huric dataset
    trim regulates how to split the sentences:
    - 'right' finds the last reference on the right to tokens, while on the left takes up to the previous frame in the sentence
    - 'both' finds the minimum and maximum mentioned token ids, producing a shorter sentence in output
    invoke_frame_slot adds a slot fot the lexical unit of invocation of the frame
    """

    def overlap(a, b):
        return max(0, min(a['max'], b['max']) - max(a['min'], b['min']))

    def covers(a, b):
        return (a['max'] >= b['max']) and (a['min'] <= b['min'])

    path_source = path + '/source'

    samples = []
    spatial_samples = []
    # a resume of how sentences are split into frames
    splitting_resume = {}
    slot_types = set()
    spatial_slot_types = set()

    files_list = os.listdir(path_source)

    # TODO remove this
    # 2307: nested frame (no problem, discarded) 
    # 2374: frames swapped
    # 2377: nested frame (no problem, discarded)
    # 2411: nested frame (no problem, discarded)
    # 3395: nested frame (no problem, discarded)
    # 3281, 3283, 3284: spatial problem
    # 
    print('#files: ', len(files_list))

    for file_name in sorted(files_list):
        file_location = '{}/{}'.format(path_source, file_name)
        #print(file_location)
        with open(file_location) as file_in:
            tree = ET.parse(file_in)
        
        root = tree.getroot()

        # read the tokens, saving in a map indexed by the id
        tokens_map = {int(sc.attrib['id']): sc.attrib['surface']
                      for sc in root.findall('tokens/token')}

        splitting_resume[file_name] = {'all_words': ' '.join(
            [value for (key, value) in tokens_map.items()]), 'semantic_frames': []}

        # remember where the last semantic frame is beginning
        start_of_frame = 1
        
        slots_map = {}

        # first loop: over the semantic frames
        semantic_frames = root.findall('semantics/frameSemantics/frame')
        semantic_frames = sorted(semantic_frames, key=lambda f: int(f.findall('lexicalUnit/token')[0].attrib['id']))
        for frame in semantic_frames:
            intent = frame.attrib['name']
            lexical_unit_ids = [int(t.attrib['id']) for t in frame.findall('lexicalUnit/token')]

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
            if trim == 'right':
                # start considering from the end of the previous frame
                start_considering = start_of_frame
            elif trim == 'both':
                # start considering from the minimum mentioned token
                start_considering = min_token_id
                # recompute the lexical unit ids
                lexical_unit_ids = [l - start_considering + 1 for l in lexical_unit_ids]
            else:
                raise ValueError('trim' + str(trim))
            frame_tokens = {key: value for (key, value) in tokens_map.items() if int(key) >= start_considering and int(key) <= max_token_id}
            words = [value for (key, value) in frame_tokens.items()]
            slots_objects = [slots_map.get(key, {'iob_label': 'O'}) for (key, value) in frame_tokens.items()]
            slots = [slot['iob_label'] for slot in slots_objects]
            if not len(words):
                print('WARNING: len 0 for', intent, frame_tokens_mentioned, tokens_map, 'expected in [{},{}]'.format(start_of_frame, max_token_id) , file_name)
            sample = {
                'words': words,
                'intent': intent,
                'length': len(words),
                'slots': slots,
                'file': file_name,
                'start_token_id': start_of_frame,
                'end_token_id': max_token_id,
                'id': len(samples),
                'lexical_unit_ids': lexical_unit_ids
            }
            start_of_frame = max_token_id + 1

            splitting_resume[file_name]['semantic_frames'].append(
                {'name': intent, 'words': ' '.join(words), 'slots': ' '.join(slots)})

            samples.append(sample)
            slot_types.update(slots)
        
        tokenId_by_slotType = defaultdict(list)
        for id, slot in slots_map.items():
            tokenId_by_slotType[slot['slot_type']].append(id)

        spatial_slots = {}
        # second loop: over the spatial frames
        for frame in root.findall('semantics/spatialSemantics/spatialRelation'):
            # problems in spatial frames for these files
            if file_name in ['3281.xml', '3283.xml', '3284.xml']:
                continue
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
                print('INFO: spatial relation spans multiple frame elements: ', frame_tokens_mentioned, file_name)
            
            parent_start, parent_end = tokenId_by_slotType[frame_element_min][0], tokenId_by_slotType[frame_element_max][-1]
            frame_tokens = {key: value for (key, value) in tokens_map.items() if int(key) >= parent_start and int(key) <= parent_end}
            words = [value for (key, value) in frame_tokens.items()]
            slots = [spatial_slots.get(key, 'O') for (key, value) in frame_tokens.items()]
            if not len(words):
                print('WARNING: len 0 for', spatial_frame_name, frame_tokens_mentioned, tokens_map, 'expected in [{},{}]'.format(parent_start, parent_end) , file_name, ' --> frame will be discarded')
            sample = {
                'words': words,
                'intent': spatial_frame_name,
                'length': len(words),
                'slots': slots,
                'file': file_name,
                'start_token_id': parent_start,
                'end_token_id': parent_end,
                'id': len(spatial_samples)
            }


            spatial_samples.append(sample)
            spatial_slot_types.update(slots)

    #write_json(path, 'resume.json', splitting_resume)

    # remove samples with empty word list
    samples = [s for s in samples if len(s['words'])]

    out_path = '{}_{}'.format(path, trim)
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
    print('removing samples with the following intents:', intent_remove, 'reason: #samples < #folds')
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

    folds = []
    for i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(intent_values)), intent_values)):
        #print(i, train_idx, test_idx)
        fold_data = dataset[test_idx]
        folds.append(fold_data)
        content = {
            'data': fold_data.tolist(),
            'meta': meta
        }
        write_json(out_path_preprocessed, 'fold_{}.json'.format(i + 1), content)

    # save also the 4 folds of final train for lex evaluation
    train_data = folds[:-1]
    train_data = [s for f in train_data for s in f]
    result_train = {
            'data': train_data,
            'meta': meta
        }
    
    write_json(out_path_preprocessed, 'train_samples.json', result_train)

    return result_all, spatial_result


def alexa_prepare(path, invocation_name, input_file_name='all_samples.json', alexa_file_name='alexaInteractionModel.json'):
    """Creates the interaction model schema from the annotation scheme"""
    result = {  # https://developer.amazon.com/docs/smapi/interaction-model-schema.html
        'interactionModel': {  # Conversational primitives for the skill
            'languageModel': {
                'invocationName': invocation_name,
                'intents': [],
                'types': []  # custom types {name,values}
            }
            #'dialog': [], # Rules for conducting a multi-turn dialog with the user
            #'prompts': [] # Cues to the user on behalf of the skill for eliciting data or providing feedback
        }
    }

    preprocessed_location = '{}/preprocessed'.format(path)
    with open('{}/{}'.format(preprocessed_location, input_file_name)) as json_file:
        file_content = json.load(json_file)

    all_samples = file_content['data']
    meta_data = file_content['meta']

    #print(all_samples)

    def sort_property(sample): return sample['intent']
    samples_by_intent = {intent_name: list(group) for intent_name, group in groupby(
        sorted(all_samples, key=sort_property), key=sort_property)}

    # shortcuts
    languageModel = result['interactionModel']['languageModel']
    intents = [
        #{ 'name': 'AMAZON.FallbackIntent', 'samples': [] },
        {'name': 'AMAZON.CancelIntent', 'samples': []},
        {'name': 'AMAZON.StopIntent', 'samples': []},
        {'name': 'AMAZON.HelpIntent', 'samples': []},
        # {name,slots{name,type,samples},samples}
    ]
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

    write_json('{}/amazon'.format(path), alexa_file_name, result)


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
    # remove spaces before apostrophes
    sentence = re.sub('\s\'(\S)', r"'\1", sentence)
    # this amazon thing does not want samples like "{Item}'s"
    sentence = re.sub('}\'(\S)', "}", sentence)
    sentence = re.sub('^\s+', '', sentence)
    return sentence


def lex_from_alexa(path, bot_name, alexa_file_name='alexaInteractionModel.json', lex_file_name='lexBot.json'):
    """Builds the .zip file to be uploaded directly on lex portal"""
    
    with open('{}/{}'.format(path, alexa_file_name)) as json_file:
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

    write_json(path, lex_file_name, lex_content)

    # create zip
    zf = zipfile.ZipFile('{}/{}'.format(path, lex_file_name + '.zip'), "w")
    zf.write('{}/{}'.format(path, lex_file_name), lex_file_name)
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
    """Creates a file with `file_name` on the `path` (checked for existence or created)
    with the following json content"""
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open('{}/{}'.format(out_path, file_name), 'w') as outfile:
        json.dump(serializable_content, outfile, indent=2)

def modernize_huric_xml(source_path, dest_path):
    """Translates from the old schema to the new one"""

    def get_new_id(old_id):
        return old_id.split('.')[3]
    
    def get_constituents(string):
        if not string:
            return []
        return [get_new_id(old_id) for old_id in string.split(' ')]
    
    files_list = os.listdir('{}'.format(source_path))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    for file_name in sorted(files_list):
        src_tree = ET.parse('{}/{}'.format(source_path, file_name))
        src_root = src_tree.getroot()
        xdg = src_root.find('PARAGRAPHS/P/XDGS/XDG')
        command_id = xdg.attrib['oldID']
        tokens = [{
            'id': get_new_id(src_token.attrib['serializerID']),
            'surface':src_token.attrib['surface'],
            'pos': src_token.attrib['sctype'],
            'lemma': src_token.find('LEMMAS/LM').attrib['surface']
        } for src_token in xdg.findall('CSTS/SC')]
        sentence = ' '.join([t['surface'] for t in tokens])
        dependencies = [{
            'from': src_dep.attrib['fromId'],
            'to': src_dep.attrib['toId'],
            'type': src_dep.attrib['type']
        } for src_dep in xdg.findall('ICDS/ICD')]
        frames = [{
            'name': src_frame.attrib['name'],
            'lexical_units': get_constituents(src_frame.find('constituentList').text),
            'frame_elements': [{
                'type': old_arg.attrib['argumentType'],
                'tokens': get_constituents(old_arg.find('constituentList').text)
            } for old_arg in src_frame.findall('ARGS/sem_arg')]
        } for src_frame in xdg.findall('interpretations/interpretationList/item')]
        # now generate the new xml
        new_command = ET.Element('command', {'id': command_id})
        ET.SubElement(new_command, 'sentence').text = sentence
        new_tokens = ET.SubElement(new_command, 'tokens')
        for token in tokens:
            ET.SubElement(new_tokens, 'token', token)
        new_dependencies = ET.SubElement(new_command, 'dependencies')
        for dependency in dependencies:
            ET.SubElement(new_dependencies, 'dep', dependency)
        new_semantics = ET.SubElement(new_command, 'semantics')
        new_semantic_frames = ET.SubElement(new_semantics, 'frameSemantics')
        new_spatial_frames = ET.SubElement(new_semantics, 'spatialSemantics')
        for frame in frames:
            if frame['name'] == 'Spatial_relation':
                frame_father = new_spatial_frames
                new_frame_name = 'spatialRelation'
                new_frame_element_name = 'spatialRole'
            else:
                frame_father = new_semantic_frames
                new_frame_name = 'frame'
                new_frame_element_name = 'frameElement'
            new_frame = ET.SubElement(frame_father, new_frame_name, {'name': frame['name']})
            new_lexical_unit = ET.SubElement(new_frame, 'lexicalUnit')
            for token_id in frame['lexical_units']:
                ET.SubElement(new_lexical_unit, 'token', {'id': token_id})
            for frame_element in frame['frame_elements']:
                new_frame_element = ET.SubElement(new_frame, new_frame_element_name, {'type': frame_element['type']})
                for token_id in frame_element['tokens']:
                    ET.SubElement(new_frame_element, 'token', {'id': token_id})
        # generate pretty XML
        pretty_string = minidom.parseString(ET.tostring(new_command, 'utf-8')).toprettyxml(encoding='utf-8').decode('utf-8')
        with open('{}/{}'.format(dest_path, file_name), 'w') as out_file:
            out_file.write(pretty_string)

        #tree = ET.ElementTree(new_command)
        #with open('{}/{}'.format(dest_path, file_name), 'w', encoding='utf-8') as out_file:
        #    tree.write(out_file, encoding='unicode')


def framenet_preprocess(folder, dest_path, subset=True):
    """Preprocesses the framenet corpus, taking only the frames from the HuRIC set. Produces in output a HuRIC formatted dataset on out_path"""
    # from framenet names to huric names
    frame_names_mappings = {
        'Attaching': 'Attaching',
        'Being_in_category': 'Being_in_category',
        'Being_located': 'Being_located',
        'Bringing': 'Bringing',
        'Change_operational_state': 'Change_operational_state',
        'Closure': 'Closure',
        'Arriving': 'Entering',
        'Cotheme': 'Following',
        'Giving': 'Giving',
        'Inspecting': 'Inspecting',
        'Motion': 'Motion',
        'Perception_active': 'Perception_active',
        'Placing': 'Placing',
        'Releasing': 'Releasing',
        'Scrutiny': 'Searching',
        'Taking': 'Taking'
    }
    ns = {'ns': '{http://framenet.icsi.berkeley.edu}'}

    if not subset:
        # TODO generalize to not subset
        raise ValueError('subset must be True')
    xml_path = Path(folder) / 'source/fulltext'
    xml_files_paths = [f for f in xml_path.iterdir() if f.is_file()]
    sentences = []
    for f in xml_files_paths:
        with open(f) as file_in:
            xmlstring = file_in.read()
        # remove namespace that makes ET usage cumbersome
        xmlstring = re.sub(r'\sxmlns="[^"]+"', '', xmlstring, count=1)
        root = ET.fromstring(xmlstring)
        #print(root)
        #print([el for el in root])
        # get the sentences elements
        sentences.extend(root.findall('sentence'))
    print('#sentences:', len(sentences))
    get_wanted_frames = lambda sentence: [f for f in sentence.findall('annotationSet') if f.attrib.get('frameName', None) in frame_names_mappings and f.attrib['status'] == 'MANUAL']
    # select the sentences that have an interesting frame name
    sentences = [s for s in sentences if len(get_wanted_frames(s))]
    print('#sentences_sel_frames:', len(sentences))
    #exit(1)
    samples = [{
        'words': s.find('text').text.split(),
        'intent_cnt': len(get_wanted_frames(s)),
        'intents': [f.attrib.get('frameName', None) for f in get_wanted_frames(s)],
    } for s in sentences]

    cnt = 0
    for s in samples:
        if s['intent_cnt'] > 1:
            print(s['intents'], ' '.join(s['words']))
            cnt+=1
    print('#multiple:', cnt)
    # TODO transform to HuRIC
    # - get the required attributes on the tree
    # - translate the frame names
    # - produce output files
    for s in sentences:
        command_id = s.attrib['ID']
        text = s.find('text').text
        words = text.split()
        try:
            doc = nlp(text)
        except AssertionError as e:
            print(command_id, text, e)
            raise e
        framenet_annotations = s.find("annotationSet[@status='UNANN']/layer").findall("label")
        #print([t.text for t in doc], words)
        try:
            assert len(doc) == len(framenet_annotations)
        except AssertionError as e:
            print('different lengths', len(doc), len(framenet_annotations), ' for', text)
            #raise e
        # FrameNet annotations sometimes are missing some tokens. See frame 312908 for example
        tokens = [{
            'id': t.i + 1,
            'start': t.idx,
            'end': t.idx + len(t.text),
            'lemma': t.lemma_,
            'pos': t.tag_, # or l.attrib['name']
            'surface': t.text
        } for t in doc]
        # TODO where are dependencies?
        dependencies = [{
            'from': str(t.head.i + 1) if t.dep_.lower() != 'root' else str(0),
            'to': str(t.i + 1),
            'type': t.dep_.lower()
        } for t in doc]

        new_command = ET.Element('command', {'id': command_id})
        ET.SubElement(new_command, 'sentence').text = text
        new_tokens = ET.SubElement(new_command, 'tokens')
        for t in tokens:
            ET.SubElement(new_tokens, 'token', {'id': str(t['id']), 'lemma': t['lemma'], 'pos': t['pos'], 'surface': t['surface']})
        new_dependencies = ET.SubElement(new_command, 'dependencies')
        for d in dependencies:
            ET.SubElement(new_dependencies, 'dep', d)
        new_semantic_frames = ET.SubElement(ET.SubElement(new_command, 'semantics'), 'frameSemantics')

        wanted_frames = get_wanted_frames(s)
        for f in wanted_frames:
            frame_name = frame_names_mappings[f.attrib['frameName']]
            lexical_unit = f.find("layer[@name='Target']/label")
            start, end = int(lexical_unit.attrib['start']), int(lexical_unit.attrib['end'])
            lexical_unit_ids = [t['id'] for t in tokens if (t['end']>=start and t['start']<=end)]
            fes = f.findall("layer[@name='FE']/label")
            #print(command_id, fes)
            frame_elements = [{'type': f.attrib['name'], 'ids': [t['id'] for t in tokens if (f.attrib.get('start', None) != None and t['end']>=int(f.attrib['start']) and t['start']<=int(f.attrib['end'])+1)]} for f in fes]

            new_frame = ET.SubElement(new_semantic_frames, 'frame', {'name': frame_name})
            new_lexical_unit = ET.SubElement(new_frame, 'lexicalUnit')
            for l in lexical_unit_ids:
                ET.SubElement(new_lexical_unit, 'token', {'id': str(l)})
            for fe in frame_elements:
                if len(fe['ids']):
                    new_frame_element = ET.SubElement(new_frame, 'frameElement', {'type': fe['type']})
                    for t_id in fe['ids']:
                        ET.SubElement(new_frame_element, 'token', {'id': str(t_id)})
        
        pretty_string = minidom.parseString(ET.tostring(new_command, 'utf-8')).toprettyxml(encoding='utf-8').decode('utf-8')
        with open('{}/{}.xml'.format(dest_path, command_id), 'w') as out_file:
            out_file.write(pretty_string)


def main():
    #nlp_en = load_nlp()
    #nlp_it = load_nlp('it')
    which = os.environ.get('DATASET', 'huric_eb')
    print(which)

    if which == 'huric_eb':
        modernize_huric_xml('huric_eb/source', 'huric_eb/modern/source')
        res, spatial_res = huric_preprocess('huric_eb/modern')
        alexa_prepare('huric_eb/modern', 'roo bot')
        alexa_prepare('huric_eb/modern/spatial', 'office robot spatial')
        lex_from_alexa('huric_eb/modern/amazon', 'kmi_EB')
        lex_from_alexa('huric_eb/modern/spatial/amazon', 'spatial_EB')
        # for lex evaluation
        alexa_prepare('huric_eb/modern', 'roo bot train', 'train_samples.json', 'alexa_train.json')
        lex_from_alexa('huric_eb/modern/amazon', 'train_only', 'alexa_train.json', 'lexTrainBot.json')


    elif which == 'framenet_subset':
        framenet_preprocess('framenet', 'framenet/subset/source', True)
        huric_preprocess('framenet/subset')

    elif which == 'framenet_subset_short':
        framenet_preprocess('framenet', 'framenet/subset/source', True)
        huric_preprocess('framenet/subset', trim='both')

if __name__ == '__main__':
    main()
