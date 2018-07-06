import json
import os
from itertools import groupby
from pathlib import Path

# The tables file is excluded for versioning
# TODO remove sensitive attributes and add to git
from tables import speakers, audio_files, XDG

# speakers table indexes
S_NATIONALITY_COLUMN = 2
S_UID_COLUMN = 1
S_PROFICIENCY_COLUMN = 4
S_NATIVE_COLUMN = 3
# audio table indexes
A_SPEAKER_UID_COLUMN = 1
A_SID_COLUMN = 0
# XDG indexes
X_ID_COLUMN = 0
X_SID_COLUMN = 1

MY_LOCATION = Path(os.path.dirname(__file__))
XML_LOCATION = MY_LOCATION / '../modern/source'


nationality_mappings = {
    'italiana': 'italian',
    'italo-britannica': 'italian',
    'united states of america': 'american',
    'usa': 'american'
}
english_countries = ['american', 'british', 'australian', 'irish']

proficiency_mappings = { # put together 7 and 8 (living in anglo-saxon and weak)
    'living in anglo-saxon countries for at least 2 years': 'weak english speaker',
    'week english speaker': 'weak english speaker'
}

speaker_to_en_countries = lambda x: normalize_key(x[S_NATIONALITY_COLUMN], nationality_mappings) in english_countries
speaker_to_proficiency = lambda x: normalize_key(x[S_PROFICIENCY_COLUMN], proficiency_mappings)
speaker_to_native = lambda x: normalize_key(x[S_NATIVE_COLUMN])

def normalize_key(str, mappings={}):
    str = str.lower()
    # get association if needed
    str = mappings.get(str, str)
    return str

def group_by(table, key_fn, result_column_idx, verbose=False):
    groups = {k:[r[result_column_idx] for r in v] for (k,v) in groupby(sorted(table, key=key_fn), key=key_fn)}
    if verbose:
        print([(k, len(v)) for k,v in groups.items()])

    return groups

def get_command_ids_from_speakers_uid(speakers_uid):
    # get the audio_files ids for this speaker
    audio_sentences_ids = {a[A_SID_COLUMN] for a in audio_files if a[A_SPEAKER_UID_COLUMN] in speakers_uid}
    # and now get the corresponding command ids
    command_ids = {x[X_ID_COLUMN] for x in XDG if x[X_SID_COLUMN] in audio_sentences_ids}
    
    return command_ids

def group_by_criterions():
    # should do something not to have code around
    pass

groups = {
    'en_nation': group_by(speakers, speaker_to_en_countries, S_UID_COLUMN, True),
    'native': group_by(speakers, speaker_to_proficiency, S_UID_COLUMN, True),
    'proficiency': group_by(speakers, speaker_to_native, S_UID_COLUMN, True)
}
xml_existing_files = sorted([x.name for x in XML_LOCATION.iterdir() if x.is_file()])
print('existing XML files:', len(xml_existing_files))
results = {}
for group_criterion, groups in groups.items():
    print('group_criterion:', group_criterion)
    results[group_criterion] = {}
    for group_name, group in groups.items():
        commands = get_command_ids_from_speakers_uid(group)
        commands = ['{}.xml'.format(c) for c in commands]
        # filter by existing XML files in HuRIC
        existing_commands = [c for c in commands if c in xml_existing_files]
        print('\t', group_name, 'commands all:', len(commands), 'commands existing:', len(existing_commands))
        results[group_criterion][group_name] = sorted(existing_commands)
# save the groups
with open('{}/{}'.format(MY_LOCATION, 'groups.json'), 'w') as json_file:
    json.dump(results, json_file)