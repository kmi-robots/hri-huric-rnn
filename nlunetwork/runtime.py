"""This module runs the inference for the neural network"""
import spacy
import numpy as np

from spacy.gold import iob_to_biluo, offsets_from_biluo_tags

from nlunetwork import runtime_model
from nlunetwork.data import get_language_model_name

# TODO get max_len from a tensor in the model!!
max_len = 100

class NLUWrapper(object):

    def __init__(self,
model_path, intent_treshold_score=0.01):
        self.language_model_name = get_language_model_name('en', 'large')
        self.intent_treshold_score = intent_treshold_score
        self.nlp = spacy.load(self.language_model_name)
        self.model = runtime_model.RuntimeModel(model_path, 300, 'en', self.nlp)

    def parse(self, text):

        return self.model_predict(self.model, text)

    def model_predict(self, model, text):
        doc = self.nlp.make_doc(text)

        words_true = [w.text for w in doc]
        length = len(words_true)
        words_true += ['<EOS>']
        words = words_true + ['<PAD>'] * (max_len-len(words_true))
        words = np.array(words)
        batch = [{
            'words': words,
            'length': length
        }]

        result = model.test(batch)
        # batch only contains one element
        decoder_prediction = result['slots'][0][:length]
        intent = result['intent'][0]
        intent_score = result['intent_confidence'][0]
        intent_attentions = result['intent_attentions'][0]
        biluo_tags = iob_to_biluo(decoder_prediction)
        entities_offsets = offsets_from_biluo_tags(doc, biluo_tags)
        entities = []
        for ent in entities_offsets:
            e_parts = ent[2].split('.')
            if len(e_parts) > 1:
                # role.type
                entity = {'role': e_parts[0], 'type': e_parts[1]}
            else:
                entity = {'role': None, 'type': e_parts[0]}
            value = text[ent[0]: ent[1]]
            entities.append({'entity': entity['type'],
                'start': ent[0],
                'end': ent[1],
                'role': entity['role'],
                'value': value,
            })

        # now convert to the same format as wit.ai, applying the treshold
        if intent_score < self.intent_treshold_score:
            intent_result = None
        else:
            intent_result = {'confidence': str(intent_score), 'value': intent, 'attentions': [str(val) for val in intent_attentions[:length]]}

        entities_result = {}
        for ent in entities:
            if ent['role']:
                entities_result[ent['role']] = ent
            else:
                entities_result[ent['entity']] = ent

        return {
            'intent': intent_result,
            'entities': entities_result,
            'slots': decoder_prediction.tolist()
        }
