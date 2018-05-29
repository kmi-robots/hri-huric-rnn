import random
import sys
import os
import json
import time
import spacy
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict

from . import data
from .model import Model
from . import metrics

# embedding size for labels
embedding_size = 64
# size of LSTM cells
hidden_size = 100
# size of batch
batch_size = 16
# number of training epochs
epoch_num = 50

MY_PATH = os.path.dirname(os.path.abspath(__file__))

DATASET = os.environ.get('DATASET', 'atis')
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', 'last')
MODE = os.environ.get('MODE', None)
if not MODE:
    # for those two datasets, default to train full
    if DATASET == 'wit_en' or DATASET == 'wit_it':
        MODE = 'runtime'
    else:
        MODE = 'measures'
# the type of recurrent unit on the multi-turn: rnn or CRF
RECURRENT_MULTITURN=os.environ.get('RECURRENT_MULTITURN','gru')

# set this to 'no_all', 'no_bot_turn', 'no_previous_intent' for a partial single-turned net on multi-turn datasets
FORCE_SINGLE_TURN = os.environ.get('FORCE_SINGLE_TURN', False)
if FORCE_SINGLE_TURN:
    OUTPUT_FOLDER += '_single_' + FORCE_SINGLE_TURN
if RECURRENT_MULTITURN != 'gru':
    OUTPUT_FOLDER += '_' + RECURRENT_MULTITURN
if MODE=='measures':
    # don't overwrite anything
    #OUTPUT_FOLDER += str(time.time())
    pass

LOSS_SUM = os.environ.get('LOSS_SUM', 'both') # 'both' if want to reduce loss of both intents and slots, otherwise 'intent' or 'slots'
OUTPUT_FOLDER += '_loss_' + LOSS_SUM
SLOTS_TYPE = os.environ.get('SLOTS_TYPE', 'full') # what part of the slots to consider: 'full': B-Location, 'iob_only': B (corresponds to only boundary detection), 'slot_only': Location
OUTPUT_FOLDER += '_slottype_' + SLOTS_TYPE
WORD_EMBEDDINGS = os.environ.get('WORD_EMBEDDINGS', 'large')
OUTPUT_FOLDER += '_we_' + WORD_EMBEDDINGS
RECURRENT_CELL = os.environ.get('RECURRENT_CELL', 'lstm')
OUTPUT_FOLDER += '_recurrent_cell_' + RECURRENT_CELL
ATTENTION = os.environ.get('ATTENTION', 'slots') # intents, slots, both, none
OUTPUT_FOLDER += '_attention_' + ATTENTION
THREE_STAGES = os.environ.get('THREE_STAGES', False) # add Boundary Detection intermediate level
if THREE_STAGES:
    OUTPUT_FOLDER += '_three_stages'

print('environment variables:')
print('DATASET:', DATASET, '\nOUTPUT_FOLDER:', OUTPUT_FOLDER, '\nMODE:', MODE, '\nRECURRENT_MULTITURN:', RECURRENT_MULTITURN, '\nFORCE_SINGLE_TURN:', FORCE_SINGLE_TURN, '\nWORD_EMBEDDINGS:', WORD_EMBEDDINGS, '\nRECURRENT_CELL:', RECURRENT_CELL, '\nATTENTION:', ATTENTION)

def get_model(vocabs, tokenizer, language, multi_turn, input_steps, nlp):
    model = Model(input_steps, embedding_size, hidden_size, vocabs, WORD_EMBEDDINGS, RECURRENT_CELL, ATTENTION, LOSS_SUM, multi_turn, None, RECURRENT_MULTITURN, THREE_STAGES)
    model.build(nlp, tokenizer, language)
    return model


def train(mode):
    # maximum length of sentences
    input_steps = 50
    # load the train and dev datasets
    folds = data.load_data(DATASET, mode, SLOTS_TYPE)
    # fix the random seeds
    random_seed_init(len(folds[0]['data']))
    # preprocess them to list of training/test samples
    # a sample is made up of a tuple that contains
    # - an input sentence (list of words --> strings, padded)
    # - the real length of the sentence (int) to be able to recognize padding
    # - an output sequence (list of IOB annotations --> strings, padded)
    # - an output intent (string)
    multi_turn = folds[0]['meta'].get('multi_turn', False)
    print('multi_turn:', multi_turn)
    if multi_turn:
        input_steps *=2
        folds = [data.collapse_multi_turn_sessions(fold, FORCE_SINGLE_TURN) for fold in folds]
    folds = [data.adjust_sequences(fold, input_steps) for fold in folds]

    all_samples = [s for fold in folds for s in fold['data']] 
    meta_data = folds[0]['meta']

    
    # turn off multi_turn for the required additional feeds and previous intent RNN
    if multi_turn and FORCE_SINGLE_TURN == 'no_all' or FORCE_SINGLE_TURN == 'no_previous_intent':
        multi_turn = False
    # get the vocabularies for input, slot and intent
    vocabs = data.get_vocabularies(all_samples, meta_data)
    # and get the model
    if FORCE_SINGLE_TURN == 'no_previous_intent':
        # changing this now, implies that the model doesn't have previous intent
        multi_turn = False
    
    language_model_name = data.get_language_model_name(meta_data['language'], WORD_EMBEDDINGS)
    nlp = spacy.load(language_model_name)

    real_folder = MY_PATH + '/results/' + OUTPUT_FOLDER + '/' + DATASET
    if not os.path.exists(real_folder):
        os.makedirs(real_folder)

    create_empty_array = lambda: np.zeros((epoch_num, len(folds)))
    
    # initialize the history that will collect some measures
    history = {
        'intent': defaultdict(create_empty_array), # the intents
        'slot_sequence': defaultdict(create_empty_array), # the raw sequence on word level, considering full B-Type
        'sequence_boundaries': defaultdict(create_empty_array), # the raw sequence on word level, but considering only IOB, not Types (Boundary Detection, not Argument Classification)
        'slots_boundaries': defaultdict(create_empty_array), # the Boundary Detection: on slot level, but only considering IOB
        'slots_boundaries_cond': defaultdict(create_empty_array), # as before, but true positive only if the intent is correct
        'slots': defaultdict(create_empty_array), # value+role+entity (Boundary Detection + Argument Classification)
        'slots_cond': defaultdict(create_empty_array), # as before, but conditioned to correct intent
    }

    for fold_number in range(0, len(folds)):
        # reset the graph for next iteration
        tf.reset_default_graph()
    
        training_samples = [s for (count,fold) in enumerate(folds) if count != fold_number for s in fold['data']]
        test_samples = folds[fold_number]['data']
        print('train samples', len(training_samples))
        if test_samples:
            print('test samples', len(test_samples))


        model = get_model(vocabs, meta_data['tokenizer'], meta_data['language'], multi_turn, input_steps, nlp)
        
        global_init_op = tf.global_variables_initializer()
        table_init_op = tf.tables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        
        # initialize the required parameters
        sess.run(global_init_op)
        sess.run(table_init_op)

        if multi_turn:
            print('i am multi turn')
        for epoch in range(epoch_num):
            #mean_loss = 0.0
            #train_loss = 0.0
            for i, batch in enumerate(data.get_batch(batch_size, training_samples)):
                # perform a batch of training
                #print(batch)
                #_, loss, bd_prediction, decoder_prediction, intent, mask = model.step(sess, 'train', batch)
                model.step(sess, 'train', batch)
                """
                mean_loss += loss
                train_loss += loss
                if i % 10 == 0:
                    if i > 0:
                        mean_loss = mean_loss / 10.0
                    #print('Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
                    print('.', end='')
                    sys.stdout.flush()
                    mean_loss = 0
                """
            #train_loss /= (i + 1)
            #print('[Epoch {}] Average train loss: {}'.format(epoch, train_loss))

            if test_samples:
                # test each epoch once
                pred_iob = []
                pred_intents = []
                true_intents = []
                previous_intents = []
                if fold_number == 0:
                    # copy just on the first fold, avoid overwriting
                    data.copy_huric_xml_to('{}/xml/epoch_{}'.format(real_folder, epoch))
                
                predicted = []
                for j, batch in enumerate(data.get_batch(batch_size, test_samples)):
                    results = model.step(sess, 'test', batch)
                    intent = results['intent']
                    if THREE_STAGES:
                        bd_prediction = results['bd']
                        ac_prediction = results['ac']
                        bd_prediction = np.transpose(bd_prediction, [1, 0])
                        ac_prediction = np.transpose(ac_prediction, [1, 0])
                        decoder_prediction = np.array([data.rebuild_slots_sequence(bd_seq, ac_seq) for bd_seq, ac_seq in zip(bd_prediction, ac_prediction)])
                    else:
                        decoder_prediction = results['slots']
                        # from time-major matrix to sample-major
                        decoder_prediction = np.transpose(decoder_prediction, [1, 0])
                        decoder_prediction = decoder_prediction.tolist()

                    #print(results)
                    predicted_batch = metrics.clean_predictions(decoder_prediction, intent, batch)
                    data.huric_add_json('{}/xml/epoch_{}'.format(real_folder, epoch), predicted_batch)
                    predicted.extend(predicted_batch)
                    if j == 0:
                        index = random.choice(range(len(batch)))
                        # index = 0
                        print('Input Sentence        :', batch[index]['words'][:batch[index]['length']])
                        if THREE_STAGES:
                            print('BD Truth              :', batch[index]['boundaries'][:batch[index]['length']])
                            print('BD Prediction         :', bd_prediction[index][:batch[index]['length']].tolist())
                            print('AC Truth              :', batch[index]['types'][:batch[index]['length']])
                            print('AC Prediction         :', ac_prediction[index][:batch[index]['length']].tolist())
                        print('Slot Truth            :', batch[index]['slots'][:batch[index]['length']])
                        print('Slot Prediction       :', decoder_prediction[index][:batch[index]['length']])
                        print('Intent Truth          :', batch[index]['intent'])
                        print('Intent Prediction     :', intent[index])
                    # the temporal length of prediction for this batch
                    slot_pred_length = list(np.shape(decoder_prediction))[1]
                    # pad with '<PAD>' (two steps because of numpy issues)
                    pred_padded = np.pad(decoder_prediction, ((0, 0), (0, input_steps-slot_pred_length)),
                                            mode='constant', constant_values=0)
                    pred_padded[pred_padded == 0] = '<PAD>'
                    # TODO the same for bd

                    # save the results of prediction
                    # pred_padded is an array of shape (batch_size, pad_length)
                    pred_iob.append(pred_padded)
                    pred_intents.extend(intent)
                    # and save also the true values
                    true_intent = [sample['intent'] for sample in batch]
                    true_intents.extend(true_intent)
                    true_slot = np.array([sample['slots'] for sample in batch])
                    true_length = np.array([sample['length'] for sample in batch])
                    true_slot = true_slot[:, :slot_pred_length]
                    # print(np.shape(true_slot), np.shape(decoder_prediction))
                    # print(true_slot, decoder_prediction)
                    print('.', end='')
                    if multi_turn:
                        previous_intents.extend([sample['previous_intent'] for sample in batch])

                data.save_predictions('{}/json/epoch_{}'.format(real_folder, epoch), fold_number + 1, predicted)
                """
                # put together
                pred_iob_a = np.vstack(pred_iob)
                # pred_iob_a is of shape (n_test_samples, sequence_len)
                #print('pred_iob_a: ', pred_iob_a.shape)
                true_slots_iob = np.array([sample['slots'] for sample in test_samples])[:pred_iob_a.shape[0]]
                intents_scores = metrics.precision_recall_f1_for_intents(true_intents, pred_intents)
                #accuracy_intents = accuracy_score(true_intents, pred_intents)
                slots_iob_scores = metrics.precision_recall_f1_for_sequence(true_slots_iob, pred_iob_a)
                # convert IOB to slots stringified LABEL:START_IDX-END_IDX for comparison
                true_slots = data.sequence_iob_to_ents(true_slots_iob)
                pred_slots = data.sequence_iob_to_ents(pred_iob_a)
                
                # compute metrics: precision, recall, f1
                slots_scores = metrics.precision_recall_f1_slots(true_slots, pred_slots)
                slots_cond_scores = metrics.precision_recall_f1_slots_conditioned_intent(true_slots, pred_slots, true_intents, pred_intents)
                
                # only boundary detection measures
                true_slots_iob_only = np.array([data.slots_to_iob_only(iob_seq) for iob_seq in true_slots_iob.tolist()])
                pred_slots_iob_only = np.array([data.slots_to_iob_only(iob_seq) for iob_seq in pred_iob_a.tolist()])
                sequence_boundaries_scores = metrics.precision_recall_f1_for_sequence(true_slots_iob_only, pred_slots_iob_only)
                true_boundaries = data.sequence_iob_to_ents(true_slots_iob_only)
                pred_boundaries = data.sequence_iob_to_ents(pred_slots_iob_only)
                slots_boundaries_scores = metrics.precision_recall_f1_slots(true_boundaries, pred_boundaries)
                slots_boundaries_cond_scores = metrics.precision_recall_f1_slots_conditioned_intent(true_boundaries, pred_boundaries, true_intents, pred_intents)
                """
                # epoch resume
                print('epoch {}/{} on fold {}/{} ended'.format(epoch + 1, epoch_num, fold_number + 1, len(folds)))
                performance = metrics.evaluate_epoch(predicted)
                print('INTENTS:  ', performance['intent']['f1'])
                print('SLOTS:    ', performance['slots']['f1'])
                print('BD:       ', performance['bd']['f1'])
                print('BD COND:  ', performance['bd_cond']['f1'])
                print('AC:       ', performance['ac']['f1'])
                print('AC COND:  ', performance['ac_cond']['f1'])
                """
                print('INTENTS               : {}'.format(intents_scores))
                print('SEQUENCE              : {}'.format(slots_iob_scores))
                print('SEQUENCE BOUNDARIES   : {}'.format(sequence_boundaries_scores))
                print('SLOTS BOUNDARIES      : {}'.format(slots_boundaries_scores))
                print('SLOTS BOUNDARIES COND : {}'.format(slots_boundaries_cond_scores))
                print('SLOTS                 : {}'.format(slots_scores))
                print('SLOTS COND            : {}'.format(slots_cond_scores))
                # TODO compute properly on the intermediate stage of boundary detection
                for score_name, scores in intents_scores.items(): history['intent'][score_name][epoch, fold_number] = scores
                for score_name, scores in slots_iob_scores.items(): history['slot_sequence'][score_name][epoch, fold_number] = scores
                for score_name, scores in slots_scores.items(): history['slots'][score_name][epoch, fold_number] = scores
                for score_name, scores in slots_cond_scores.items(): history['slots_cond'][score_name][epoch, fold_number] = scores

                for score_name, scores in sequence_boundaries_scores.items(): history['sequence_boundaries'][score_name][epoch, fold_number] = scores
                for score_name, scores in slots_boundaries_scores.items(): history['slots_boundaries'][score_name][epoch, fold_number] = scores
                for score_name, scores in slots_boundaries_cond_scores.items(): history['slots_boundaries_cond'][score_name][epoch, fold_number] = scores

                if multi_turn:
                    # evaluate the intent transitions in samples and the transition inferred
                    true_intent_changes, pred_intent_changes = list(zip(*[(prev != true, prev != pred) for prev, pred, true in zip(previous_intents, pred_intents, true_intents)]))
                    true_positives, true_negatives = list(zip(*[(true and pred, not true and not pred) for true, pred in zip(true_intent_changes, pred_intent_changes)]))
                    print('INTENT CHANGE statistics for epoch {}: {} true positives and {} true negatives over {} samples'.format(epoch, sum(true_positives), sum(true_negatives), len(true_positives)))
                """

        # the iteration on the fold has completed

    if test_samples:
        print('computing the metrics for all epochs on all the folds merged')
        for epoch in range(epoch_num):
            json_fold_location = '{}/json/epoch_{}'.format(real_folder, epoch)
            merged_predicitons = data.merge_prediction_folds(json_fold_location)
            epoch_metrics = metrics.evaluate_epoch(merged_predicitons)
            with open('{}/epoch_{}.json'.format(real_folder, epoch), 'w') as f:
                json.dump(epoch_metrics, f, indent=2)
        # compute aggregated values: output_type(intent, slots, ...) -> score_name(precision, recall, f1) -> mean/stddev
        stats = defaultdict(lambda: defaultdict(dict))
        for output_type, values in history.items():
            for score_name, scores in values.items():
                stats[output_type][score_name]['mean'] = np.mean(scores, axis=1)
                stats[output_type][score_name]['stddev'] = np.std(scores, axis=1)

        print('averages over the K folds have been computed')
    
        to_plot_f1_mean = {output_type: values['f1']['mean'] for output_type, values in stats.items()}
        to_plot_f1_stddev = {output_type: values['f1']['stddev'] for output_type, values in stats.items()}
        metrics.plot_history('{}/f1_mean.png'.format(real_folder) , to_plot_f1_mean)
        metrics.plot_history('{}/f1_stddev.png'.format(real_folder) , to_plot_f1_stddev)
        
        save_file(stats, '{}/history_aggregated.json'.format(real_folder))
        save_file(history, '{}/history_full.json'.format(real_folder))
    else:
        saver = tf.train.Saver()
        saver.save(sess, '{}/model.ckpt'.format(real_folder))


def random_seed_init(seed):
    random.seed(seed)
    tf.set_random_seed(seed)

def save_file(file_content, file_path):
    with open(file_path, 'w') as out_file:
        json.dump(file_content, out_file, indent=2, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    train(MODE)