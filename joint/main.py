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
embedding_size = int(os.environ.get('LABEL_EMB_SIZE', 64))
# size of LSTM cells
hidden_size = int(os.environ.get('LSTM_SIZE', 100))
# size of batch
batch_size = int(os.environ.get('BATCH_SIZE', 16))
# number of training epochs
epoch_num = int(os.environ.get('MAX_EPOCHS', 100))

MY_PATH = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', '')
DATASET = os.environ.get('DATASET', 'huric_eb/modern')
# possible MODE:
# - 'dev_cross' that excludes the last fold and performs (k-1)-fold, last fold untouched
# - 'cross' that performs k-fold
# - 'eval' that does the train on k-1 and test on last (untouched fold)
MODE = os.environ.get('MODE', 'dev_cross')
OUTPUT_FOLDER += MODE


# the type of recurrent unit on the multi-turn: rnn or CRF
RECURRENT_MULTITURN=os.environ.get('RECURRENT_MULTITURN','gru')

# set this to 'no_all', 'no_bot_turn', 'no_previous_intent' for a partial single-turned net on multi-turn datasets
FORCE_SINGLE_TURN = os.environ.get('FORCE_SINGLE_TURN', False)
if FORCE_SINGLE_TURN:
    OUTPUT_FOLDER += '_single_' + FORCE_SINGLE_TURN
if RECURRENT_MULTITURN != 'gru':
    OUTPUT_FOLDER += '_' + RECURRENT_MULTITURN

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


# hyperparams
OUTPUT_FOLDER += '___hyper:LABEL_EMB_SIZE={},LSTM_SIZE={},BATCH_SIZE={},MAX_EPOCHS={}'.format(embedding_size, hidden_size, batch_size, epoch_num)

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
    folds = data.load_data(DATASET, SLOTS_TYPE)
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

    create_empty_array = lambda: np.zeros((epoch_num, ))

    train_folds = []
    test_folds = []
    if mode == 'dev_cross':
        # cross on 1...k-1
        folds = folds[:-1]
    if mode == 'cross' or mode == 'dev_cross':
        for fold_number in range(0, len(folds)):
            train = [s for (count,fold) in enumerate(folds) if count != fold_number for s in fold['data']]
            test = folds[fold_number]['data']
            train_folds.append(train)
            test_folds.append(test)
    elif mode == 'eval':
        # train on 1...k-1, test on k
        train_folds.append([s for (count,fold) in enumerate(folds[:-1]) for s in fold['data']])
        test_folds.append(folds[-1]['data'])
    else:
        raise ValueError('invalid mode')

    for fold_number, (training_samples, test_samples) in enumerate(zip(train_folds, test_folds)):
        # reset the graph for next iteration
        tf.reset_default_graph()

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
                print('.', end='')
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
                if fold_number == 0:
                    # copy just on the first fold, avoid overwriting
                    data.copy_huric_xml_to('{}/xml/epoch_{}'.format(real_folder, epoch))
                
                predicted = []
                for j, batch in enumerate(data.get_batch(batch_size, test_samples)):
                    results = model.step(sess, 'test', batch)
                    print('.', end='')
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
                

                data.save_predictions('{}/json/epoch_{}'.format(real_folder, epoch), fold_number + 1, predicted)
                # epoch resume
                print('epoch {}/{} on fold {}/{} ended'.format(epoch + 1, epoch_num, fold_number + 1, len(train_folds)))
                performance = metrics.evaluate_epoch(predicted)
                print('INTENTS:    ', performance['intent']['f1'])
                print('SLOTS:      ', performance['slots']['f1'])
                print('SLOTS COND: ', performance['slots_cond']['f1'])
                print('BD:         ', performance['bd']['f1'])
                print('BD COND:    ', performance['bd_cond']['f1'])
                print('AC:         ', performance['ac']['f1'])
                print('AC COND:    ', performance['ac_cond']['f1'])

        # the iteration on the fold has completed

    if test_samples:
        print('computing the metrics for all epochs on all the folds merged')
        
        # initialize the history that will collect some measures
        history = defaultdict(lambda:  defaultdict(create_empty_array))
        for epoch in range(epoch_num):
            json_fold_location = '{}/json/epoch_{}'.format(real_folder, epoch)
            merged_predicitons = data.merge_prediction_folds(json_fold_location)
            data.save_predictions('{}/json/epoch_{}'.format(real_folder, epoch), 'full', merged_predicitons)
            epoch_metrics = metrics.evaluate_epoch(merged_predicitons)
            save_file(epoch_metrics, '{}/scores'.format(real_folder), 'epoch_{}.json'.format(epoch))
            for key, measures in epoch_metrics.items():
                    for measure_name, value in measures.items():
                        history[key][measure_name][epoch] = value

        print('averages over the K folds have been computed')
    
        to_plot_precision = {output_type: values['precision'] for output_type, values in history.items()}
        to_plot_recall = {output_type: values['recall'] for output_type, values in history.items()}
        to_plot_f1 = {output_type: values['f1'] for output_type, values in history.items()}
        metrics.plot_history('{}/f1.png'.format(real_folder) , to_plot_f1)
        save_file(history, real_folder, 'history_full.json')
    else:
        saver = tf.train.Saver()
        saver.save(sess, '{}/model.ckpt'.format(real_folder))


def random_seed_init(seed):
    random.seed(seed)
    tf.set_random_seed(seed)

def save_file(file_content, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open('{}/{}'.format(file_path, file_name) , 'w') as out_file:
        json.dump(file_content, out_file, indent=2, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    train(MODE)
