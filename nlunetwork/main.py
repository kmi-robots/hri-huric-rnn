import random
import sys
import os
import json
import time
import spacy
import dotenv
import itertools
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm

from nlunetwork import data
from nlunetwork.model import Model
from nlunetwork import metrics
from nlunetwork import runtime_model

MY_PATH = os.path.dirname(os.path.abspath(__file__))

def load_config(config_override_file_name=None):
    config_folder = '{}/../configurations/'.format(MY_PATH)
    if config_override_file_name:
        config_override_file_path = config_override_file_name
        if not os.path.isfile(config_override_file_path):
            raise ValueError('config file not found: ' + config_override_file_path)
        dotenv.load_dotenv(dotenv_path=config_override_file_path, verbose=True)

    dotenv.load_dotenv(dotenv_path='configurations/default.env')

    config = {}
    # embedding size for labels
    config['LABEL_EMB_SIZE'] = int(os.environ.get('LABEL_EMB_SIZE'))
    # size of LSTM cells
    config['LSTM_SIZE'] = int(os.environ.get('LSTM_SIZE'))
    # size of batch
    config['BATCH_SIZE'] = int(os.environ.get('BATCH_SIZE'))
    # number of training epochs
    config['MAX_EPOCHS'] = int(os.environ.get('MAX_EPOCHS'))
    # the output folder
    config['OUTPUT_FOLDER'] = os.environ.get('OUTPUT_FOLDER')
    # the dataset used
    config['DATASET'] = os.environ.get('DATASET', 'huric_eb/modern_right')
    # switches the behaviour between several train/test scenarios
    config['MODE'] = os.environ.get('MODE', 'dev_cross')
    # specific to test mode
    config['MODEL_PATH'] = os.environ.get('MODEL_PATH')
    # the type of recurrent unit on the multi-turn: rnn or CRF
    config['RECURRENT_MULTITURN']=os.environ.get('RECURRENT_MULTITURN','gru')
    # partial single-turned net on multi-turn datasets
    config['FORCE_SINGLE_TURN'] = os.environ.get('FORCE_SINGLE_TURN', 'no')
    if config['FORCE_SINGLE_TURN'] in ('false', 'no', '0'):
        config['FORCE_SINGLE_TURN'] = False
    # which loss to reduce
    config['LOSS_SUM'] = os.environ.get('LOSS_SUM', 'both')
    # what part of the slots to consider
    config['SLOTS_TYPE'] = os.environ.get('SLOTS_TYPE', 'full')
    # the size of the word embeddings
    config['WORD_EMBEDDINGS'] = os.environ.get('WORD_EMBEDDINGS', 'large')
    # the recurrent cell
    config['RECURRENT_CELL'] = os.environ.get('RECURRENT_CELL', 'lstm')
    config['ATTENTION'] = os.environ.get('ATTENTION', 'both') # intents, slots, both, none
    # if and how to perform three stages
    config['THREE_STAGES'] = os.environ.get('THREE_STAGES', 'true_highway').lower()
    if config['THREE_STAGES'] in ('false', 'no', '0'):
        config['THREE_STAGES'] = False
    config['INTENT_EXTRACTION_MODE'] = os.environ.get('INTENT_EXTRACTION_MODE', 'bi-rnn') # intent comes out of bi-rnn or only a weighted mean (attention intent must be turned on)

    config['DROPOUT_KEEP_PROB'] = float(os.environ.get('DROPOUT_KEEP_PROB', 1))

    # recap and final setting of the output folder
    config['OUTPUT_FOLDER'] += '/'
    if config_override_file_name:
        # removing the extension .env
        config['OUTPUT_FOLDER'] += '.'.join(config_override_file_name.split('.')[:-1])

    print('configuration:', config)

    return config

def get_model(vocabs, tokenizer, language, multi_turn, input_steps, nlp):
    model = Model(input_steps, config['LABEL_EMB_SIZE'], config['LSTM_SIZE'], vocabs, config['WORD_EMBEDDINGS'], config['RECURRENT_CELL'], config['ATTENTION'], config['FORCE_SINGLE_TURN'], multi_turn, None, config['RECURRENT_MULTITURN'], config['THREE_STAGES'], config['INTENT_EXTRACTION_MODE'])
    model.build(nlp, tokenizer, language)
    return model


def train(mode, config):
    # maximum length of sentences
    input_steps = 100
    # load the train and dev datasets
    folds = data.load_data(config['DATASET'], config['SLOTS_TYPE'])
    # preprocess them to list of training/test samples
    # a sample is made up of a tuple that contains
    # - an input sentence (list of words --> strings, padded)
    # - the real length of the sentence (int) to be able to recognize padding
    # - an output sequence (list of IOB annotations --> strings, padded)
    # - an output intent (string)
    batch_size = config['BATCH_SIZE']
    multi_turn = folds[0]['meta'].get('multi_turn', False)
    print('multi_turn:', multi_turn)
    if multi_turn:
        input_steps *=2
        folds = [data.collapse_multi_turn_sessions(fold, config['FORCE_SINGLE_TURN']) for fold in folds]
    folds = [data.adjust_sequences(fold, input_steps) for fold in folds]

    all_samples = [s for fold in folds for s in fold['data']]
    meta_data = folds[0]['meta']


    # turn off multi_turn for the required additional feeds and previous intent RNN
    if multi_turn and config['FORCE_SINGLE_TURN'] == 'no_all' or config['FORCE_SINGLE_TURN'] == 'no_previous_intent':
        multi_turn = False
    # get the vocabularies for input, slot and intent
    vocabs = data.get_vocabularies(all_samples, meta_data)
    # and get the model
    if config['FORCE_SINGLE_TURN'] == 'no_previous_intent':
        # changing this now, implies that the model doesn't have previous intent
        multi_turn = False

    language_model_name = data.get_language_model_name(meta_data['language'], config['WORD_EMBEDDINGS'])
    nlp = spacy.load(language_model_name)

    real_folder = MY_PATH + '/results/' + config['OUTPUT_FOLDER'] + '/' + config['DATASET']
    if not os.path.exists(real_folder):
        os.makedirs(real_folder)

    with open('{}/config.env'.format(real_folder), 'w') as f:
        string = '\n'.join(['{}={}'.format(k,v) for k,v in config.items()])
        f.write(string)

    create_empty_array = lambda: np.zeros((config['MAX_EPOCHS'], ))

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
    elif mode == 'cross_nested':
        # look at TODO documentation for this mess
        fold_indexes = list(range(0, len(folds)))
        #for i in fold_indexes:
        #    not_i = list(fold_indexes).remove(i)
        #    for j in not_i:
        #        not_i_and_j = list(not_i).remove(j)
        for comb in itertools.combinations(fold_indexes, len(fold_indexes) - 2):
            not_i_and_j = list(comb)
            i_and_j = [el for el in fold_indexes if el not in not_i_and_j]
            #print(not_i_and_j, i_and_j)
            train_folds.append([s for idx, fold in enumerate(folds) if idx in not_i_and_j for s in fold['data']])
            test_folds.append([s for idx, fold in enumerate(folds) if idx in i_and_j for s in fold['data']])
    elif mode == 'eval':
        # train on 1...k-1, test on k
        train_folds.append([s for (count,fold) in enumerate(folds[:-1]) for s in fold['data']])
        test_folds.append(folds[-1]['data'])
    elif mode == 'train_all':
        train_folds.append([s for (count,fold) in enumerate(folds) for s in fold['data']])
        test_folds.append([])
    elif mode.startswith('test'):
        train_folds.append([])
        if mode == 'test':
            test_folds.append(folds[-1]['data'])
        elif mode == 'test_all':
            test_folds.append([s for (count,fold) in enumerate(folds) for s in fold['data']])
    else:
        raise ValueError('invalid mode')

    # filter what is tested, for the dataset combination
    for f in test_folds:
        old_len = len(f)
        f[:] = [s for s in f if not ('origin' in s)]
        new_len = len(f)
        if old_len != new_len:
            print('the test is being filtered on the samples without attribute \'origin\'', old_len, new_len)


    for fold_number, (training_samples, test_samples) in enumerate(zip(train_folds, test_folds)):
        # reset the graph for next iteration
        tf.reset_default_graph()
        # fix the random seeds
        random_seed_init(len(folds[0]['data']))

        print('train samples', len(training_samples))
        if test_samples:
            print('test samples', len(test_samples))

        if mode.startswith('test'):
            # restore a model
            model, sess = restore_graph(config['MODEL_PATH'], nlp)
            config['MAX_EPOCHS'] = 1
        else:
            model, sess = build_graph(nlp, vocabs, meta_data, multi_turn, input_steps)

        for epoch in range(config['MAX_EPOCHS']):
            print('epoch {}/{}'.format(epoch + 1, config['MAX_EPOCHS']))
            #mean_loss = 0.0
            #train_loss = 0.0
            if not mode.startswith('test'):
                for i, batch in tqdm(enumerate(data.get_batch(batch_size, training_samples)), total=len(training_samples)//batch_size):
                    # perform a batch of training
                    #print(batch)
                    #_, loss, bd_prediction, decoder_prediction, intent, mask = model.step(sess, 'train', batch)
                    model.step(sess, 'train', batch, config['DROPOUT_KEEP_PROB'])

            if test_samples:
                test_epoch(model, sess, test_samples, fold_number, real_folder, epoch, input_steps)

        if not mode.startswith('test'):
            # the iteration on the fold has completed
            # save the model
            saver = tf.train.Saver()
            saver.save(sess, '{}/model_fold_{}.ckpt'.format(real_folder, fold_number))

    if test_samples:
        print('computing the metrics for all epochs on all the folds merged')

        # initialize the history that will collect some measures
        history = defaultdict(lambda:  defaultdict(create_empty_array))
        for epoch in range(config['MAX_EPOCHS']):
            json_fold_location = '{}/json/epoch_{}'.format(real_folder, epoch)
            merged_predicitons = data.merge_prediction_folds(json_fold_location)
            data.save_predictions('{}/json/epoch_{}'.format(real_folder, epoch), 'full', merged_predicitons)
            epoch_metrics = metrics.evaluate_epoch(merged_predicitons)
            save_file(epoch_metrics, '{}/scores'.format(real_folder), 'epoch_{}.json'.format(epoch))
            for key, measures in epoch_metrics.items():
                    if isinstance(measures, dict):
                        for measure_name, value in measures.items():
                            history[key][measure_name][epoch] = value

        print('averages over the K folds have been computed')

        to_plot_precision = {output_type: values['precision'] for output_type, values in history.items()}
        to_plot_recall = {output_type: values['recall'] for output_type, values in history.items()}
        to_plot_f1 = {output_type: values['f1'] for output_type, values in history.items()}
        metrics.plot_history('{}/f1.png'.format(real_folder) , to_plot_f1)
        save_file(history, real_folder, 'history_full.json')

def build_graph(nlp, vocabs, meta_data, multi_turn, input_steps):
    """Builds the computational graph"""
    model = get_model(vocabs, meta_data['tokenizer'], meta_data['language'], multi_turn, input_steps, nlp)

    global_init_op = tf.global_variables_initializer()
    table_init_op = tf.tables_initializer()
    sess = tf.Session()

    # initialize the required parameters
    sess.run(global_init_op)
    sess.run(table_init_op)

    return model, sess

def restore_graph(model_path, nlp):
    """Restores the stored computational graph, together with the optimized weights"""
    model = runtime_model.RuntimeModel(model_path, 300, 'en', nlp)

    return model, model.sess

def train_epoch(model, data):
    """Perform an epoch of training"""
    pass # TODO

def test_epoch(model, sess, test_samples, fold_number, real_folder, epoch, input_steps):
    """Perform an epoch of testing"""
    batch_size = config['BATCH_SIZE']
    if fold_number == 0:
        # copy just on the first fold, avoid overwriting
        data.copy_huric_xml_to('{}/xml/epoch_{}'.format(real_folder, epoch))

    predicted = []
    for j, batch in tqdm(enumerate(data.get_batch(batch_size, test_samples)), total=len(test_samples)//batch_size):
        results = model.step(sess, 'test', batch)
        intent = results['intent']
        intent_attentions = results['intent_attentions']
        if config['THREE_STAGES']:
            bd_prediction = results['bd']
            bd_prediction = np.transpose(bd_prediction, [1, 0])
            ac_prediction = results['ac']
            ac_prediction = np.transpose(ac_prediction, [1, 0])
            # all the attention matrices are in shape (time, batch, time)
            bd_attentions = results['bd_attentions']
            bd_attentions = np.transpose(bd_attentions, [1, 0, 2])
            ac_attentions = results['ac_attentions']
            ac_attentions = np.transpose(ac_attentions, [1, 0, 2])
            #print('bd_attentions.shape', bd_attentions.shape)
            decoder_prediction = np.array([data.rebuild_slots_sequence(bd_seq, ac_seq) for bd_seq, ac_seq in zip(bd_prediction, ac_prediction)])
            slots_attentions = np.zeros((len(batch), input_steps, input_steps))
        else:
            decoder_prediction = results['slots']
            # from time-major matrix to sample-major
            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            slots_attentions = results['slots_attentions']
            slots_attentions = np.transpose(slots_attentions, [1, 0, 2])
            bd_attentions = np.zeros((len(batch), input_steps, input_steps))
            ac_attentions = np.zeros((len(batch), input_steps, input_steps))

        #print(results)
        predicted_batch = metrics.clean_predictions(decoder_prediction, intent, batch, intent_attentions, bd_attentions, ac_attentions, slots_attentions)
        if config['DATASET'] == 'huric':
            data.huric_add_json('{}/xml/epoch_{}'.format(real_folder, epoch), predicted_batch)
        predicted.extend(predicted_batch)
        if j == 0:
            index = random.choice(range(len(batch)))
            # index = 0
            print('Input Sentence        :', batch[index]['words'][:batch[index]['length']])
            if config['THREE_STAGES']:
                print('BD Truth              :', batch[index]['boundaries'][:batch[index]['length']])
                print('BD Prediction         :', bd_prediction[index][:batch[index]['length']].tolist())
                print('AC Truth              :', batch[index]['types'][:batch[index]['length']])
                print('AC Prediction         :', ac_prediction[index][:batch[index]['length']].tolist())
                #print('BD atts', bd_attentions[index])
                #print('AC atts', ac_attentions[index])
            print('Slot Truth            :', batch[index]['slots'][:batch[index]['length']])
            print('Slot Prediction       :', decoder_prediction[index][:batch[index]['length']].tolist())
            print('Intent Truth          :', batch[index]['intent'])
            print('Intent Prediction     :', intent[index])
            print('Intent atts     :', intent_attentions[index][:batch[index]['length']])


    data.save_predictions('{}/json/epoch_{}'.format(real_folder, epoch), fold_number + 1, predicted)
    # epoch resume
    print('epoch {}/{} on fold {}'.format(epoch + 1, config['MAX_EPOCHS'], fold_number + 1))
    performance = metrics.evaluate_epoch(predicted)
    for metric_name, value in performance.items():
        print('%20s' % metric_name, value)

def random_seed_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def save_file(file_content, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open('{}/{}'.format(file_path, file_name) , 'w') as out_file:
        json.dump(file_content, out_file, indent=2, cls=data.NumpyEncoder)


if __name__ == '__main__':
    config = load_config(os.environ.get('CONFIG_FILE', None))
    train(config['MODE'], config)
