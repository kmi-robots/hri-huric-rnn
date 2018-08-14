import tensorflow as tf
import numpy as np
from nlunetwork import data
from nlunetwork.model import Model

class RuntimeModel(Model):
    """
    Restores a model from a checkpoint
    """

    def __init__(self, model_path, embedding_size, language, nlp, multi_turn=False, three_stages=True):
        # TODO would be better to store things like multi_turn into the model serialization
        self.multi_turn = multi_turn
        self.three_stages = three_stages
        # Step 1: restore the meta graph

        with tf.Graph().as_default() as graph:
            saver = tf.train.import_meta_graph(model_path + "model_fold_0.ckpt.meta")
        
            self.graph = graph

            # get tensors for inputs and outputs by name
            #self.decoder_prediction = graph.get_tensor_by_name('decoder_prediction:0')
            self.bd_prediction = graph.get_tensor_by_name('bd_prediction:0')
            self.ac_prediction = graph.get_tensor_by_name('ac_prediction:0')
            self.intent = graph.get_tensor_by_name('intent:0')
            self.intent_score = graph.get_tensor_by_name('intent_score:0')
            self.words_inputs = graph.get_tensor_by_name('words_inputs:0')
            self.attention_ac_scores = graph.get_tensor_by_name('attention_alpha_ac:0')
            self.attention_bd_scores = graph.get_tensor_by_name('attention_alpha_bd:0')
            self.encoder_inputs_actual_length = graph.get_tensor_by_name('encoder_inputs_actual_length:0')
            self.attention_scores_intent = graph.get_tensor_by_name('attention_alpha_intent:0')
            # redefine the py_func that is not serializable
            def static_wrapper(words):
                return data.spacy_wrapper(embedding_size, language, nlp, words)

            after_py_func = tf.py_func(static_wrapper, [self.words_inputs], tf.float32, stateful=False, name='spacy_wrapper')

            # Step 2: restore weights
            self.sess = tf.Session()
            self.sess.run(tf.tables_initializer())
            saver.restore(self.sess, model_path + "model_fold_0.ckpt")


    def test(self, inputs):

        seq_in, length = list(zip(*[(sample['words'], sample['length']) for sample in inputs]))
        
        output_feeds = [self.intent, self.intent_score, self.attention_scores_intent, self.bd_prediction, self.ac_prediction]
        feed_dict = {self.words_inputs: np.transpose(seq_in, [1, 0]), self.encoder_inputs_actual_length: length}

        results = self.sess.run(output_feeds, feed_dict=feed_dict)

        intent_batch, intent_score_batch, attention_scores_intent, bd_batch, ac_batch = results
        bd_batch = np.transpose(bd_batch, [1, 0])
        ac_batch = np.transpose(ac_batch, [1, 0])
        for idx, bd in enumerate(bd_batch):
            bd_batch[idx] = np.array([slot.decode('utf-8') for slot in bd])
        for idx, ac in enumerate(ac_batch):
            ac_batch[idx] = np.array([slot.decode('utf-8') for slot in ac])
        slots_batch = np.array([data.rebuild_slots_sequence(bd_seq, ac_seq) for bd_seq, ac_seq in zip(bd_batch, ac_batch)])
        for idx, intent in enumerate(intent_batch):
            intent_batch[idx] = intent.decode('utf-8')
        results = {
            'slots': slots_batch,
            'intent': intent_batch,
            'intent_confidence': intent_score_batch,
            'intent_attentions': attention_scores_intent,
            #'_bd': [bd.tolist() for bd in bd_batch],
            #'_ac': [ac.tolist() for ac in ac_batch]
        }
        return results