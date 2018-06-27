import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, GRUCell

def rnn_cell(hidden_size, cell_type='lstm'):
    if cell_type == 'lstm':
        cell = BasicLSTMCell
    elif cell_type == 'gru':
        cell = GRUCell
    else:
        raise ValueError(cell_type)
    return cell(hidden_size)

def monodirectional_rnn(inputs, hidden_size, cell_type='lstm', sequence_length=None):
    cell = rnn_cell(hidden_size, cell_type)
    encoder_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_length, dtype=tf.float32, time_major=True)

    if cell_type == 'lstm':
        encoder_final_state = final_state.h
    elif cell_type == 'gru':
        encoder_final_state = final_state
    else:
        raise ValueError(cell_type)
    
    return encoder_outputs, encoder_final_state


def bidirectional_rnn(inputs, hidden_size, cell_type='lstm', sequence_length=None):
    """Returns the """
    fw = rnn_cell(hidden_size, cell_type)
    bw = rnn_cell(hidden_size, cell_type)
    # The size of the following four variables：(T*B*D，T*B*D)，(B*D，B*D)
    (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)  = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw, cell_bw=bw,inputs=inputs, sequence_length=sequence_length, dtype=tf.float32, time_major=True)
    # The encoder outputs are the concatenation of the outputs of each direction.
    # The concatenation is done on the third dimension. Dimensions: (time, batch, hidden_size)
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
    # Also concatenate things for the final state. Dimensions: (batch, hidden_size)
    if cell_type == 'lstm':
        encoder_final_state = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
    elif cell_type == 'gru':
        encoder_final_state = tf.concat((encoder_fw_final_state, encoder_bw_final_state), 1)
    else:
        raise ValueError(cell_type)
    
    return encoder_outputs, encoder_final_state
    
def feedforward(inputs, input_size, output_size, scope_name):
    # input size could be estimated on inputs?
    with tf.variable_scope(scope_name):
        # Define the weights and biases to perform the output projection
        W = tf.get_variable('W', initializer=tf.random_uniform([input_size, output_size], -0.1, 0.1), dtype=tf.float32)
        b = tf.get_variable("b", initializer=tf.zeros([output_size]), dtype=tf.float32)
    logits = tf.add(tf.matmul(inputs, W), b)
    return logits