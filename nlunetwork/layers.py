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

def monodirectional_rnn(inputs, hidden_size, cell_type='lstm', sequence_lengths=None):
    cell = rnn_cell(hidden_size, cell_type)
    encoder_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_lengths, dtype=tf.float32, time_major=True)

    if cell_type == 'lstm':
        encoder_final_state = final_state.h
    elif cell_type == 'gru':
        encoder_final_state = final_state
    else:
        raise ValueError(cell_type)
    
    return encoder_outputs, encoder_final_state


def bidirectional_rnn(inputs, hidden_size, cell_type='lstm', sequence_lengths=None):
    """Returns the """
    fw = rnn_cell(hidden_size, cell_type)
    bw = rnn_cell(hidden_size, cell_type)
    # The size of the following four variables：(T*B*D，T*B*D)，(B*D，B*D)
    (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)  = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw, cell_bw=bw,inputs=inputs, sequence_length=sequence_lengths, dtype=tf.float32, time_major=True)
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

def dense(inputs, output_size):
    return tf.layers.dense(inputs, output_size)

def decoder(inputs, hidden_size, cell_type, sequence_lengths, re_embed_fn, pad_with_index, output_vocab_size, max_sequence_length, attention_tensor_name, attention=True):
    """parameters:
    inputs: the input tensor of shape (time, batch)
    hidden_size: rnn cell size 
    cell_type: 'lstm' or 'gru'
    sequence_lengths: real lenghts of the sentences
    re_embed_function: function to call to get the embedding values from word indexes
    pad_with_index: the word index of padding, used to provide back fixed-length sequences
    output_vocab_size: the size of the output dictionary
    max_sequence_length: the maximum length of the sentences, used for padding the results
    attention_tensor_name: a name to be given to the attention tensor in order to make it retrievable
    attention: whether to enable or disable it
    
    returns:
    outputs: a tensor of shape (time, batch) with the word ids of the output sequence
    attentions: a tensor of shape (batch, time, time) with the matrices of attention weights for the decoding """
    # get the tensor for batch size
    batch_size_tensor = tf.shape(inputs)[1]
    # Initial values to provide to the decoding stage
    # generate a tensor of batch_size * '<PAD>' for start of sentence.
    # This value will be passed to first iteration of decoding in place of the previous slot label
    sos_time_slice = tf.fill((batch_size_tensor,), pad_with_index)

    """Returns decoder_output, attention_matrix"""
    def sample_fn(time, outputs, state):
        """defines how to sample from the output of the RNN cell"""
        # take the argmax from the logits
        prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
        return prediction_id

    def initial_fn():
        """defines how to provide the input to the decoder RNN cell at time 0, together means BD + AC in a single step"""
        # get the embedded representation of the initial fake previous-output-label
        #sos_step_embedded = tf.one_hot(sos_time_slice, depth=hidden_size)
        sos_step_embedded = re_embed_fn(sos_time_slice)
        # then concatenate it with the encoder output at time 0
        initial_input = tf.concat((sos_step_embedded, inputs[0]), 1)
        initial_elements_finished = (0 >= sequence_lengths)  # all False at the initial step
        return initial_elements_finished, initial_input

    def next_inputs_fn(time, outputs, state, sample_ids):
        """defines how to provide the input to the RNN cell at timesteps>0"""
        # From the last output, represented by sample_ids, get its embedded value
        pred_embedding = re_embed_fn(sample_ids)
        # Now concatenate it with the output of the decoder at the current timestep.
        # This is the new input to the RNN cell
        next_inputs = tf.concat((pred_embedding, inputs[time]), 1)
        # Establish which samples in the batch have already finished the decoding
        elements_finished = (time >= sequence_lengths)  # this operation produces boolean tensor of [batch_size]
        # don't modify the state
        next_state = state
        return elements_finished, next_inputs, next_state

    # Decoding function
    def decode(helper):
        # The decoding recurrent cell
        cell = rnn_cell(hidden_size, cell_type)
        if attention:
            # Get the memory representation (for the attention) by making the
            # encoder outputs dimensions from (time, batch, hidden_size) to (batch, time, hidden_size)
            memory = tf.transpose(inputs, [1, 0, 2])
            # Use the BahdanauAttention on the memory
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=hidden_size, memory=memory,
                memory_sequence_length=sequence_lengths)
            # that gets wrapped inside the attention mechanism
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=hidden_size, alignment_history=True)
            # and gets wrapped inside a output projection wrapper (weights+biases),
            # to have an output with logits on the slot labels dimension
        else:
            # no attention
            attn_cell = cell
        # the output_vocab_size is needed to provide one-hot encodings of the labels as results
        # and to build this OutputProjectionWrapper
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(
            attn_cell, output_vocab_size
        )
        # Define the decoder by combining the helper with the RNN cell
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=out_cell, helper=helper,
            initial_state=out_cell.zero_state(
                dtype=tf.float32, batch_size=batch_size_tensor))
        # And finally perform the decode
        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, output_time_major=True,
            impute_finished=True, maximum_iterations=max_sequence_length
        )
        return final_outputs, final_state

    # Build the helper with the declared functions
    my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)
    outputs, dec_states = decode(my_helper)
    if attention:
        attention_scores = dec_states.alignment_history.stack()
    else:
        attention_scores = tf.fill((batch_size_tensor, max_sequence_length, max_sequence_length), 0.0)
    # rename the tensor for the caller
    attention_scores = tf.identity(attention_scores, name=attention_tensor_name)

    # outputs shaped (time, batch, output_size) but time is truncated to the longest in the batch, so need to pad
    # pad in the first dimension (time) by adding requested_size - actual size, in the second&third dimension nothing
    print(outputs.rnn_output)
    padding_size = [[0, max_sequence_length - tf.shape(outputs.rnn_output)[0]], [0, 0], [0, 0]]
    outputs_logits_padded = tf.pad(outputs.rnn_output, padding_size, constant_values=0)
    # and set the shape to (time_padded, batch)
    outputs_logits_padded.set_shape([max_sequence_length, outputs.rnn_output.get_shape()[1], outputs.rnn_output.get_shape()[2]])
    print(outputs_logits_padded)

    return outputs_logits_padded, attention_scores
