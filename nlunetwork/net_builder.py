"""This module builds the computational graph from the NEAT representation of nodes and links"""

def build(genome, nlp, tokenizer='space'):
    """This method builds the tensorflow graph from the given neat genome"""
    for node in genome.nodes:
        print(node)
    for connection in genome.connections:
        print(connection)
    # check the shape of the graph
    input_node_type = ['embeddings']
    output_node_type = ['out_single', 'out_seq']
    # for each node_name store the output tensor
    tf_created = {node_name: None for node_name in genome.nodes}
    # for each node name, a list of the incoming nodes
    nodes_inputs = {node_name: list([src for src, dst in genome.connections if dst == node_name]) for node_name in genome.nodes}
    print(nodes_inputs)
    while not all(tf_created.values()):
        # find a node to be created: all the input nodes are already created
        #ready = set([node_name for (node_name, tf_value) in tf_created.items() if (tf_value and all([tf_created[incoming] for incoming in nodes_inputs[node_name]]))])
        ready = [node_name for (node_name, incomings) in nodes_inputs.items() if (not tf_created[node_name] and all([tf_created[incoming] for incoming in incomings]))]
        print('ready nodes', ready)
        if not ready:
            print([c for c in genome.connections])
            raise RuntimeError('There is no ready node, check the graph structure')
        selected_name = ready.pop()
        selected_node = genome.nodes[selected_name]
        print('selected', selected_node)
        # build the layer
        if selected == 'embeddings':
            layer = FixedEmbeddings(tokenizer, language, nlp)
        tf_created[selected_name] = True # TODO put the actual tensor
    exit(1)