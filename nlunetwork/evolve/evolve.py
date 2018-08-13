from random import random

import neat
from nlunetwork.evolve import visualize
from nlunetwork import net_builder

GENERATIONS = 50

nlp = TODO instantiate model

def eval_genomes(genomes, config):
    #print(config.genome_config.__dict__)
    for genome_id, genome in genomes:
        # create the graph
        graph = net_builder.build(genome)

        # TODO this is just an experiment, there you should create the network and evaluate it
        genome.fitness = (len([g for g in genome.connections.values() if g.enabled])) * 1.0
        for (id, g) in genomes:
            #print(id)
            #print(id, g.nodes, g.connections)
            #print('genome')
            for node_name, value in g.nodes.items() :
                #print(node_name, value.type_of_layer)
                pass
            for (src, dst), value in g.connections.items():
                #print('{}->{}:{}'.format(src, dst, value.enabled))
                pass
        #exit(1)
        #net = neat.nn.FeedForwardNetwork.create(genome, config)
        #for xi, xo in zip(xor_inputs, xor_outputs):
        #    output = net.activate(xi)
        #    genome.fitness -= (output[0] - xo[0]) ** 2


# Load configuration.
config = neat.Config(neat.DefaultGenome_deep, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner = p.run(eval_genomes, GENERATIONS)
visualize.draw_net(config, winner, view=True, prune_unused=True)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
#print('\nOutput:')
#winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

