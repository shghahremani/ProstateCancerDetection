import tensorflow as tf
import keras.backend as K
from tensorflow.python.framework import graph_util
import argparse


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        for i, n in enumerate(node.input):
            print('\t %d   %s' % (i, n))

def prepare_graph_for_freezing(model_folder):
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    saver = tf.train.Saver()

    with tf.Session(config= tf.ConfigProto(allow_soft_placement=True)) as sess:
        K.set_session(sess)
        saver.restore(sess)
        tf.gfile.MakeDirs( model_folder +'freeze')
        saver.save(sess , model_folder +'freeze/checkpoint' ,global_step=0)

def freeze_graph(model_folder,output_nodes):
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    print(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    absolute_model_folder= "/".join(input_checkpoint.split('/')[:-1])
    output_graph= absolute_model_folder+ "/frozen_model.pb"
    # in order to freeze the graph, we need to tell TF which nodes will be used at utput
    # We can get the node names from our previous keras model definition
    output_node_name= output_nodes

    # if we trained the model on GPU, we want to be sure there are no explicit GPU directives on the graph nides
    clear_devices= True
    new_saver= tf.train.import_meta_graph(input_checkpoint+ '.meta' , clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess2:
        print(input_checkpoint)
        new_saver.restore(sess2, input_checkpoint)

        graph_util.remove_training_nodes(
            input_graph_def,
            protected_nodes=None
        )
        # Since we are freeing the models, we want to turn all the rainable variables to constants
        output_graph_def= graph_util.convert_variables_to_constants(sess2,  # This is used to retrieve the weights
                  input_graph_def,  # The graph def is used to retrieve the nodes
                  output_node_name)  # The output node names are used to select the useful nodes
        # We can now freze the graph and save it
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph" % len(output_graph_def.node))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Freezing the Network')
    parser.add_argument('Folder',help='Folder containing meta file',default="results/freeze")
    parser.add_argument('OutputNodes', nargs='*',help='Name of the outout nodes',default="multi_label_output/Reshape")

    args = parser.parse_args()
    print(args)

    tf.reset_default_graph()
    myargs=vars(args)
    print(myargs)
    # prepare_graph_for_freezing("freeze/")
    freeze_graph(myargs['Folder'],myargs['OutputNodes'])

