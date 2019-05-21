from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

import numpy as np

def export_model_for_mobile(model_name, input_node_name, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        model_name + '_graph.pbtxt')

    tf.train.Saver().save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, \
        False, 'out/' + model_name + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_name], [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

def main():
    datasetPref = np.loadtxt("alarm2.csv", delimiter=",")
    X = datasetPref[:,0:4]
    Y = datasetPref[:,4]

    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer="Adam")
    model.fit(X, Y, batch_size=len(X), epochs=10000)
    model.save("test.model")

    export_model_for_mobile('xor_nn', "dense_1_input", "dense_2/BiasAdd")
    model.summary()

if __name__=='__main__':
    main()
