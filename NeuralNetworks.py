from collections import defaultdict
import tensorflow as tf
from AppUtils import logger, TaskReporter
import numpy as np
import os
from time import time
from datetime import datetime


class NeuralNetworks:

    def __init__(self, save_model_path):
        '''

        :param inputs: Feature data from DataCenter, X
        :param targets: Target data from DataCenter, Y
        :param save_model_path: Path of directory saving the metal model
        :param model_paras: model parameters for NN, this is controllable from app.py
        '''
        self.save_model_path = save_model_path
        self.model_paras = {
            'feature_dimension': 18,
            'lr': 0.005,
            'keep_prob': 0.5,
            'n_output_classes': 3,
            'threshold_negative': 0.35,
            'threshold_positive': 0.35,
            'hidden_layer_units': [4096, 2048],
        }
        self.batch_size = 32
        self.epochs = 80
        self.iteration_display = 10
        path_meta = "{}/meta.npy".format('./Model')
        if not os.path.exists("./Model"):
            os.makedirs("./Model")
        np.save(path_meta, self.model_paras)

    def __split_data(self, inputs, targets):
        '''
        Split data into Train, Val, Test according to 2:1:1
        :param inputs:
        :param targets:
        :return:
        '''
        cut_train = int(0.5*len(inputs))
        cut_validation = int(0.75*len(inputs))
        input_train, target_train = inputs[:cut_train], targets[:cut_train]
        input_val, target_val = inputs[cut_train:cut_validation], targets[cut_train:cut_validation]
        input_test, target_test = inputs[cut_validation:], targets[cut_validation:]
        return input_train, target_train, input_val, target_val, input_test, target_test

    def __get_batch(self, inputs, targets):
        '''
        Chunk the training data based on batch_size
        :param inputs:
        :param targets:
        :return:
        '''
        assert len(inputs) == len(targets), 'Inputs and targets have different size'
        for id in range(0, len(inputs), self.batch_size):
            x = inputs[id: id + self.batch_size]
            y = targets[id: id + self.batch_size]
            yield x, y

    def __build_inputs(self):
        '''

        :return: tensors of placeholders
        '''
        # Dimension:[Batch_size(default), #feature]
        inputs = tf.placeholder(tf.float32, [None, self.model_paras['feature_dimension']], name='inputs')
        # 0: positive, 1: neutral, 2: negative
        targets = tf.placeholder(tf.int32, [None, 3], name='targets')
        learning_rate = tf.placeholder(tf.float32, name='lr')
        # drop out rate
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        is_training = tf.placeholder(tf.bool, name='is_training')
        return inputs, targets, learning_rate, keep_prob, is_training

    def __build_output(self, inputs, keep_prob, is_training):
        """

        :param inputs: input tensor
        :param keep_prob: dropout tensor
        :param is_training: bool flag for batch normalization
        :return: output logits and probability of output classes
        """
        assert isinstance(self.model_paras['hidden_layer_units'], list), "hidden_layer_units should be a list of int"
        # iterating hidden layers, usually 2 hidden layers are already enough for the data
        for number_hidden_units in self.model_paras['hidden_layer_units']:
            layer = tf.layers.dense(inputs=inputs,
                                    units=number_hidden_units,
                                    activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            layer = tf.layers.batch_normalization(layer, training=is_training)
            layer = tf.nn.relu(layer)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob)
            inputs = layer

        logits = tf.layers.dense(inputs=inputs,
                                 units=self.model_paras['n_output_classes'],
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        label_prob = tf.nn.softmax(logits, name='label_prob')
        return logits, label_prob

    def __build_loss(self, logits, targets):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        loss = tf.reduce_mean(loss, name="loss")
        return loss

    def __build_optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)
        return optimizer

    def __build_graph(self):
        """
        Check if graph already built, if not, re-build the graph
        :return: flag for __load_graph to determine whether needs to import meta graph
        """
        need_build_graph = False
        logger.info("Checking model graph...")
        if os.path.exists("{}.meta".format(self.save_model_path)):
            logger.info("Graph existed, ready to be reloaded...")
        else:
            need_build_graph = True
            logger.info("Graph not existed, create a new graph and save to {}".format(self.save_model_path))
            tf.reset_default_graph()
            inputs, targets, learning_rate, keep_prob, is_training = self.__build_inputs()
            logits, label_prob = self.__build_output(inputs=inputs, keep_prob=keep_prob, is_training=is_training)
            loss = self.__build_loss(logits=logits, targets=targets)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = self.__build_optimizer(loss=loss, learning_rate=learning_rate)

            # Saving the meta model
            saver = tf.train.Saver()
            if not os.path.exists('./Model'):
                os.mkdir('./Model')
            logger.info("Save the model to {}".format(self.save_model_path))
            with tf.Session(graph=tf.get_default_graph()) as sess:
                sess.run(tf.global_variables_initializer())
                saver.save(sess, self.save_model_path)

        logger.info("Finish building model graph!")
        return need_build_graph

    def __load_graph(self, sess, need_build_graph):
        """
        Loading the meta graph and restore the trained paras if possible.
        :param sess:
        :param need_build_graph:
        :return: dictionary of tensors
        """
        graph = defaultdict()
        # if the model graph just built from __build_graph, then no need to import meta graph again,
        # else, import the pre-built model graph.
        if need_build_graph:
            loader = tf.train.Saver()
        else:
            loader = tf.train.import_meta_graph(self.save_model_path + '.meta')

        graph['inputs'] = sess.graph.get_tensor_by_name("inputs:0")
        graph['targets'] = sess.graph.get_tensor_by_name("targets:0")
        graph['lr'] = sess.graph.get_tensor_by_name("lr:0")
        graph['keep_prob'] = sess.graph.get_tensor_by_name("keep_prob:0")
        graph['is_training'] = sess.graph.get_tensor_by_name("is_training:0")
        graph['label_prob'] = sess.graph.get_tensor_by_name("label_prob:0")
        graph['loss'] = sess.graph.get_tensor_by_name('loss:0')
        graph['optimizer'] = sess.graph.get_operation_by_name("optimizer")

        logger.info("model is ready, good to go!")
        check_point = tf.train.latest_checkpoint('checkpoints')
        # if no check_point found, means we need to start training from scratch, just initialize the variables.
        if not check_point:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())
        else:
            logger.info("check point path:{}".format(check_point))
            loader.restore(sess, check_point)
        return graph

    def __predict_label(self, label_probs):
        """
        Predict the label based on two steps.
        1. Class with largest prob.
        2. If class is not neutral, then compare with corresponding threshold
        :param label_probs:
        :return: predicted label
        """
        def driver(prob):
            candidate = np.argmax(prob)
            if candidate == 0 and prob[0] > self.model_paras['threshold_positive']:
                return 0
            elif candidate == 2 and prob[2] > self.model_paras['threshold_negative']:
                return 2
            else:
                return 1

        labels = list(map(driver, label_probs))
        return labels

    def __print_stat(self, labels_predict, labels_true, validationOrTest):
        """
        Print statistics during training.
        :param labels_predict: Estimated Y
        :param labels_true: True Y
        :param validationOrTest: If using validation or test set.
        :return:
        """
        labels_true = np.argmax(labels_true, axis=1)
        correct_pred = np.equal(labels_predict, labels_true).astype(int)
        accuracy = np.mean(correct_pred)
        logger.info("{} data Accuracy:{}".format(validationOrTest, accuracy))

    @TaskReporter("Training model...")
    def train(self, inputs, targets):
        input_train, target_train, input_val, target_val, input_test, target_test = self.__split_data(inputs, targets)

        need_build_graph = self.__build_graph()
        iteration = 0
        with tf.Session(graph=tf.get_default_graph()) as sess:
            graph = self.__load_graph(sess, need_build_graph)
            saver = tf.train.Saver()
            for epoch in range(self.epochs):
                for x, y in self.__get_batch(input_train, target_train):
                    start = time()
                    feed_dict={
                        graph['inputs']: x,
                        graph['targets']: y,
                        graph['lr']: self.model_paras['lr'],
                        graph['keep_prob']: self.model_paras['keep_prob'],
                        graph['is_training']: True
                    }
                    batch_loss, _ = sess.run([graph['loss'], graph['optimizer']], feed_dict=feed_dict)
                    end = time()
                    logger.info('Epoch: {}/{}... '.format(epoch+1, self.epochs) +
                                'Training Step: {}... '.format(iteration) +
                                'Training loss: {:.4f}... '.format(batch_loss) +
                                '{:.4f} sec/batch'.format((end-start)))

                    # Print validation result.
                    if iteration % self.iteration_display == 0:
                        feed_dict_val = {
                            graph['inputs']: input_train,
                            graph['keep_prob']: 1.0,
                            graph['is_training']: False
                        }
                        prob = sess.run(graph['label_prob'], feed_dict=feed_dict_val)
                        label_predict = self.__predict_label(prob)
                        self.__print_stat(label_predict, target_train, "Validation")
                    # if iteration % self.save_every_n == 0:
                    #     saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.lstm_size))
                    iteration += 1

            feed_dict_test = {
                graph['inputs']: input_test,
                graph['keep_prob']: 1.0,
                graph['is_training']: False
            }
            prob = sess.run(graph['label_prob'], feed_dict=feed_dict_test)
            label_predict = self.__predict_label(prob)
            self.__print_stat(label_predict, target_test, "Test")

            time_stamp = datetime.now()
            checkpoint_marker = "{}_{}_{}_{}".format(time_stamp.year, time_stamp.month, time_stamp.day, time_stamp.hour)
            if not os.path.exists("./checkpoints"):
                os.mkdir("./checkpoints")
            saver.save(sess, "checkpoints/{}.ckpt".format(checkpoint_marker))

    def sample(self, current_transaction):
        """
        Inference step for the coming transaction.
        :param current_transaction:
        :return: predicted labels.
        """
        need_build_graph = self.__build_graph()
        with tf.Session(graph=tf.get_default_graph()) as sess:
            graph = self.__load_graph(sess, need_build_graph)
            assert graph['inputs'].get_shape().as_list()[-1] == current_transaction.shape[-1], \
                'new transaction should have same feature dimension as model'
            feed_dict={
                graph['inputs']: current_transaction,
                graph['keep_prob']: 1.0,
                graph['is_training']: False
            }
            prob = sess.run(graph['label_prob'], feed_dict=feed_dict)
            label_predict = self.__predict_label(prob)
        label_names = ["Positive", "Neutral", "Negative"]
        label_predict = [label_names[label] for label in label_predict]
        return label_predict