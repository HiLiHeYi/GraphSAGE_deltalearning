from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.supervised_models_6layer import SupervisedGraphsagesix
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import scipy.interpolate.interpnd

import networkx as nx
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

tf.compat.v1.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'graphsage_mean',
                    'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small",
                    "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '',
                    'prefix identifying training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0,
                   'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer(
    'samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer(
    'dim_1', 4, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer(
    'dim_2', 4, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True,
                     'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 14755, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')

flags.DEFINE_integer(
    'identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

# logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.',
                    'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000,
                     "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 512,
                     "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 1, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10,
                     "Maximum total number of iterations")

flags.DEFINE_boolean('ppr', False, 'compute PPR scores')
if FLAGS.ppr:
    flags.DEFINE_integer('top_k', 15, 'return top_k PPR scores')

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8


def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

# Define model evaluation function


def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                             feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)


def evaluate_ppr(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    preds, loss, nodes = sess.run([model.preds, model.loss, list(feed_dict_val)[1]],
                                  feed_dict=feed_dict_val)
    return nodes, loss, preds, labels, (time.time() - t_test)


def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
        model=FLAGS.model,
        model_size=FLAGS.model_size,
        lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(
            size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)


def incremental_evaluate_ppr(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, nodes = minibatch_iter.incremental_node_val_feed_dict(
            size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    return nodes, np.mean(val_losses), node_outs_val[0], labels, (time.time() - t_test)


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.compat.v1.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.compat.v1.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data, test_data=None, predis=[], nxtime=None):




    iteration = []
    valLoss = []
    trainLoss = []

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map = train_data[4]
    for _ in G.nodes():
        predis.append(np.ndarray(0))

    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G,
                                      id_map,
                                      placeholders,
                                      class_map,
                                      num_classes,
                                      batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree,
                                      context_pairs=context_pairs)
    adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler,
                                    FLAGS.samples_2, FLAGS.dim_2),
                           SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [
                SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos,
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [
            SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1)]
        # layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
        #                SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="gcn",
                                    model_size=FLAGS.model_size,
                                    concat=False,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="seq",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="maxpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="meanpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.compat.v1.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.compat.v1.Session(config=config)
    merged = tf.compat.v1.summary.merge_all()
    summary_writer = tf.compat.v1.summary.FileWriter(log_dir(), sess.graph)

    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer(),
             feed_dict={adj_info_ph: minibatch.adj})

    # Train model

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []
    losses = []
    full_time = time.time()
    full_train_t = 0
    train_adj_info = tf.compat.v1.assign(adj_info, minibatch.adj)
    val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)
    f = open(log_dir() + "results.txt", "w")
    z = open(log_dir() + "sample.txt", "w")
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        f.write('Epoch: %04d' % (epoch + 1) + "\n")
        epoch_val_costs.append(0)

        #############################################################################################################
        #### some debugs ############################################################################################
        # feed_dict, labels, b_nodes = minibatch.next_minibatch_feed_dict()
        # nodes, val_cost, y_pred, y_true, duration = evaluate_ppr(sess, model, minibatch, FLAGS.validate_batch_size)
        # print("Pre Loss: ", y_pred[0], y_pred[1])
        # outs = sess.run([merged, model.opt_op, model.loss, model.preds, model.placeholders['labels'], model.node_preds], feed_dict=feed_dict)
        # summary_writer.add_summary(outs[0], total_steps)
        # total_steps += 1
        # nodes, val_cost, y_pred, y_true, duration = evaluate_ppr(sess, model, minibatch, FLAGS.validate_batch_size)
        # print("Loss: ", outs[2], y_pred[0], y_pred[1])
        #############################################################################################################
        #############################################################################################################


        while not minibatch.end():
            # Construct feed dictionary
            full_train_temp = time.time()
            batch = None
            feed_dict, labels = minibatch.next_minibatch_feed_dict(batch)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, 
                           model.loss,
                             model.preds], feed_dict=feed_dict)
            full_train_t = full_train_t + (time.time() - full_train_temp)
            train_cost = outs[2]




            losses.append(outs[2])

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    if not FLAGS.ppr:
                        val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(
                            sess, model, minibatch, FLAGS.batch_size)
                    else:
                        nodes, val_cost, y_pred, y_true, duration = incremental_evaluate_ppr(
                            sess, model, minibatch, FLAGS.batch_size)
                else:
                    if not FLAGS.ppr:
                        val_cost, val_f1_mic, val_f1_mac, duration = evaluate(
                            sess, model, minibatch, FLAGS.validate_batch_size)
                    else:
                        nodes, val_cost, y_pred, y_true, duration = evaluate_ppr(
                            sess, model, minibatch, FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            # Print results
            avg_time = (avg_time * total_steps +
                        time.time() - t) / (total_steps + 1)

            if not FLAGS.ppr:
                if total_steps % FLAGS.print_every == 0:
                    train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                          "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                          "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                          "time=", "{:.5f}".format(avg_time))
            else:
                if total_steps % FLAGS.print_every == 0:
                    # TODO debug
                    # print("Batch nodes: ", feed_dict[list(feed_dict)[1]])
                    iteration.append(epoch)
                    trainLoss.append(train_cost)
                    valLoss.append(val_cost)
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost)
                          )
                    # TODO DEBUG PRINTS
                    # for n, p, t in zip(nodes, y_pred, y_true):
                    #     print(f"Node: {n}: pred: {p}; true: {t}")

                    print("val_loss=", "{:.5f}".format(val_cost),
                          "time=", "{:.5f}".format(avg_time))
                nodes_t, val_cost_t, y_pred_t, y_true_t, duration_t = evaluate_ppr(
                    sess, model, minibatch, FLAGS.validate_batch_size)
                f.write("Batch nodes: " +
                        str(feed_dict[list(feed_dict)[1]]) + "\n")
                f.write("Iter:" + '%04d' % iter + "train_loss=" +
                        "{:.5f}".format(train_cost) + "\n")
                # for n, p, t in zip(nodes_t, y_pred_t, y_true_t):
                #     f.write(f"Node: {n}: pred: {p}; true: {t}\n")
                #     predis[n] = np.append(predis[n], p)

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break

    print("Optimization Finished!")
    sess.run(val_adj_info.op)
    if not FLAGS.ppr:
        val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(
            sess, model, minibatch, FLAGS.batch_size)
        print("Full validation stats:",
              "loss=", "{:.5f}".format(val_cost),
              "f1_micro=", "{:.5f}".format(val_f1_mic),
              "f1_macro=", "{:.5f}".format(val_f1_mac),
              "time=", "{:.5f}".format(duration))
        with open(log_dir() + "val_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                     format(val_cost, val_f1_mic, val_f1_mac, duration))

        print("Writing test set stats to file (don't peak!)")
        val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(
            sess, model, minibatch, FLAGS.batch_size, test=True)
        with open(log_dir() + "test_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                     format(val_cost, val_f1_mic, val_f1_mac))

    else:
        nodes, val_cost, y_pred, y_true, duration = incremental_evaluate_ppr(
            sess, model, minibatch, FLAGS.batch_size)
        print("Full validation stats:",
              "loss=", "{:.5f}".format(val_cost),
              "time=", "{:.5f}".format(duration))
        with open(log_dir() + "val_stats.txt", "w") as fp:
            fp.write("loss={:.5f} time={:.5f}".
                     format(val_cost, duration))
        print("Writing test set stats to file (don't peak!)")


        #########################################################################
        ###### creating some plots ##############################################
        # fig = plt.figure(figsize=([8,4]))
        # np.save("loss_idem2_rnn.npy", losses, allow_pickle=True) 
        # idem0 = np.load("PPR_res/loss_idem0_rnn.npy")
        # idem1 = np.load("PPR_res/loss_idem1_rnn.npy")
        # idem2 = np.load("loss_idem2_rnn.npy")
        # plt.plot(idem0, label="identity_dim = 0")
        # plt.plot(idem1, label="identity_dim = 1")
        # plt.plot(idem2, label="identity_dim = 2")
        # plt.plot(trainf, label="trainable feature")
        # plt.plot(idem, label="identity_dim=1")
        # plt.ylabel("Train loss", fontsize=12)
        # plt.xlabel("Epoch", fontsize=12)
        # plt.legend()
        # plt.tight_layout()

        # fig.text(.95, .85, "full duration = " + "{:.5f}".format(time.time()-full_time), fontsize=12,
        #          transform=plt.gca().transAxes, horizontalalignment='right',
        #          verticalalignment='bottom')
        # fig.text(.95, .80, "full_train_time = " + "{:.5f}".format(full_train_t), fontsize=12,
        #          transform=plt.gca().transAxes, horizontalalignment='right',
        #          verticalalignment='bottom')

        # fig.text(.95, .75, "only predict = " + "{:.5f}".format(duration), fontsize=12,
        #          transform=plt.gca().transAxes, horizontalalignment='right',
        #          verticalalignment='bottom')

        # fig.text(.95, .7, "nx_time = " + "{:.5f}".format(nxtime), fontsize=12,
        #          transform=plt.gca().transAxes, horizontalalignment='right',
        #          verticalalignment='bottom')
        # fig.savefig("./losses/4_nodes/losses_mae_rnn.pdf")
        # plt.show()
        ########################################################################
        ########################################################################

    f.close()
    z.close()
    if FLAGS.ppr:
        batch = [node for node in G.nodes()]
        feed_dict, labels = minibatch.next_minibatch_feed_dict(batch)
        # print(feed_dict)

        for key, value in feed_dict.items() :
            print("ok")
            print (key, value)
        # print(feed_dict)
        # print(len(labels))
        # print(len(labels[20]))
        ########## ?? what is out         
        outs = sess.run([model.preds], feed_dict=feed_dict)
        print(len(outs[0]))
        ppr_vals = {}
        for n, p, t in zip(batch, outs[0], labels):
            ppr_vals[n] = [p[0], t[0]]
            # ppr_vals[n] = [p, t]


        # print(labels[6999])
        # print(labels[7000])
        # print(labels[7001])
        # np.savetxt('ppr_prediction.csv', ppr_vals, delimiter=',')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(iteration, valLoss, color ='tab:blue',label='train loss')
        ax.plot(iteration, trainLoss, color ='tab:orange',label='validation loss')
        ax.legend(['train loss', 'validation loss '])
        ax.set_xlabel("number of epochs")
        ax.set_ylabel("loss")
        ax.set_title('loss for singel feature with source node 7000')
        # plt.show()

        print
        return_query(0, ppr_vals, FLAGS.top_k, G)

def return_query(query_node, ppr_vals, k, graph):
    ppr_sorted_full = sorted(
        ppr_vals.items(), key=lambda x: x[1][0], reverse=True)
    ppr_sorted = ppr_sorted_full[:k]
    ppr_sorted_real = sorted(
        ppr_vals.items(), key=lambda x: x[1][1], reverse=True)

    ########################################################################
    ###### saving for debug/plots ##########################################
    # np.save("PPR_res/n_" + str(FLAGS.identity_dim) + "-d_" +
    #         str(FLAGS.dropout) +"_true.npy", ppr_sorted_real, allow_pickle=True)
    # np.save("PPR_res/n_" + str(FLAGS.identity_dim) + "-d_" +
    #         str(FLAGS.dropout) +"_pred.npy", ppr_sorted_full, allow_pickle=True)
    ########################################################################
    ########################################################################


    # printing results
    print(f"top PPR for Node {query_node} ({len(ppr_sorted)}):")
    for i in range(len(ppr_sorted)):
        print(
            f"   {i+1}. Node {ppr_sorted[i][0]} ({ppr_sorted_real[i][0]}): {ppr_sorted[i][1][0]} (expected: {ppr_sorted[i][1][1]})")
    print(f"real top PPR for Node {query_node} ({len(ppr_sorted)}):")
    for i in range(len(ppr_sorted)):
        print(
            f"   {i+1}. Node {ppr_sorted_real[i][0]} ({ppr_sorted[i][0]}): {ppr_sorted_real[i][1][1]} (predicted: {ppr_sorted_real[i][1][0]})")



"""
small test graph for testing our algorithm
4-nodes; 3-edges
"""
def generate_data(query_node):
    G = nx.Graph()
    G.add_nodes_from(range(4),
                     label=[0],
                     feature=[0],
                     event_type="update",
                     entity_type="node",
                     operation_type="insertion",
                     val=False,
                     test=False
                     )
    for i in range(len(G.nodes())):
        G.node[i]["id"] = i

    G.add_edges_from([(0, 1), (1, 2), (2, 3)],
                     event_type="update",
                     entity_type="link",
                     operation_type="insertion"
                     )
    for edge in G.edges():
        G[edge[0]][edge[1]]['source'] = edge[0]
        G[edge[0]][edge[1]]['target'] = edge[1]
    print(nx.info(G))

    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = False
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    f = plt.figure()
    nx.draw_networkx(G)
    plt.show(block=False)
    # plt.savefig("graph.png")
    plt.pause(0.001)

    personalization = {}

    id_map = {}
    for node in G.nodes():
        id_map[node] = node
        if node == query_node:
            personalization[node] = 1
            G.node[node]['feature'] = [1]
        else:
            personalization[node] = 0
            G.node[node]['feature'] = [0]

    print(personalization)

    t = time.time()
    ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
    t = time.time() - t

    features = np.ndarray(shape=[0, 1])

    class_map = {}

    for i in ppr:
        class_map[i] = [ppr[i]]
        features = np.append(features, np.array(
            [[personalization[i]]]), axis=0)
    print(f"PPRs: {ppr}")
    return G, features, id_map, [], class_map, t


def generate_data_ff(data, query_node):
    G = data[0]
    personalization = {}

    id_map = {}
    for node in G.nodes():
        id_map[node] = node
        if node == query_node:
            personalization[node] = 1
        else:
            personalization[node] = 0

    print("Calculating PPRs..")
    t = time.time()
    # ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
    ppr = np.load("PPRs_toy-ppi.npy", allow_pickle=True).item()
    t = time.time() - t
    print("Calculated PPRs..")

    # np.save("PPRs_toy-ppi.npy", ppr)

    features = np.ndarray(shape=[0, 1], dtype=int)

    class_map = {}
    for i in ppr:
        class_map[i] = [ppr[i]]
        features = np.append(features, np.array(
            [[personalization[i]]]), axis=0)

    f = plt.figure()
    if len(G.nodes()) < 30:
        nx.draw_networkx(G)
        plt.show(block=False)
        plt.pause(0.001)
    # print(sample)
    return G, features, data[2], data[3], class_map, t


def gnp_random_connected_graph(n, p, seed=None):
    from itertools import combinations, groupby
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    adapted: https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx?noredirect=1&lq=1
    """
    if not seed is None:
        random.seed(seed)
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n),
                    label=[0],
                    feature=[0],
                    event_type="update",
                    entity_type="node",
                    operation_type="insertion",
                    val=False,
                    test=False
                    )
    for i in range(len(G.nodes())):
        G.node[i]["id"] = i
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G


"""
Generate random connected graph with 50 nodes (edge propability 0.01)
Select k-hop neighborhood until reaching at least 30 nodes
"""
def generate_data_random(query_node):
    G = gnp_random_connected_graph(50, 0.01, seed=123)
    H = nx.Graph(G.subgraph( nx.single_source_shortest_path_length(G,0, cutoff=1)))
    i=2
    while len(H.nodes())<30:
        H = nx.Graph(G.subgraph( nx.single_source_shortest_path_length(G,0, cutoff=i)))
        i+=1
    print(len(G.nodes()), len(H.nodes()))
    G=H
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = False
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    f = plt.figure()
    nx.draw_networkx(G)
    plt.show(block=False)
    # plt.savefig("graph.png")
    plt.pause(0.001)

    personalization = {}

    id_map = {}
    for node in G.nodes():
        id_map[node] = node
        if node == query_node:
            personalization[node] = 1
            G.node[node]['feature'] = [1]
        else:
            personalization[node] = 0
            G.node[node]['feature'] = [0]

    print(personalization)
    id_map = {}
    i=0
    for node in G.nodes():
        id_map[node] = i
        i+=1

    t = time.time()
    ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
    t = time.time() - t

    features = np.ndarray(shape=[0, 1])

    class_map = {}

    for i in ppr:
        class_map[i] = [ppr[i]]
        features = np.append(features, np.array(
            [[personalization[i]]]), axis=0)
    print(f"PPRs: {ppr}")
    # fname = 'example.json'
    # json.dump(dict(nodes=[[n, G.node[n]] for n in G.nodes()],edges=[[u, v, G.edge[u][v]] for u,v in G.edges()]),
    # open(fname, 'w'), indent=2)
    return G, features, id_map, [], class_map, t


"""
Just a test function for toy data!!!
""" 
def generate_data_neighborhood(train_data, query_node):
    G = train_data[0]
    G = nx.Graph(G.subgraph(nx.single_source_shortest_path_length(G,0, cutoff=1)))
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = False
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    f = plt.figure()
    nx.draw_networkx(G)
    plt.show(block=False)
    # plt.savefig("graph.png")
    plt.pause(0.001)

    personalization = {}

    id_map = {}
    for node in G.nodes():
        id_map[node] = node
        if node == query_node:
            personalization[node] = 1
            G.node[node]['feature'] = [1]
        else:
            personalization[node] = 0
            G.node[node]['feature'] = [0]

    print(personalization)
    id_map = {}
    i=0
    for node in G.nodes():
        id_map[node] = i
        i+=1

    t = time.time()
    ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
    t = time.time() - t

    features = np.ndarray(shape=[0, 1])

    class_map = {}

    for i in ppr:
        class_map[i] = [ppr[i]]
        features = np.append(features, np.array(
            [[personalization[i]]]), axis=0)

    print(f"PPRs: {ppr}")
    return G, features, id_map, [], class_map, t



def main(argv=None):
    # print("Loading training data..")
    # if FLAGS.ppr:
    #     ## generate own random graph - PPR scores etc.
    #     train_data = generate_data_random(0)  
    #     #  G = train_data[0]
    #     #  features = train_data[1]
    #     #  id_map = train_data[2]
    #     # class_map = train_data[4]

    #     # print(train_data[2])
    #     # print(train_data[4])
    #     G = train_data[0]
    #     fname = 'example.json'
    #     json.dump(dict(nodes=[[n, G.node[n]] for n in G.nodes()],edges=[[u, v, G.edge[u][v]] for u,v in G.edges()]),
    #     open(fname, 'w'), indent=2)
    #     # ## small test graph 4 nodes
    #     # train_data = generate_data(0) 

    #     ## small state on toy data - 1 hop
    #     # train_data = generate_data_neighborhood(generate_data_ff(load_data(FLAGS.train_prefix), 0)[:-1], 0) 

    #     train(train_data[:-1], nxtime=train_data[-1])
    # else:
    #     train_data = load_data(FLAGS.train_prefix)
    #     train(train_data)

    train_data = load_data(FLAGS.train_prefix)
    # print(train_data[0])#G
    # G = train_data[0]
    # fname = 'example.json'
    # json.dump(dict(nodes=[[n, G.node[n]] for n in G.nodes()],edges=[[u, v, G.edge[u][v]] for u,v in G.edges()]),
    # open(fname, 'w'), indent=2)
    # print(train_data[1])#Feature
    # print(train_data[2])
    # print(train_data[3])

    train(train_data)
    print("Done loading training data..")



if __name__ == '__main__':
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    # tf.compat.v1.disable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    tf.compat.v1.app.run()




