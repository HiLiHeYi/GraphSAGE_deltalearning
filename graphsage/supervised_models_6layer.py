import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


class SupervisedGraphsagesix(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
                 placeholders, features, adj, degrees,
                 layer_infos, concat=True, aggregator_type="mean",
                 model_size="small", sigmoid_loss=False, identity_dim=0, only_ppr=False,
                 **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
            self.embeds = tf.compat.v1.get_variable(
                "node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception(
                    "Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(
                features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [
            (0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend(
            [layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer1 = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        self.optimizer2 = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        self.optimizer3 = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        self.optimizer4 = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        self.optimizer5 = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate) 

        self.loss1 = 0
        self.loss2 = 0
        self.loss3 = 0
        self.loss4 = 0
        self.loss5 = 0

        self.build(only_ppr)

    def build(self, only_ppr):
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        num_samples = [
            layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators1 = self.aggregate(samples1, [self.features], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size) 

        # TODO
        # self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        self.layer1 = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x: x, logging = True, name = 'layer1')
        self.layer2 = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x: x, logging = True, name = 'layer2')
        self.layer3 = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x: x, logging = True, name = 'layer3')
        self.layer4 = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x: x, logging = True, name = 'layer4')
        self.layer5 = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x: x, logging = True, name = 'layer5')
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x: x, logging = True, name = 'layer6_predlayer')
        # TF graph management
        self.out1 = self.layer1(self.outputs1)
        self.outputs2, self.aggregators2 = self.aggregate(samples1, [self.out1], self.dims, num_samples,
                                                    support_sizes1, concat=self.concat, model_size=self.model_size) 
        self.out2 = self.layer2(self.outputs2)
        self.outputs3, self.aggregators3 = self.aggregate(samples1, [self.out2], self.dims, num_samples,
                                                    support_sizes1, concat=self.concat, model_size=self.model_size) 
        self.out3 = self.layer3(self.outputs3)
        self.outputs4, self.aggregators4 = self.aggregate(samples1, [self.out3], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size) 
        self.out4 = self.layer4(self.outputs4)
        self.outputs5, self.aggregators5 = self.aggregate(samples1, [self.out4], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)   
        self.out5 = self.layer5(self.outputs5)
        self.outputs6, self.aggregators = self.aggregate(samples1, [self.out5], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size) 
        self.node_preds = self.node_pred(self.outputs6)


        ### ouputs in loggings
        # tf.compat.v1.summary.histogram('outputs layer 1', self.out1)
        # tf.compat.v1.summary.histogram('outputs layer 2', self.out2)
        # tf.compat.v1.summary.histogram('outputs layer 3', self.out3)
        # tf.compat.v1.summary.histogram('outputs layer 4', self.out4)
        # tf.compat.v1.summary.histogram('outputs layer 5', self.out5)
        # tf.compat.v1.summary.histogram('outputs layer 6 (predictions)', self.node_preds)

        ### inputs of layer ? output of aggregate funktion
        # tf.compat.v1.summary.histogram('outputs aggregate layer 1', self.outputs1)                           
        # tf.compat.v1.summary.histogram('outputs aggregate layer 2', self.outputs2)  
        # tf.compat.v1.summary.histogram('outputs aggregate layer 3', self.outputs3)  
        # tf.compat.v1.summary.histogram('outputs aggregate layer 4', self.outputs4)   
        # tf.compat.v1.summary.histogram('outputs aggregate layer 5', self.outputs5)  
        # tf.compat.v1.summary.histogram('outputs aggregate layer 6', self.outputs6)    

        self._loss1(only_ppr)
        self._loss2(only_ppr)
        self._loss3(only_ppr)
        self._loss4(only_ppr)
        self._loss5(only_ppr)
        self._loss(only_ppr)

        
        grads_and_vars1 = self.optimizer1.compute_gradients(self.loss1)
        clipped_grads_and_vars1 = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars1]
        self.grad1, _ = clipped_grads_and_vars1[0]
        # self.opt_op1 = self.optimizer1.apply_gradients(clipped_grads_and_vars1)
        
        grads_and_vars2 = self.optimizer2.compute_gradients(self.loss2)
        clipped_grads_and_vars2 = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars2]
        self.grad2, _ = clipped_grads_and_vars2[0]
        # self.opt_op2 = self.optimizer2.apply_gradients(clipped_grads_and_vars2)
        
        grads_and_vars3 = self.optimizer3.compute_gradients(self.loss3)
        clipped_grads_and_vars3 = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars3]
        self.grad3, _ = clipped_grads_and_vars3[0]
        # self.opt_op3 = self.optimizer3.apply_gradients(clipped_grads_and_vars3)
        
        grads_and_vars4 = self.optimizer4.compute_gradients(self.loss4)
        clipped_grads_and_vars4 = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars4]
        self.grad4, _ = clipped_grads_and_vars4[0]
        # self.opt_op4 = self.optimizer4.apply_gradients(clipped_grads_and_vars4)
        
        grads_and_vars5 = self.optimizer5.compute_gradients(self.loss5)
        clipped_grads_and_vars5 = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars5]
        self.grad5, _ = clipped_grads_and_vars5[0]
        # self.opt_op5 = self.optimizer5.apply_gradients(clipped_grads_and_vars5)
        
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        self.preds = self.predict(only_ppr)

        tf.compat.v1.summary.histogram('grads layer 1', self.grad1) 
        tf.compat.v1.summary.histogram('grads layer 2', self.grad2) 
        tf.compat.v1.summary.histogram('grads layer 3', self.grad3) 
        tf.compat.v1.summary.histogram('grads layer 4', self.grad4) 
        tf.compat.v1.summary.histogram('grads layer 5', self.grad5) 
        tf.compat.v1.summary.histogram('grads layer 6', self.grad) 

    def _loss(self, only_ppr):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        if not only_ppr:
            # classification loss
            if self.sigmoid_loss:
                self.loss += tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
            else:
                self.loss += tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=tf.stop_gradient(self.placeholders['labels'])))
        else:
            self.loss += tf.reduce_mean(input_tensor=tf.keras.losses.mae(
                self.placeholders['labels'],
                self.node_preds))
        tf.compat.v1.summary.scalar('loss', self.loss)

    def _loss1(self, only_ppr):
        # Weight decay loss
        for aggregator in self.aggregators1:
            for var in aggregator.vars.values():
                self.loss1 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layer1.vars.values():
            self.loss1 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss1 += tf.reduce_mean(input_tensor=tf.keras.losses.mae(
            self.placeholders['labels'],
            self.out1))
        tf.compat.v1.summary.scalar('loss layer 1', self.loss1)

    def _loss2(self, only_ppr):
        # Weight decay loss
        for aggregator in self.aggregators2:
            for var in aggregator.vars.values():
                self.loss2 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layer2.vars.values():
            self.loss2 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss2 += tf.reduce_mean(input_tensor=tf.keras.losses.mae(
            self.placeholders['labels'],
            self.out2))
        tf.compat.v1.summary.scalar('loss layer 2', self.loss2)

    def _loss3(self, only_ppr):
        # Weight decay loss
        for aggregator in self.aggregators3:
            for var in aggregator.vars.values():
                self.loss3 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layer3.vars.values():
            self.loss3 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss3 += tf.reduce_mean(input_tensor=tf.keras.losses.mae(
            self.placeholders['labels'],
            self.out3))
        tf.compat.v1.summary.scalar('loss layer 3', self.loss3)

    def _loss4(self, only_ppr):
        # Weight decay loss
        for aggregator in self.aggregators4:
            for var in aggregator.vars.values():
                self.loss4 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layer4.vars.values():
            self.loss4 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss4 += tf.reduce_mean(input_tensor=tf.keras.losses.mae(
            self.placeholders['labels'],
            self.out4))
        tf.compat.v1.summary.scalar('loss layer 4', self.loss4)

    def _loss5(self, only_ppr):
        # Weight decay loss
        for aggregator in self.aggregators5:
            for var in aggregator.vars.values():
                self.loss5 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layer5.vars.values():
            self.loss5 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss5 += tf.reduce_mean(input_tensor=tf.keras.losses.mae(
            self.placeholders['labels'],
            self.out5))
        tf.compat.v1.summary.scalar('loss layer 5', self.loss5)


    def predict(self, only_ppr):
        if not only_ppr:
            if self.sigmoid_loss:
                return tf.nn.sigmoid(self.node_preds)
            else:
                return tf.nn.softmax(self.node_preds)
        else:
            return self.node_preds
            
