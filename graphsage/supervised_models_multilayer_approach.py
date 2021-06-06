import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


class SupervisedGraphsage(models.SampleAndAggregate):
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

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        self.a , self.b= None, None    
        self.build(only_ppr)

    def build(self, only_ppr):
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        num_samples = [
            layer_info.num_samples for layer_info in self.layer_infos]

        # Learning over layers: Concurrent learning
        # layer 1
        outputs1, aggregators1 = self.aggregate(samples1, [self.features], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)                            

        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x: x)

        # TF graph management
        self.node_preds1 = self.node_pred(outputs1)
        self.loss1 = self._loss2(self.node_preds1, aggregators1)

        # layer 1
        grads_and_vars = self.optimizer.compute_gradients(self.loss1)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        # layer 2
        outputs2, aggregators2 = self.aggregate(samples1, [self.node_preds1], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)
        self.node_preds2 = self.node_pred(outputs2)
        self.loss2 = self._loss2(self.node_preds2, aggregators2)

        
        # layer 2
        grads_and_vars = self.optimizer.compute_gradients(self.loss2)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        # layer 3
        outputs3, aggregators3 = self.aggregate(samples1, [self.node_preds2], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)
        self.node_preds3 = self.node_pred(outputs3)
        self.loss3 = self._loss2(self.node_preds3, aggregators3)

                # layer 3
        grads_and_vars = self.optimizer.compute_gradients(self.loss3)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        # layer 4
        outputs4, aggregators4 = self.aggregate(samples1, [self.node_preds3], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)
        self.node_preds4 = self.node_pred(outputs4)
        self.loss4 = self._loss2(self.node_preds4, aggregators4)

        
        # layer 4
        grads_and_vars = self.optimizer.compute_gradients(self.loss4)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        # layer 5
        outputs5, aggregators5 = self.aggregate(samples1, [self.node_preds4], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)
        self.node_preds5 = self.node_pred(outputs5)
        self.loss5 = self._loss2(self.node_preds5, aggregators5)

                # layer 5
        grads_and_vars = self.optimizer.compute_gradients(self.loss5)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        # layer 6
        outputs6, self.aggregators = self.aggregate(samples1, [self.node_preds5], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)
        self.node_preds6 = self.node_pred(outputs6)
        self.node_preds = self.node_preds6
        self._loss(only_ppr)

        # applying gradients
        
        # layer 6
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict(only_ppr)        

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

    def _loss2(self, node_preds, aggregators):
        # Weight decay loss
        loss = 0
        for aggregator in aggregators:
            for var in aggregator.vars.values():
                loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        loss += tf.reduce_mean(input_tensor=tf.keras.losses.mae(
            self.placeholders['labels'],
            node_preds))
        return loss

    def predict(self, only_ppr):
        if not only_ppr:
            if self.sigmoid_loss:
                return tf.nn.sigmoid(self.node_preds)
            else:
                return tf.nn.softmax(self.node_preds)
        else:
            return self.node_preds
