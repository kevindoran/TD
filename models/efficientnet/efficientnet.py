import tensorflow as tf
import numpy as np
from ..registry import register

from . import efficientnet_builder as model_builder
from . import utils


@register("efficientnet")
def get_efficientnet(hparams, lr):
    """Callable model function compatible with Experiment API.

    Args:
        hparams: a HParams object containing values for fields:
            - (to fill in)
    """
    def efficientnet_model_fn(features, labels, mode, params):
        """The model_fn to be used with TPUEstimator.

        Args:
          features: `Tensor` of batched images.
          labels: `Tensor` of labels for the data samples
          mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
          params: `dict` of parameters passed to the model from the TPUEstimator,
              `params['batch_size']` is always provided and should be used as the
              effective batch size.

        Returns:
          A `TPUEstimatorSpec` for the model
        """
        if isinstance(features, dict):
            features = features['feature']
        if hparams.use_tpu and 'batch_size' in params.keys():
            hparams.batch_size = params['batch_size']
        # In most cases, the default data format NCHW instead of NHWC should be
        # used for a significant performance boost on GPU. NHWC should be used
        # only if the network needs to be run on CPU since the pooling operations
        # are only supported on NHWC. TPU uses XLA compiler to figure out best layout.
        if hparams.data_format == 'channels_first':
            assert not hparams.transpose_input  # channels_first only for GPU
            features = tf.transpose(features, [0, 3, 1, 2])
            stats_shape = [3, 1, 1]
        else:
            stats_shape = [1, 1, 3]

        if hparams.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
            features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        has_moving_average_decay = (hparams.moving_average_decay > 0)
        # This is essential, if using a keras-derived model.
        tf.keras.backend.set_learning_phase(is_training)
        tf.logging.info('Using open-source implementation.')

        def normalize_features(features, mean_rgb, stddev_rgb):
            """Normalize the image given the means and stddevs."""
            features -= tf.constant(mean_rgb, shape=stats_shape,
                                    dtype=features.dtype)
            features /= tf.constant(stddev_rgb, shape=stats_shape,
                                    dtype=features.dtype)
            return features

        def build_model():
            """Build model using the model_name given through the command line."""
            normalized_features = normalize_features(features,
                                                     model_builder.MEAN_RGB,
                                                     model_builder.STDDEV_RGB)
            logits, _ = model_builder.build_model(
                normalized_features,
                model_name=hparams.model_variant,
                training=is_training,
                override_params={},
                model_dir=hparams.output_dir,
                hparams=hparams)
            return logits

        # if params['use_bfloat16']:
        #     with tf.contrib.tpu.bfloat16_scope():
        #         logits = tf.cast(build_model(), tf.float32)
        # else:
        #    logits = build_model()
        logits = build_model()

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                })

        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        one_hot_labels = tf.one_hot(labels, hparams.num_classes)
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits,
            onehot_labels=one_hot_labels,
            label_smoothing=hparams.label_smoothing)

        # Add weight decay to the loss for non-batch-normalization variables.
        loss = cross_entropy + hparams.weight_decay_rate * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
             if 'batch_normalization' not in v.name])

        global_step = tf.train.get_global_step()
        if has_moving_average_decay:
            ema = tf.train.ExponentialMovingAverage(
                decay=hparams.moving_average_decay, num_updates=global_step)
            ema_vars = utils.get_ema_vars()

        host_call = None
        restore_vars_dict = None
        if is_training:
            # Compute the current epoch and associated learning rate from global_step.
            current_epoch = (
                    tf.cast(global_step, tf.float32) *
                    tf.cast(hparams.batch_size, tf.float32) / hparams.epoch_size)
            optimizer = utils.build_optimizer(lr)
            if hparams.use_tpu:
                # When using TPU, wrap the optimizer with CrossShardOptimizer which
                # handles synchronization details between different TPU cores. To the
                # user, this should look like regular synchronous training.
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

            # Batch normalization requires UPDATE_OPS to be added as a dependency to
            # the train operation.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)

            if has_moving_average_decay:
                with tf.control_dependencies([train_op]):
                    train_op = ema.apply(ema_vars)

            if not hparams.skip_host_call:
                def host_call_fn(gs, lr, ce):
                    """Training host call. Creates scalar summaries for training metrics.

                    This function is executed on the CPU and should not directly reference
                    any Tensors in the rest of the `model_fn`. To pass Tensors from the
                    model to the `metric_fn`, provide as part of the `host_call`. See
                    https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
                    for more information.

                    Arguments should match the list of `Tensor` objects passed as the second
                    element in the tuple passed to `host_call`.

                    Args:
                      gs: `Tensor with shape `[batch]` for the global_step
                      lr: `Tensor` with shape `[batch]` for the learning_rate.
                      ce: `Tensor` with shape `[batch]` for the current_epoch.

                    Returns:
                      List of summary ops to run on the CPU host.
                    """
                    gs = gs[0]
                    # Host call fns are executed FLAGS.iterations_per_loop times after one
                    # TPU loop is finished, setting max_queue value to the same as number of
                    # iterations will make the summary writer only flush the data to storage
                    # once per loop.
                    with tf.contrib.summary.create_file_writer(
                            hparams.output_dir,
                            max_queue=hparams.tpu_iterations_per_loop).as_default():
                        with tf.contrib.summary.always_record_summaries():
                            tf.contrib.summary.scalar('learning_rate', lr[0],
                                                      step=gs)
                            tf.contrib.summary.scalar('current_epoch', ce[0],
                                                      step=gs)

                            return tf.contrib.summary.all_summary_ops()

                # To log the loss, current learning rate, and epoch for Tensorboard, the
                # summary op needs to be run on the host CPU via host_call. host_call
                # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
                # dimension. These Tensors are implicitly concatenated to
                # [params['batch_size']].
                gs_t = tf.reshape(global_step, [1])
                lr_t = tf.reshape(lr, [1])
                ce_t = tf.reshape(current_epoch, [1])

                host_call = (host_call_fn, [gs_t, lr_t, ce_t])

        else:
            train_op = None
            if has_moving_average_decay:
                # Load moving average variables for eval.
                restore_vars_dict = ema.variables_to_restore(ema_vars)

        eval_metrics = None
        if mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(labels, logits):
                """Evaluation metric function. Evaluates accuracy.

                This function is executed on the CPU and should not directly reference
                any Tensors in the rest of the `model_fn`. To pass Tensors from the model
                to the `metric_fn`, provide as part of the `eval_metrics`. See
                https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
                for more information.

                Arguments should match the list of `Tensor` objects passed as the second
                element in the tuple passed to `eval_metrics`.

                Args:
                  labels: `Tensor` with shape `[batch]`.
                  logits: `Tensor` with shape `[batch, num_classes]`.

                Returns:
                  A dict of the metrics to return from evaluation.
                """
                predictions = tf.argmax(logits, axis=1)
                top_1_accuracy = tf.metrics.accuracy(labels, predictions)
                in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5),
                                   tf.float32)
                top_5_accuracy = tf.metrics.mean(in_top_5)

                return {
                    'top_1_accuracy': top_1_accuracy,
                    'top_5_accuracy': top_5_accuracy,
                }

            eval_metrics = (metric_fn, [labels, logits])

        num_params = np.sum(
            [np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('number of trainable parameters: {}'.format(num_params))

        def _scaffold_fn():
            saver = tf.train.Saver(restore_vars_dict)
            return tf.train.Scaffold(saver=saver)

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            host_call=host_call,
            eval_metrics=eval_metrics,
            scaffold_fn=_scaffold_fn if has_moving_average_decay else None)

    return efficientnet_model_fn

