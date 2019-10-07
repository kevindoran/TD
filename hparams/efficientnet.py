import tensorflow as tf

from .registry import register
from .defaults import *

NUM_IMAGENET_TEST_IMAGES = 50000

@register
def efficientnet_imagenet224():
    # From efficientnet.utils.build_learning_rate() we have:
    # decay_factor = 0.97
    # decay_epochs = 2.4
    # warmup_epochs = 5
    # lr_decay_type = 'exponential'
    # steps_per_epoch

    # Efficientnet's approach:
    # lr = tf.train.exponential_decay(
    #    initial_lr, (scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0))
    #    global_step,
    #    decay_steps, (steps_per_epoch * decay_epochs)
    #    decay_factor,  (0.97)
    #    staircase=True)

    hps = default_imagenet224()
    hps.model = "efficientnet"
    hps.model_variant = "efficientnet-b0"
    hps.data = "efficientnet_imagenet"
    # Optimizer
    hps.optimizer = 'sgd'
    # Loss
    hps.weight_decay_rate = 1e-5
    # Learning rate
    hps.lr_scheme = "warmup_exponential_decay"
    hps.batch_size = 128 * 8
    hps.eval_steps = int(NUM_IMAGENET_TEST_IMAGES / hps.batch_size)
    hps.learning_rate_decay_rate = 0.97
    hps.staircased = True # What does this mean?
    assert hps.epoch_size
    steps_per_epoch = int(hps.epoch_size / hps.batch_size)
    train_batch_size = hps.batch_size
    decay_epochs = 2.4
    warmup_epochs = 5
    hps.learning_rate_decay_interval = steps_per_epoch * decay_epochs
    hps.learning_rate = 0.016 * train_batch_size / 256.0
    hps.warmup_steps = int(warmup_epochs * steps_per_epoch)
    # Unused:
    # hps.delay = None
    # Override?
    # initializer = some sort of variance scaling. # "glorot_normal_initializer",
    hps.num_classes = 1000 # not 1001
    hps.output_shape = [1000] # not [1001]

    # Same as defaults, but override to insure a match with efficientnet
    # implementation:
    hps.label_smoothing = 0.1
    # hps.batch_norm_momentum =
    # hps.batch_norm_epsilon =

    # Extra:
    hps.transpose_input = False
    hps.moving_average_decay = 0 # default: 0.9999
    hps.skip_host_call = False
    return hps

@register
def efficientnet_imagenet224_trgtd_weight_ramping_80_80():
  hps = efficientnet_imagenet224()
  hps.drop_rate = 0.80
  hps.dropout_type = "targeted_weight_piecewise"
  hps.targ_rate = 0.80
  hps.linear_drop_rate = True
  return hps