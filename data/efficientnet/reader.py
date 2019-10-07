import os

from ..registry import register
from .imagenet_input import ImageNetInput


@register("efficientnet_imagenet", None)
def image_reader(tf_record_dir, hparams, training):
    num_vcpu = os.cpu_count()
    num_parallel_calls = num_vcpu
    use_tpu_transpose_trick = hparams.transpose_input
    use_bfloat16 = False
    use_autoaugmentation = False
    autoaugment_name = 'v0' if use_autoaugmentation else None
    imagenet_input = ImageNetInput(
        is_training=training,
        data_dir= tf_record_dir,
        # What is this option for? It puts the batch dim last.
        # transpose_input=use_tpu_transpose_trick,
        transpose_input=use_tpu_transpose_trick,
        cache=training,
        image_size=hparams.input_shape[0],
        num_parallel_calls=num_parallel_calls,
        use_bfloat16=use_bfloat16,
        # Whether to use 1001 classes and ignore class index 0, or use 1000.
        include_background_label=False,
        autoaugment_name=autoaugment_name)
    return imagenet_input.input_fn
