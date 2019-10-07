import getpass
import os
import subprocess

import tensorflow as tf

from .envs import get_env


def validate_flags(FLAGS):
  messages = []
  if not FLAGS.env:
    messages.append("Missing required flag --env")
  if not FLAGS.hparams:
    messages.append("Missing required flag --hparams")

  if len(messages) > 0:
    raise Exception("\n".join(messages))

  return FLAGS


def update_hparams(FLAGS, hparams, hparams_name):
  hparams.env = FLAGS.env
  hparams.use_tpu = hparams.env == "tpu"
  hparams.train_epochs = FLAGS.train_epochs or hparams.train_epochs
  hparams.eval_steps = FLAGS.eval_steps or hparams.eval_steps
  hparams.tpu_iterations_per_loop = FLAGS.tpu_iterations_per_loop

  env = get_env(FLAGS.env)
  hparams.data_dir = FLAGS.data_dir or env.data_dir
  hparams.output_dir = FLAGS.output_dir or env.output_dir

  return hparams
