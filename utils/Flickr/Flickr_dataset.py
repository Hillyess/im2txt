#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from dataset import DataSet
import time
import pandas as pd
import tensorflow as tf
import os
import numpy as np
import tqdm
from utils.Flickr import inputs as input_ops

# TFRcord文件
TRAIN_FILE = 'fcn_train.record'
VALIDATION_FILE = 'fcn_val.record'

# 图片信息
NUM_CLASSES = 21

def build_inputs(self, config):
    """Input prefetching, preprocessing and batching.
    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask