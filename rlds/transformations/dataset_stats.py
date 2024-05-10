# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
"""Utils to normalize a field in an RLDS dataset."""

from typing import Any, Callable, Dict, Tuple

import numpy as np
from rlds import rlds_types
from rlds.transformations import flexible_batch
from rlds.transformations import shape_ops
import tensorflow as tf


def sar_fields_mask(
    step: rlds_types.Step) -> Tuple[Dict[str, Any], Dict[str, bool]]:
  """Obtains the core fields assuming SAR alignment.

  This can be used as a parameter to `mean_and_std` when the steps user SAR
  alignment (the RLDS default) and observation and action are formed by fields
  whose mean and std can be obtained (for example, they don't contain serialized
  images).

  Args:
    step: RLDS step.

  Returns:
    Tuple with the data (observation, action, reward) and a mask indicating
    if the fields are valid.
  """

  data = {
      k: step[k]
      for k in [rlds_types.OBSERVATION, rlds_types.ACTION, rlds_types.REWARD]
  }
  # In SAR alignment, the action and the reward are invalid in the last step.
  mask = {
      rlds_types.OBSERVATION: True,
      rlds_types.ACTION: not step[rlds_types.IS_LAST],
      rlds_types.REWARD: not step[rlds_types.IS_LAST],
  }
  return (data, mask)


def mean_and_std(
    episodes_dataset: tf.data.Dataset,
    get_step_fields: Callable[[rlds_types.Step], Tuple[Dict[str, Any],
                                                       Dict[str, bool]]],
    optimization_batch_size: int = flexible_batch.BATCH_AUTO_TUNE
) -> Tuple[Any, Any]:
  """Calculates the mean and std of a set of fields accross the dataset.

  Args:
    episodes_dataset: dataset of episodes.
    get_step_fields: function applied to each step and returns a dictionary with
      the data for which we will calculate the stats, as well as a set with the
      keys of the fields that have valid data in this step. These fields are
      expected to have a numeric type.
    optimization_batch_size: how many steps to process in a single batch.
      The default value (BATCH_AUTO_TUNE) makes batch selection automatic.

  Returns:
    Tuple with the mean and std as float64. It maintains the same shape and
    dtype of the output of `get_step_fields`.
  """
  optimization_batch_size = flexible_batch.get_batch_size(
      episodes_dataset, optimization_batch_size)

  @tf.function
  def _get_fields_for_sum_and_count(step):
    fields, mask = get_step_fields(step)
    out_fields = dict()
    for k in fields:
      if mask[k]:
        out_fields[k] = fields[k]
      else:
        out_fields[k] = tf.nest.map_structure(tf.zeros_like, fields[k])
    return out_fields, mask

  def steps_sum(values):
    return tf.reduce_sum(values, 0)

  def map_steps_for_sum_and_count(steps):
    values, mask = tf.vectorized_map(_get_fields_for_sum_and_count, steps)
    return {
        'count': tf.nest.map_structure(tf.math.count_nonzero, mask),
        'data': tf.nest.map_structure(steps_sum, values)
    }

  def steps_std(values, mean):
    return tf.reduce_sum(
        tf.math.subtract(
            tf.cast(values, tf.float64), mean)**2, 0)

  def map_steps_for_std(steps, mean):
    @tf.function
    def _get_fields_for_std(step):
      fields, mask = get_step_fields(step)
      out_fields = dict()
      for k in fields:
        if mask[k]:
          out_fields[k] = tf.nest.map_structure(
              lambda x: tf.cast(x, tf.float64), fields[k])
        else:
          out_fields[k] = mean[k]
      return out_fields

    values = tf.vectorized_map(_get_fields_for_std, steps)
    return tf.nest.map_structure(steps_std, values, mean)

  batched_data = episodes_dataset.flat_map(lambda x: x[rlds_types.STEPS]).batch(
      optimization_batch_size)

  data_for_mean = batched_data.map(
      map_steps_for_sum_and_count, num_parallel_calls=tf.data.AUTOTUNE)
  zero_field = shape_ops.zeros_from_spec(data_for_mean.element_spec)
  result = data_for_mean.reduce(
      zero_field,
      lambda accum, step: tf.nest.map_structure(tf.add, accum, step))

  result = tf.nest.map_structure(lambda x: tf.cast(x, tf.float64), result)
  mean = dict()
  for k in result['count']:
    count = result['count'][k]

    mean[k] = tf.nest.map_structure(lambda x: x/count, result['data'][k])

  data_for_std = batched_data.map(
      lambda steps: map_steps_for_std(steps, mean),
      num_parallel_calls=tf.data.AUTOTUNE)
  zero_std = tf.nest.map_structure(lambda x: tf.cast(x, tf.float64),
                                   zero_field['data'])
  result_for_std = data_for_std.reduce(
      zero_std, lambda accum, step: tf.nest.map_structure(tf.add, accum, step))

  std = dict()
  for k in result['count']:
    std[k] = tf.nest.map_structure(
        lambda x: np.sqrt(x / (result['count'][k] - 1)), result_for_std[k])
  return mean, std
