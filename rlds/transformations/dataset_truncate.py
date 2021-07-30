# Copyright 2021 Google LLC.
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
"""Truncation tools."""

from typing import Callable

from rlds import rlds_types
from rlds.transformations import shape_ops
import tensorflow as tf


def truncate_after_condition(
    steps_dataset: tf.data.Dataset,
    truncate_condition: Callable[[rlds_types.Step], bool]) -> tf.data.Dataset:
  """Truncates a dataset after the first step that satisfies a condition.

  Args:
    steps_dataset: dataset of steps.
    truncate_condition: function that takes one step and evaluates the
      truncation condition.

  Returns:
    Dataset truncated after the first step that satisfies the
    `truncate_condition`.

  """
  # TODO(sabela): Provide a version optimized with batching.
  # First, we convert the dataset of steps to a dataset of tuples containing
  # (step, condition_in_prev_step). This is needed to make sure we include the
  # step for which the condition is True.
  scan_fn = tf.data.experimental.scan(
     False, lambda prev_condition, step:
     (truncate_condition(step), (step, prev_condition)))
  data_with_prev = steps_dataset.apply(scan_fn)
  take_fn = tf.data.experimental.take_while(
     lambda step, stop_condition: not stop_condition)
  data_with_prev = data_with_prev.apply(take_fn)

  return data_with_prev.map(lambda step, tag: step)


# TODO(damienv, sabela): avoid exposing this through the API.
def next_is_terminal(steps_dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Transforms a dataset into a dataset of tuples (step, next_is_terminal).

  Args:
    steps_dataset: dataset of steps. Steps are dictionaries and expected to
      contain an `IS_TERMINAL` key with a boolean value.

  Returns:
    A dataset with tuples of (step, bool) where the bool indicates if the next
    step is terminal.
  """
  # TODO(sabela): Provide a version optimized with batching.
  step0 = tf.data.experimental.get_single_element(steps_dataset.take(1))
  next_steps = steps_dataset.skip(1).concatenate(
      shape_ops.zero_dataset_like(steps_dataset))

  scan_fn = tf.data.experimental.scan(
    step0, lambda step, next_step:
     (next_step, (step, next_step[rlds_types.IS_TERMINAL])))
  return next_steps.apply(scan_fn)

