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


