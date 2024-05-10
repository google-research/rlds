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
"""Library functions for concatenation of two datasets."""

from typing import Callable, Mapping

from rlds import rlds_types
from rlds.transformations import nested_ops
from rlds.transformations import shape_ops
import tensorflow as tf


def _add_empty_values(step: rlds_types.Step,
                      spec: Mapping[str, tf.TensorSpec]) -> rlds_types.Step:
  """Adds zero elements to step for the keys in spec not present in step.

  Args:
    step: Dictionary representing a Step.
    spec: Tensor spec.

  Returns:
    Dictionary containing all the k:v paris from step, and k:zeros for those
    elements in the spec that are not yet part of step.
  """
  for k in spec:
    if k not in step:
      step[k] = shape_ops.zeros_from_spec(spec[k])
  return step


def concatenate(steps1: tf.data.Dataset,
                steps2: tf.data.Dataset) -> tf.data.Dataset:
  """Concatenates the two datasets.

  If one of the datasets contains fields that are not present in the other
  dataset, those fields are added to the other dataset initialized to zeros.

  It assumest that the elements in the datasets are dictionaries.

  Args:
    steps1: First dataset.
    steps2: Second dataset.

  Returns:
    Dataset of steps1 and steps2.
  """


  spec_step1 = steps1.element_spec
  spec_step2 = steps2.element_spec
  steps1 = steps1.map(lambda step: _add_empty_values(step, spec_step2))
  steps2 = steps2.map(lambda step: _add_empty_values(step, spec_step1))
  return steps1.concatenate(steps2)


@tf.function
def concat_if_terminal(
    steps: tf.data.Dataset,
    make_extra_steps: Callable[[rlds_types.Step], tf.data.Dataset]
) -> tf.data.Dataset:
  """Concats the datasets if the steps end in terminal and applies a map.

  Provides the skeleton to add absorbing states to an episode.

  Args:
    steps: dataset of steps. Each step is expected to contain `IS_TERMINAL`.
    make_extra_steps: dataset of step(s) built based on the last step
      and that will be added at the end of the dataset of steps if it ends in
      `IS_TERMINAL = True`.

  Returns:
    A dataset with the extra steps only if the original dataset ends in a
    terminal state and the original steps are transformed by `map_step_fn`.
  """

  final_step = nested_ops.final_step(steps)
  ends_in_terminal = final_step[rlds_types.IS_TERMINAL]
  if ends_in_terminal:
    extra_steps_ds = make_extra_steps(final_step)
    steps = concatenate(steps, extra_steps_ds)
  else:
    # We concatenate with the empty dataset because otherwise the type of the
    # dataset is different in the if/else branches.
    steps = concatenate(steps, steps.take(0))
  return steps
