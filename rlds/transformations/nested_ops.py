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
"""Module with utils to manipulate nested datasets."""

from typing import Any, Callable, Dict

from rlds import rlds_types
from rlds.transformations import flexible_batch
from rlds.transformations import shape_ops
import tensorflow as tf


def _get_episode_metadata(episode: rlds_types.Episode) -> Dict[str, Any]:
  return {k: episode[k] for k in episode if k != rlds_types.STEPS}


def _map_episode(
    episode: rlds_types.Episode,
    transform_step: Callable[[rlds_types.Step], Any],
    in_place: bool,
    optimization_batch_size: int = 1) -> rlds_types.Episode:
  """Applies a transformation to all the steps of an episode.

  Args:
    episode: single episode.
    transform_step: Function that takes one step and applies a transformation.
      The return type is not necessarily a step.
    in_place: Whether the operation is done in place on the original episode.
    optimization_batch_size: when >1, how many steps to process in a single
      batch.

  Returns:
    An episodes where all the steps are transformed according to the
    transformation function and stored in the given episode_key.
  """
  def vectorized_transform(steps):
    return tf.vectorized_map(transform_step, steps)

  if optimization_batch_size > 1:
    steps = episode[rlds_types.STEPS].batch(optimization_batch_size).map(
        vectorized_transform).unbatch()
  else:
    steps = episode[rlds_types.STEPS].map(transform_step)
  if in_place:
    episode[rlds_types.STEPS] = steps
    return episode
  else:
    return rlds_types.build_episode(steps=steps,
                                    metadata=_get_episode_metadata(episode))


def map_nested_steps(
    episodes_dataset: tf.data.Dataset,
    transform_step: Callable[[rlds_types.Step], Any],
    optimization_batch_size: int = flexible_batch.BATCH_AUTO_TUNE
) -> tf.data.Dataset:
  """Applies a transformation to all the steps of a dataset.

  Args:
    episodes_dataset: Dataset of episodes.
    transform_step: Function that takes one step and applies a transformation.
      The return type is not necessarily a step.
    optimization_batch_size: how many steps to process in a single batch.
      The default value (BATCH_AUTO_TUNE) makes batch size selection automatic.

  Returns:
    A dataset of episodes where all the steps are transformed according to the
    transformation function.
  """
  optimization_batch_size = flexible_batch.get_batch_size(
      episodes_dataset, optimization_batch_size)
  # Note that doing the operation in place does not modify the input dataset.

  return episodes_dataset.map(lambda e: _map_episode(
      e,
      transform_step,
      in_place=True,
      optimization_batch_size=optimization_batch_size))



def map_steps(
    steps_dataset: tf.data.Dataset,
    transform_step: Callable[[rlds_types.Step], Any],
    optimization_batch_size: int = flexible_batch.BATCH_AUTO_TUNE
) -> tf.data.Dataset:
  """Applies a transformation to all the elements of a dataset.

  Args:
    steps_dataset: Dataset of steps.
    transform_step: Function that takes one step and applies a transformation.
      The return type is not necessarily a step.
    optimization_batch_size: how many steps to process in a single batch.
      The default value (BATCH_AUTO_TUNE) makes batch size selection automatic.

  Returns:
    A dataset where all the steps are transformed according to the
    transformation function.
  """
  optimization_batch_size = flexible_batch.get_batch_size(
      steps_dataset, optimization_batch_size)

  def vectorized_transform(steps):
    return tf.vectorized_map(transform_step, steps)

  if optimization_batch_size > 1:
    steps = steps_dataset.batch(optimization_batch_size).map(
        vectorized_transform).unbatch()
  else:
    steps = steps_dataset.map(transform_step)
  return steps


def _apply_episode(
    episode: rlds_types.Episode,
    transform_step_dataset: Callable[[tf.data.Dataset], Any],
    in_place: bool) -> rlds_types.Episode:
  """Applies a transformation to the steps dataset of an episode.


  Args:
    episode: single episode.
    transform_step_dataset: Function that takes a dataset of steps and applies
      a transformation.
    in_place: Whether the operation is done in place on the original episode.

  Returns:
    An episode where the dataset of steps is transformed according to the
    transformation function.
  """
  steps = episode[rlds_types.STEPS].apply(transform_step_dataset)
  if in_place:
    episode[rlds_types.STEPS] = steps
    return episode
  else:
    return rlds_types.build_episode(steps=steps,
                                    metadata=_get_episode_metadata(episode))


def apply_nested_steps(
    episodes_dataset: tf.data.Dataset,
    transform_step_dataset: Callable[[tf.data.Dataset], Any]
    ) -> tf.data.Dataset:
  """Applies for each episode a transformation on the dataset of steps.

  Args:
    episodes_dataset: Dataset of episodes.
    transform_step_dataset: A function that takes a dataset of steps and applies
      a transformation.

  Returns:
    A dataset of episodes where all the nested dataset of steps are transformed
    according to the transformation function.
  """
  return episodes_dataset.map(
      lambda e: _apply_episode(e, transform_step_dataset, in_place=True))


def sum_dataset(
    steps_dataset: tf.data.Dataset,
    data_to_sum: Callable[[rlds_types.Step], Any],
    optimization_batch_size: int = flexible_batch.BATCH_AUTO_TUNE) -> Any:
  """Accumulates the values of all steps in a dataset.

  It expects all fields in the steps to have dtypes that support `tf.add`.

  Args:
    steps_dataset: dataset of steps. Each of the steps is a dictionary
    data_to_sum: function applied to each step and returns a structure values
      of which are to be accumulated over all steps. Each returned element
      has to have dtypes that support `tf.add`.
    optimization_batch_size: how many steps to process in a single batch.
      The default value (BATCH_AUTO_TUNE) makes batch size selection automatic.

  Returns:
    Aggregation of all the values returned by the `data_to_sum` for each step.
  """
  def sum_batched_steps(values):
    return tf.reduce_sum(values, 0)

  @tf.function
  def map_and_sum_batched_steps(steps):
    values = tf.vectorized_map(data_to_sum, steps)
    return tf.nest.map_structure(sum_batched_steps, values)
  optimization_batch_size = flexible_batch.get_batch_size(
      steps_dataset, optimization_batch_size)

  if optimization_batch_size > 1:
    steps_to_sum = steps_dataset.batch(optimization_batch_size).map(
        map_and_sum_batched_steps, num_parallel_calls=tf.data.AUTOTUNE)
  else:
    steps_to_sum = steps_dataset.map(
        data_to_sum, num_parallel_calls=tf.data.AUTOTUNE)
  zero_field = shape_ops.zeros_from_spec(steps_to_sum.element_spec)

  return steps_to_sum.reduce(
      zero_field, lambda initial_value, step: tf.nest.map_structure(
          tf.add, initial_value, step))



def sum_nested_steps(
    episodes_dataset: tf.data.Dataset,
    data_to_sum: Callable[[rlds_types.Step], Any],
    optimization_batch_size: int = flexible_batch.BATCH_AUTO_TUNE) -> Any:
  """Accumulates the values of all steps in a dataset of episodes.

  Args:
    episodes_dataset: dataset of episodes.
    data_to_sum: function applied to each step and returns a structure values
      of which are to be accumulated over all steps. Each returned element
      has to have dtypes that support `tf.add`.
    optimization_batch_size: how many steps to process in a single batch.
      The default value (BATCH_AUTO_TUNE) makes batch size selection automatic.

  Returns:
    Aggregation of all the values returned by the `data_to_sum` for each step.
  """
  steps = episodes_dataset.flat_map(lambda x: x[rlds_types.STEPS])
  return sum_dataset(steps, data_to_sum, optimization_batch_size)


@tf.function
def final_step(steps_dataset: tf.data.Dataset) -> Any:
  """Returns the final step of a dataset of steps.

  The dataset of steps must have at least one element.

  Args:
    steps_dataset: dataset of steps. Each of the steps is a dictionary

  Returns:
    The last step of the input dataset of steps.
  """
  cardinality = steps_dataset.cardinality()
  if cardinality != tf.data.experimental.UNKNOWN_CARDINALITY:
    return next(iter(steps_dataset.skip(steps_dataset.cardinality() - 1)))
  zero_step = shape_ops.zeros_from_spec(steps_dataset.element_spec)
  return steps_dataset.reduce(zero_step, lambda state, step: step)


def episode_length(
    steps_dataset: tf.data.Dataset,
    optimization_batch_size: int = flexible_batch.BATCH_AUTO_TUNE) -> int:
  """Obtains the episode length.

  Args:
    steps_dataset: dataset of steps.
    optimization_batch_size: how many steps to process in a single batch.
      The default value (BATCH_AUTO_TUNE) makes batch size selection automatic.

  Returns:
    Number of steps in an episode.
  """
  cardinality = steps_dataset.cardinality()
  if cardinality != tf.data.experimental.UNKNOWN_CARDINALITY:
    return tf.cast(cardinality, tf.int32)

  if optimization_batch_size == flexible_batch.BATCH_AUTO_TUNE:
    size = shape_ops.size_from_spec(steps_dataset.element_spec)
    optimization_batch_size = flexible_batch.DEFAULT_BATCH_SIZE_IN_BYTES // size

  if optimization_batch_size <= 1:
    return steps_dataset.reduce(0, lambda count, step: count + 1)

  return steps_dataset.batch(optimization_batch_size).reduce(
      0, lambda x, step: x + tf.shape(next(iter(tf.nest.flatten(step))))[0])
