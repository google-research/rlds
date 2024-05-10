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
"""Utilities to apply Reverb Patterns to RLDS."""

from typing import Any, Callable, Dict, Optional, Sequence

import reverb
from rlds import rlds_types
import tensorflow as tf


def step_spec(episodes_dataset: tf.data.Dataset) -> Dict[str, Any]:
  """Provides the step spec of an episodes dataset.

  Args:
    episodes_dataset: dataset of episodes where each episode contains a nested
      dataset of steps.

  Returns:
    A dictionary with the element spec of the steps dataset.

  """
  return episodes_dataset.element_spec[rlds_types.STEPS].element_spec


def pattern_map(episodes_dataset: tf.data.Dataset,
                configs: Sequence[reverb.structured_writer.Config],
                respect_episode_boundaries: bool = True) -> tf.data.Dataset:
  """Function to apply a Reverb Pattern to an RLDS dataset.

  Args:
    episodes_dataset: dataset of episodes where each episode contains a nested
      dataset of steps.
    configs: list of reverb patterns to apply to the dataset.
    respect_episode_boundaries: if True, the elements in the output dataset
      never merge elements from different episodes.

  Returns:
    A tf.data.Dataset resulting of applying the pattern to the input dataset.
  """
  return reverb.PatternDataset(
      # We convert the dataset of episodes into a dataset of steps
      input_dataset=episodes_dataset.flat_map(lambda e: e[rlds_types.STEPS]),
      configs=configs,
      # By setting this to true, we don't generate transitions that mix steps
      # from two episodes.
      respect_episode_boundaries=respect_episode_boundaries,
      is_end_of_episode=lambda step: step[rlds_types.IS_LAST],
  )


def pattern_map_from_transform(
    episodes_dataset: tf.data.Dataset,
    transform_fn: Callable[[reverb.structured_writer.ReferenceStep],
                           reverb.structured_writer.Pattern],
    conditions: Optional[Sequence[reverb.structured_writer.Condition]] = None,
    respect_episode_boundaries: bool = True) -> tf.data.Dataset:
  """Function to apply a simple Reverb Pattern to an RLDS dataset.

  This function can be used when the pattern consists of one transformation and
  there are no conditions. Otherwise, use `pattern_map`.
  For example:
  ```
    def _transformation(step):
      return{
         last_three_observations: tf.nest.map_structure(lambda x: x[-3:],
                                                        step[rlds.OBSERVATION])
      }

    trajectories_dataset = pattern_map_from_transform(
                episodes_dataset,
                _transformation,
                respect_episode_boundaries=True,
            )
  ```
  This will produce a trajectories dataset where each trajectory contains three
  consecutive observations. None of the trajectories will contain observations
  from different episodes.


  Args:
    episodes_dataset: dataset of episodes where each episode contains a nested
      dataset of steps.
    transform_fn: function that takes a step and returns a pattern. Note that
      the step is assumed to be batched. See the example.
    conditions: list of conditions to decide whether to apply the pattern or
      not.
    respect_episode_boundaries: if True, the elements in the output dataset
      never merge elements from different episodes.

  Returns:
    A tf.data.Dataset resulting of applying the pattern to the input dataset.
  """
  if not conditions:
    conditions = []
  spec = step_spec(episodes_dataset)
  pattern = reverb.structured_writer.pattern_from_transform(spec, transform_fn)
  # The table is unused
  config = reverb.structured_writer.create_config(
      pattern, table='transition', conditions=conditions)

  return pattern_map(
      episodes_dataset=episodes_dataset,
      configs=[config],
      respect_episode_boundaries=respect_episode_boundaries,
  )
