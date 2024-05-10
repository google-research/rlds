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
"""Types used in RL Datasets."""
from typing import Any, Callable, Dict, List, Optional, Union

import tensorflow as tf

# Constants representing Step fields
OBSERVATION = 'observation'
ACTION = 'action'
REWARD = 'reward'
IS_TERMINAL = 'is_terminal'
IS_FIRST = 'is_first'
IS_LAST = 'is_last'
DISCOUNT = 'discount'

CORE_STEP_FIELDS = frozenset(
    [OBSERVATION, ACTION, REWARD, IS_TERMINAL, IS_FIRST, IS_LAST, DISCOUNT])

# Constants representing Episode fields
STEPS = 'steps'

# Constants representing optional Step fields
ALIGNMENT = 'alignment'

# Types of RLDS data
Episode = Dict[str, Any]
Step = Dict[str, Any]
BatchedStep = Step
BatchedEpisode = Episode

# Step(s) transformation function types.
StepFilterFn = Callable[[Step], bool]
EpisodeFilterFn = Callable[[Episode], bool]
StepsToStepsFn = Callable[[tf.data.Dataset], tf.data.Dataset]
StepMapFn = Callable[[Step], Step]


def build_step(
    observation: Optional[Any],
    action: Optional[Any],
    reward: Optional[Any],
    discount: Optional[Any],
    # Union[List[X], X] is used to allow batching in the dataset.
    is_terminal: Optional[Union[List[bool], bool]],
    is_first: Union[List[bool], bool],
    is_last: Union[List[bool], bool],
    metadata: Optional[Dict[str, Any]] = None) -> Step:
  """Returns a dictionary representing an environment step.

  Each step encodes the current observation, the action applied to this
  observation, and the reward obtained by applying the action. If the step is
  terminal (i.e., is_terminal=true), the reward, action and discount are
  meaningless as this observation is a terminal state.

  If an episode ends in a step where `is_terminal = False`, it means that this
  episode has been truncated. In this case, depending on the environment, the
  action, reward and discount might be empty as well.


  Args:
    observation: tensor representing the current observation.
    action: tensor representing the action applied after `observation` was made.
    reward: tensor representing the reward obtained as a result of applying
      `action`.
    discount: tensor representing the discount factor (at the same step as the
      `reward`).
    is_terminal: true iff the observation is terminal. Note that `reward` and
      `action` are meaningless if `is_terminal=true`.
    is_first: true iff the observation is the first observation in the episode.
    is_last: true iff the observation is the last observation in the episode.
      When true, `action`, `reward` and `discount` are considered invalid.
    metadata: dictionary with step metadata.

  Returns:
    A dictionary representing the RL Step.

  Raises:
    ValueError if any key in the metadata corresponds to any of the step keys.
  """
  if metadata is None:
    metadata = {}
  step = {
      IS_FIRST: is_first,
      IS_LAST: is_last,
  }
  if observation is not None:
    step[OBSERVATION] = observation
  if action is not None:
    step[ACTION] = action
  if reward is not None:
    step[REWARD] = reward
  if discount is not None:
    step[DISCOUNT] = discount
  if is_terminal is not None:
    step[IS_TERMINAL] = is_terminal
  for k, v in metadata.items():
    if k in step:
      raise ValueError(f'Invalid Step Metadata: it contains step key {k}')
    else:
      step[k] = v
  return step


def build_episode(
    # Union[List[X], X] is used to allow batching an Episode dataset.
    steps: Union[List[Any], tf.data.Dataset],
    metadata: Dict[str, Any]) -> Episode:
  """Represents an RL Episode.

  Args:
    steps: dataset of dictionaries representing one timestep.
    metadata: dictionary with the episode metadata.

  Returns:
    A dictionary representing the RL Episode.
  """
  if STEPS in metadata.keys():
    raise ValueError(
        f'Invalid Episode Metadata: it contains an episode key ({STEPS})')
  return {
      STEPS: steps,
      **metadata,
  }


def _is_valid_steps_dataset(dataset: tf.data.Dataset) -> bool:
  """Basic validation of a steps dataset with all the fields defined.

  Args:
    dataset: steps dataset.

  Returns:
    true if each step contains the required step keys, false otherwise.

  """
  step = next(iter(dataset))
  return CORE_STEP_FIELDS.issubset(step.keys())


def is_valid_rlds_dataset(dataset: tf.data.Dataset) -> bool:
  """Basic validation of an episodes dataset.

  Args:
    dataset: episodes dataset. All the basic fields in the dataset have to be
      defined.

  Returns:
    true if each episode contains the required episode keys and a valid steps
    dataset, false otherwise.

  """
  episode = next(iter(dataset))
  return STEPS in episode and _is_valid_steps_dataset(episode[STEPS])
