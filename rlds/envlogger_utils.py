# Copyright 2022 Google LLC.
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
"""EnvLogger utility to add standard episode metadata."""

import contextlib
import enum
import importlib
import json
import secrets
from typing import Any, Dict, Optional, Text

import dm_env
import gin
from rlds import rlds_types
import tensorflow as tf


Metadata = Dict[Text, Any]


@enum.unique
class MetadataSerialization(enum.Enum):
  """Supported metadata serializations."""
  NONE = enum.auto()
  JSON = enum.auto()


class EpisodeMetadataProvider:
  """Provides some episode metadata that can change along an episode.

  Note that Envlogger calls this function on every step. For each episode,
  it stores the last value returned that is not None.
  """

  def __init__(
      self,
      agent_metadata: Optional[Any] = None,
      env_config: Optional[Any] = None,
      experiment_metadata: Optional[Any] = None,
      serialization: MetadataSerialization = MetadataSerialization.JSON):
    default_value = 'None'
    self._agent_metadata = (default_value if agent_metadata is None
                            else agent_metadata)
    self._env_config = default_value if env_config is None else env_config
    self._experiment_metadata = experiment_metadata
    if self._experiment_metadata is None:
      self._experiment_metadata = default_value

    self._current_episode_metadata = None
    self._serialization = serialization

  def _serialize(self, data):
    if self._serialization == MetadataSerialization.JSON:
      return json.dumps(data)
    if self._serialization == MetadataSerialization.NONE:
      # Rely on the underlying envlogger serialization.
      return data
    raise ValueError('Unsupported serialization scheme')

  def set_agent_metadata(self, agent_metadata: Any):
    """Sets the agent specification."""
    if self._serialization == MetadataSerialization.NONE:
      old_spec = tf.nest.map_structure(tf.type_spec_from_value,
                                       self._agent_metadata)
      new_spec = tf.nest.map_structure(tf.type_spec_from_value, agent_metadata)
      if old_spec != new_spec:
        raise ValueError(f'New agent metadata has spec {new_spec} that is'
                         f' incompatible with the current spec {old_spec}')
    self._agent_metadata = agent_metadata

  def set_env_config(self, env_config: Any):
    """Sets the environment configuration."""
    if self._serialization == MetadataSerialization.NONE:
      old_spec = tf.nest.map_structure(tf.type_spec_from_value,
                                       self._env_config)
      new_spec = tf.nest.map_structure(tf.type_spec_from_value, env_config)
      if old_spec != new_spec:
        raise ValueError(f'New env config has spec {new_spec} that is'
                         f' incompatible with the current spec {old_spec}')
    self._env_config = env_config

  def get_episode_metadata(self,
                           timestep: dm_env.TimeStep,
                           action: Any,
                           env: dm_env.Environment) -> Optional[Metadata]:
    """Returns some standard episode metadata."""
    del action
    del env
    if timestep.first():
      self._current_episode_metadata = {
          rlds_types.EPISODE_ID: secrets.token_bytes(16),
          rlds_types.AGENT_ID: self._serialize(self._agent_metadata),
          rlds_types.ENVIRONMENT_CONFIG: self._serialize(self._env_config),
          rlds_types.EXPERIMENT_ID: self._serialize(self._experiment_metadata),
          rlds_types.INVALID: True,
      }

    if timestep.last():
      # When the last step is written, the episode is considered as valid.
      self._current_episode_metadata[rlds_types.INVALID] = False

    return self._current_episode_metadata

  def _serialized_spec(self, metadata):
    if self._serialization == MetadataSerialization.JSON:
      return tf.TensorSpec(shape=(), dtype=tf.string)
    else:
      return tf.nest.map_structure(tf.type_spec_from_value, metadata)

  def get_episode_metadata_spec(self):
    """Returns the spec of the metadata as tf.TensorSpec."""
    return {
        rlds_types.EPISODE_ID:
            tf.TensorSpec(shape=(), dtype=tf.string),
        rlds_types.AGENT_ID:
            self._serialized_spec(self._agent_metadata),
        rlds_types.ENVIRONMENT_CONFIG:
            self._serialized_spec(self._env_config),
        rlds_types.EXPERIMENT_ID:
            self._serialized_spec(self._experiment_metadata),
        rlds_types.INVALID:
            tf.TensorSpec(shape=(), dtype=tf.bool),
    }


def get_standard_episode_metadata_fn(
    agent_metadata: Optional[Any] = None,
    env_config: Optional[Any] = None,
    experiment_metadata: Optional[Any] = None,
    serialization: MetadataSerialization = MetadataSerialization.JSON):
  """Returns a function that provides some recommanded RLDS episode metadata.

  This function assumes the metadata given as input remains constant across
  episodes. If it changes, EpisodeMetadataProvider should be used instead.

  Args:
    agent_metadata: Specification of the agent if any. Can be any JSON
      serializable object.
    env_config: Configuration of the environment that generates the episodes if
      any. Can be any JSON serializable object.
    experiment_metadata: Specification of the experiment that generates
      the episodes. Can be any JSON serializable object.
    serialization: Serialization scheme for complex metadata structures.
      When serialization is NONE, no serialization is applied at this level.
      Note that when no serialization is applied, the metadata structures must
      remain the same along the lifetime of the EnvLogger since datasets must
      have fixed specs. When other serialization schemes are used, the dataset
      spec of complex metadata is a string regardless of the content and the
      underlying structures may change between the different episodes.

  Returns:
    A function suitable for the EnvLogger wrapper which returns some standard
    episode metadata as defined in rlds_types.
  """
  episode_metadata_provider = EpisodeMetadataProvider(
      agent_metadata, env_config, experiment_metadata, serialization)

  return episode_metadata_provider.get_episode_metadata


def make_env(env_config: Dict[str, Any]) -> Any:
  """Makes a new environment given an RLDS environment config.

  Args:
    env_config: A config of the environment. This is a dictionary with 'module',
      'factory' and 'config' as mandatory keys and 'gin_scope' and 'gin_config'
      as optional keys. The 'module' and 'factory' defines which function to
      execute to create the environment while the 'config' is a list of named
      parameters that are given to this function. Optionally, one can pass
      a gin scope and a gin config to pass extra parameters using GIN.

  Returns:
    A new environment. Note that there is no constraint on the type of
    supported environments, it can be a dm_env.Environment or a GYM environment.
  """
  module = importlib.import_module(env_config['module'])
  factory = getattr(module, env_config['factory'])
  if 'gin_config' in env_config:
    gin.parse_config(env_config['gin_config'])
  if 'gin_scope' in env_config:
    context = gin.config_scope(env_config['gin_scope'])
  else:
    context = contextlib.nullcontext()
  with context:
    env = factory(**env_config['config'])
  return env
