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

import enum
import importlib
import json
import secrets
from typing import Any, Dict, Optional, Text

import dm_env
from rlds import rlds_types



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
    default_value = 'Unknown'
    self._agent_metadata = (default_value if agent_metadata is None
                            else agent_metadata)
    self._env_config = default_value if env_config is None else env_config
    self._experiment_metadata = experiment_metadata
    if self._experiment_metadata is None:
      self._experiment_metadata = 'Unknown'

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
    self._agent_metadata = agent_metadata

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
    env_config: A config of the environment.

  Returns:
    A new environment. Note that there is no constraint on the type of
    supported environments, it can be a dm_env.Environment or a GYM environment.
  """
  module = importlib.import_module(env_config['module'])
  factory = getattr(module, env_config['factory'])
  return factory(**env_config['config'])
