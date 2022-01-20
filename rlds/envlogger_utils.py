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

import json
import secrets
from typing import Any, Dict, Optional, Text

import dm_env
from rlds import rlds_types


Metadata = Dict[Text, Any]


class _EpisodeMetadataProvider:
  """Provides some episode metadata that can change along an episode.

  Note that Envlogger calls this function on every step. For each episode,
  it stores the last value returned that is not None.
  """

  def __init__(self,
               agent_spec: Optional[Any] = None,
               env_config: Optional[Any] = None,
               experiment_spec: Optional[Any] = None):
    default_value = 'Unknown'
    self._agent_spec = default_value if agent_spec is None else agent_spec
    self._env_config = default_value if env_config is None else env_config
    self._experiment_spec = experiment_spec
    if self._experiment_spec is None:
      self._experiment_spec = 'Unknown'

    self._current_episode_metadata = None

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
          rlds_types.AGENT_ID: json.dumps(self._agent_spec),
          rlds_types.ENVIRONMENT_CONFIG: json.dumps(self._env_config),
          rlds_types.EXPERIMENT_ID: json.dumps(self._experiment_spec),
          rlds_types.INVALID: True,
      }

    if timestep.last():
      # When the last step is written, the episode is considered as valid.
      self._current_episode_metadata[rlds_types.INVALID] = False

    return self._current_episode_metadata


def get_standard_episode_metadata_fn(agent_spec: Optional[Any] = None,
                                     env_config: Optional[Any] = None,
                                     experiment_spec: Optional[Any] = None):
  """Returns a function that provides some recommanded RLDS episode metadata.

  Args:
    agent_spec: Specification of the agent if any. Can be any JSON serializable
      object.
    env_config: Configuration of the environment that generates the episodes if
      any. Can be any JSON serializable object.
    experiment_spec: Specification of the experiment that generates
      the episodes. Can be any JSON serializable object.

  Returns:
    A function suitable for the EnvLogger wrapper which returns some standard
    episode metadata as defined in rlds_types.
  """
  episode_metadata_provider = _EpisodeMetadataProvider(
      agent_spec, env_config, experiment_spec)

  return episode_metadata_provider.get_episode_metadata
