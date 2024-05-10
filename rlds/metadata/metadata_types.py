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
"""Types/constants used in RL Datasets metadata."""

# Constants representing optional episode metadata fields.

# Unique identifier for the episode. This ID should be unique in the dataset
# and be unique with very high probability across datasets.
# UTF-8 encoded string.
EPISODE_ID = 'episode_id'

# Unique identifier of the agent(s) that generated the episode.
# In a single agent setting, the agent id is a single string (scalar).
# In a multi-agent setting, the agent id is a tensor of size N x 2
# where agent_id[k, 0] is the agent's name in the environment
# and agent_id[k, 1] is the id of the actual agent that generated the episode.
AGENT_ID = 'agent_id'

# UTF-8 encoded string describing the configuration of the environment
# as it was instantiated to generate the episode. This field can for example be
# a JSON string.
# RLDS provides a function to instantiates an environment based on this config
# provided the environment config is a dictionary made of 3 items:
# - "module": refers to the module name to import to be able to instiate the
#             environment.
# - "factory": refers to the function name to invoke to create the environment.
# - "config": is a dictionary where the keys are the keyword arguments
#             of the factory.
ENVIRONMENT_CONFIG = 'environment_config'

# Identifier of the experiment when the episode was generated as part
# of an experiment. The field is a string and could contain a non-readable ID
# or a JSON serialized experiment config.
EXPERIMENT_ID = 'experiment_id'

# This flag indicates whether the episode is invalid (and should in general
# be ignored at read time).
# Since episodes are in general recorded step by step, there are a few scenarios
# where an episode might be incomplete: e.g. machine preemption.
INVALID = 'invalid'
