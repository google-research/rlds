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
"""Unit tests of the EnvLogger util functions."""

import json

from absl.testing import absltest
import dm_env
from rlds import envlogger_utils
from rlds import rlds_types


class EnvLoggerUtilsTest(absltest.TestCase):

  def setUp(self):
    super(EnvLoggerUtilsTest, self).setUp()

  def get_agent_spec(self):
    return {'type': 'synthetic',
            'checkpoint': '/tmp/agent/synthetic.ckpt',
            }

  def get_env_config(self):
    return {'name': 'my_env',
            'attr1': 'value_attr1',
            'attr2': 'value_attr2',
            }

  def test_episode_metadata(self):
    episode_metadata_fn = envlogger_utils.get_standard_episode_metadata_fn(
        agent_spec=self.get_agent_spec(),
        env_config=self.get_env_config())

    env, dummy_action, dummy_observation = None, None, None

    # Episode #0.
    ts = dm_env.restart(dummy_observation)
    metadata = episode_metadata_fn(ts, dummy_action, env)
    self.assertTrue(metadata[rlds_types.INVALID])
    episode0_id = metadata[rlds_types.EPISODE_ID]

    for _ in range(3):
      ts = dm_env.transition(reward=0., observation=dummy_observation)
      metadata = episode_metadata_fn(ts, dummy_action, env)
      if metadata is not None:
        self.assertTrue(metadata[rlds_types.INVALID])

    ts = dm_env.termination(reward=0., observation=dummy_observation)
    metadata = episode_metadata_fn(ts, dummy_action, env)
    self.assertFalse(metadata[rlds_types.INVALID])
    self.assertEqual(metadata[rlds_types.EPISODE_ID], episode0_id)

    # Episode #1.
    ts = dm_env.restart(dummy_observation)
    metadata = episode_metadata_fn(ts, dummy_action, env)
    self.assertNotEqual(metadata[rlds_types.EPISODE_ID], episode0_id)


if __name__ == '__main__':
  absltest.main()
