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
"""Tests for pattern."""

from absl.testing import absltest
from absl.testing import parameterized
import reverb
from rlds import rlds_types
from rlds.transformations import pattern
from rlds.transformations import transformations_testlib
import tensorflow as tf


def _get_sars_pattern(step):
  return {
      rlds_types.OBSERVATION:
          tf.nest.map_structure(lambda x: x[-2], step[rlds_types.OBSERVATION]),
      rlds_types.ACTION:
          tf.nest.map_structure(lambda x: x[-2], step[rlds_types.ACTION]),
      rlds_types.REWARD:
          step[rlds_types.REWARD][-2],
      'next_observation':
          tf.nest.map_structure(lambda x: x[-1], step[rlds_types.OBSERVATION]),
  }


class PatternTest(transformations_testlib.TransformationsTest,
                  parameterized.TestCase):

  def setUp(self):
    super().setUp()
    steps0 = {
        rlds_types.OBSERVATION: {
            'field0': [[0, 0], [0, 1], [0, 2]],
            'field1': [[1, 0], [1, 1], [1, 2]]
        },
        rlds_types.ACTION: ([0, 10, 20], [10, 11, 21], [20, 21, 22]),
        rlds_types.REWARD: [0.0, 1.0, 2.0],
        rlds_types.IS_TERMINAL: [False, False, True],
        rlds_types.IS_FIRST: [True, False, False],
        rlds_types.IS_LAST: [False, False, True],
    }
    steps1 = {
        rlds_types.OBSERVATION: {
            'field0': [[0, 0]],
            'field1': [[1, 0]]
        },
        rlds_types.ACTION: ([0], [10], [20]),
        rlds_types.REWARD: [0.0],
        rlds_types.IS_TERMINAL: [False],
        rlds_types.IS_FIRST: [True],
        rlds_types.IS_LAST: [True],
    }
    self.episodes = tf.data.Dataset.from_tensor_slices({
        rlds_types.STEPS: [
            tf.data.Dataset.from_tensor_slices(steps0),
            tf.data.Dataset.from_tensor_slices(steps1)
        ]
    })
    self.steps = tf.data.Dataset.from_tensor_slices(steps0)

  @parameterized.named_parameters(
      {
          'testcase_name':
              'Respect episode boundaries',
          'respect_episode_boundaries':
              True,
          'expected_transitions':
              tf.data.Dataset.from_tensor_slices({
                  rlds_types.OBSERVATION: {
                      'field0': [[0, 0], [0, 1]],
                      'field1': [[1, 0], [1, 1]]
                  },
                  rlds_types.ACTION: ([0, 10], [10, 11], [20, 21]),
                  rlds_types.REWARD: [0.0, 1.0],
                  'next_observation': {
                      'field0': [[0, 1], [0, 2]],
                      'field1': [[1, 1], [1, 2]]
                  }
              }),
      },
      {
          'testcase_name':
              'Do not respect episode boundaries',
          'respect_episode_boundaries':
              False,
          'expected_transitions':
              tf.data.Dataset.from_tensor_slices({
                  rlds_types.OBSERVATION: {
                      'field0': [[0, 0], [0, 1], [0, 2]],
                      'field1': [[1, 0], [1, 1], [1, 2]]
                  },
                  rlds_types.ACTION: ([0, 10, 20], [10, 11, 21], [20, 21, 22]),
                  rlds_types.REWARD: [0.0, 1.0, 2.0],
                  'next_observation': {
                      'field0': [[0, 1], [0, 2], [0, 0]],
                      'field1': [[1, 1], [1, 2], [1, 0]]
                  }
              }),
      },
  )
  def test_apply_pattern(self, respect_episode_boundaries,
                         expected_transitions):

    step_spec = pattern.step_spec(self.episodes)
    sars_pattern = reverb.structured_writer.pattern_from_transform(
        step_spec, _get_sars_pattern)
    sars_config = reverb.structured_writer.create_config(
        sars_pattern, table='transition')

    steps_dataset = pattern.pattern_map(
        self.episodes,
        configs=[sars_config],
        respect_episode_boundaries=respect_episode_boundaries)

    self.expect_equal_datasets(steps_dataset, expected_transitions)

  def test_step_spec(self):
    self.assertEqual(pattern.step_spec(self.episodes), self.steps.element_spec)

  def test_apply_pattern_from_transform(self):

    expected_transitions = tf.data.Dataset.from_tensor_slices({
        rlds_types.OBSERVATION: {
            'field0': [[0, 0], [0, 1]],
            'field1': [[1, 0], [1, 1]]
        },
        rlds_types.ACTION: ([0, 10], [10, 11], [20, 21]),
        rlds_types.REWARD: [0.0, 1.0],
        'next_observation': {
            'field0': [[0, 1], [0, 2]],
            'field1': [[1, 1], [1, 2]]
        }
    })

    steps_dataset = pattern.pattern_map_from_transform(
        self.episodes,
        transform_fn=_get_sars_pattern,
        respect_episode_boundaries=True)

    self.expect_equal_datasets(steps_dataset, expected_transitions)


if __name__ == '__main__':
  absltest.main()
