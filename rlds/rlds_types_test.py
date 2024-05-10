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
"""Tests types."""

from absl.testing import absltest
from rlds import rlds_types
import tensorflow as tf


class TypesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.valid_steps = {
        rlds_types.OBSERVATION: [1, 2, 3],
        rlds_types.ACTION: [2, 3, 4],
        rlds_types.REWARD: [3.0, 4.0, 5.0],
        rlds_types.DISCOUNT: [1.0, 1.0, 0.0],
        rlds_types.IS_TERMINAL: [False, False, True],
        rlds_types.IS_FIRST: [True, False, False],
        rlds_types.IS_LAST: [False, False, True],
    }
    self.valid_step_dataset = tf.data.Dataset.from_tensor_slices(
        self.valid_steps)

  def build_valid_episode_dataset(
      self, steps_dataset: tf.data.Dataset) -> tf.data.Dataset:

    def episode_generator():
      for _ in range(0, 3):
        yield {rlds_types.STEPS: steps_dataset}
      return

    return tf.data.Dataset.from_generator(
        episode_generator,
        output_signature={
            rlds_types.STEPS: tf.data.DatasetSpec.from_value(steps_dataset)
        })

  def test_valid_dataset_is_valid(self):
    episode_dataset = self.build_valid_episode_dataset(self.valid_step_dataset)
    self.assertTrue(rlds_types.is_valid_rlds_dataset(episode_dataset))

  def test_invalid_step_dataset_is_invalid(self):
    invalid_steps = {
        rlds_types.OBSERVATION: [1, 2, 3],
        rlds_types.ACTION: [2, 3, 4],
    }
    invalid_step_dataset = tf.data.Dataset.from_tensor_slices(invalid_steps)
    episode_dataset = self.build_valid_episode_dataset(invalid_step_dataset)

    self.assertFalse(rlds_types.is_valid_rlds_dataset(episode_dataset))

  def test_invalid_episode_dataset_is_invalid(self):

    def episode_generator():
      for _ in range(0, 3):
        yield {'mysteps': self.valid_step_dataset}
      return

    invalid_episode_dataset = tf.data.Dataset.from_generator(
        episode_generator,
        output_signature={
            'mysteps': tf.data.DatasetSpec.from_value(self.valid_step_dataset)
        })

    self.assertFalse(rlds_types.is_valid_rlds_dataset(invalid_episode_dataset))

  def test_valid_step_metadata_is_valid_dataset(self):
    steps_list = rlds_types.build_step(
        observation=self.valid_steps[rlds_types.OBSERVATION],
        action=self.valid_steps[rlds_types.ACTION],
        reward=self.valid_steps[rlds_types.REWARD],
        discount=self.valid_steps[rlds_types.DISCOUNT],
        is_terminal=self.valid_steps[rlds_types.IS_TERMINAL],
        is_first=self.valid_steps[rlds_types.IS_FIRST],
        is_last=self.valid_steps[rlds_types.IS_LAST],
        metadata={
            'extra_field': [1.0, 2.0, 3.0],
        })
    steps_dataset = tf.data.Dataset.from_tensor_slices(steps_list)
    expected_extra_field = []
    for step in steps_dataset:
      expected_extra_field.append(step['extra_field'])
    self.assertEqual(expected_extra_field, [1.0, 2.0, 3.0])

    episode_dataset = self.build_valid_episode_dataset(steps_dataset)
    self.assertTrue(rlds_types.is_valid_rlds_dataset(episode_dataset))

  def test_unset_step_metadata_is_valid_dataset(self):
    steps_list = rlds_types.build_step(
        observation=self.valid_steps[rlds_types.OBSERVATION],
        action=self.valid_steps[rlds_types.ACTION],
        reward=self.valid_steps[rlds_types.REWARD],
        discount=self.valid_steps[rlds_types.DISCOUNT],
        is_terminal=self.valid_steps[rlds_types.IS_TERMINAL],
        is_first=self.valid_steps[rlds_types.IS_FIRST],
        is_last=self.valid_steps[rlds_types.IS_LAST])
    steps_dataset = tf.data.Dataset.from_tensor_slices(steps_list)

    episode_dataset = self.build_valid_episode_dataset(steps_dataset)
    self.assertTrue(rlds_types.is_valid_rlds_dataset(episode_dataset))

  def test_invalid_step_metadata_is_invalid(self):
    with self.assertRaisesRegex(ValueError,
                                'Invalid Step Metadata.*observation'):
      _ = rlds_types.build_step(
          observation=self.valid_steps[rlds_types.OBSERVATION],
          action=self.valid_steps[rlds_types.ACTION],
          reward=self.valid_steps[rlds_types.REWARD],
          discount=self.valid_steps[rlds_types.DISCOUNT],
          is_terminal=self.valid_steps[rlds_types.IS_TERMINAL],
          is_first=self.valid_steps[rlds_types.IS_FIRST],
          is_last=self.valid_steps[rlds_types.IS_LAST],
          metadata={rlds_types.OBSERVATION: [1.0, 2.0, 3.0]})

  def test_invalid_episode_metadata_is_invalid(self):
    with self.assertRaisesRegex(ValueError,
                                'Invalid Episode Metadata.*'):
      _ = rlds_types.build_episode(
          steps=self.valid_step_dataset, metadata={rlds_types.STEPS: [0, 1, 2]})


if __name__ == '__main__':
  absltest.main()
