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
"""Tests for nested_ops."""

from typing import Any, Dict

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from rlds import rlds_types
from rlds.transformations import flexible_batch
from rlds.transformations import nested_ops
from rlds.transformations import transformations_testlib
import tensorflow as tf


class NestedOpsTest(transformations_testlib.TransformationsTest):

  def setUp(self):
    super().setUp()
    steps1 = {
        rlds_types.OBSERVATION: {
            'field0': [[1., 0.], [0., 1.], [0., 2.]],
            'field1': [[1., 0.], [1., 1.], [1., 2.]]
        },
        rlds_types.ACTION: ([0, 10, 20], [10, 11, 21], [20, 21, 22]),
        rlds_types.REWARD: [0.0, 1.0, 2.0],
        rlds_types.IS_TERMINAL: [False, False, True],
        rlds_types.IS_FIRST: [True, False, False],
    }
    steps2 = {
        rlds_types.OBSERVATION: {
            'field0': [[3., 0.], [1., 1.], [1., 2.]],
            'field1': [[1., 3.], [1., 4.], [1., 2.]]
        },
        rlds_types.ACTION: ([0, 10, 20], [10, 11, 21], [20, 21, 22]),
        rlds_types.REWARD: [0.0, 1.0, 2.0],
        rlds_types.IS_TERMINAL: [False, False, True],
        rlds_types.IS_FIRST: [True, False, False],
    }

    self.steps1_dataset = tf.data.Dataset.from_tensor_slices(steps1)
    self.steps2_dataset = tf.data.Dataset.from_tensor_slices(steps2)
    self.episodes_dataset = tf.data.Dataset.from_tensor_slices({
        rlds_types.STEPS: [self.steps1_dataset, self.steps2_dataset],
    })
    self.obs_mean = {
        'field0': tf.constant([1., 1.], dtype=tf.float64),
        'field1': tf.constant([1., 2.], dtype=tf.float64),
    }
    self.obs_std = {
        'field0': tf.constant([1., np.sqrt(np.float64(4) / np.float64(6))]),
        'field1': tf.constant([0., np.sqrt(np.float64(10) / np.float64(6))]),
    }

  def check_nested_equality(self, result, expected):
    if isinstance(expected, dict):
      self.assertLen(list(result.keys()), len(list(expected.keys())))
      for k in expected:
        self.check_nested_equality(result[k], expected[k])
    else:
      self.assertLen(result, len(expected))
      for k, _ in enumerate(expected):
        self.assertEqual(result[k], expected[k])

  def test_map_episode(self):
    def add_one_to_reward(step: Dict[str, Any]) -> Dict[str, Any]:
      step[rlds_types.REWARD] += 1
      return step

    episode1 = {rlds_types.STEPS: self.steps1_dataset,
                'sample_metadata': 'metadata_value'}
    episode2_in_place = nested_ops._map_episode(
        episode1, add_one_to_reward, in_place=True)
    episode2 = nested_ops._map_episode(
        episode1, add_one_to_reward, in_place=False)
    self.assertIn('sample_metadata', episode2)
    for step1, step2_in_place, step2 in tf.data.Dataset.zip(
        (episode1[rlds_types.STEPS],
         episode2_in_place[rlds_types.STEPS],
         episode2[rlds_types.STEPS])):
      self.assertTrue(
          tf.reduce_all(tf.equal(step1[rlds_types.REWARD],
                                 step2_in_place[rlds_types.REWARD])))
      self.assertTrue(
          tf.reduce_all(tf.equal(step1[rlds_types.REWARD] + 1,
                                 step2[rlds_types.REWARD])))

  @parameterized.parameters((1,), (2,), (flexible_batch.BATCH_AUTO_TUNE,))
  def test_nested_map(self, batch_size):
    shift = tf.nest.map_structure(lambda x: -tf.cast(x, tf.float32),
                                  self.obs_mean)
    scale = tf.nest.map_structure(
        lambda x: tf.cast(1.0 / np.maximum(x, 1e-3), tf.float32), self.obs_std)

    def normalize_step(step: Dict[str, Any]) -> Dict[str, Any]:
      step[rlds_types.OBSERVATION] = tf.nest.map_structure(
          lambda x, x_offset, x_scale: (x + x_offset) * x_scale,
          step[rlds_types.OBSERVATION], shift, scale)
      return step

    normalized_ds = nested_ops.map_nested_steps(
        self.episodes_dataset,
        normalize_step,
        optimization_batch_size=batch_size)

    two_ds = tf.data.Dataset.zip(
        (self.episodes_dataset.flat_map(lambda x: x[rlds_types.STEPS]),
         normalized_ds.flat_map(lambda x: x[rlds_types.STEPS])))
    for (sample, normalized) in two_ds:
      normalized_obs = normalized[rlds_types.OBSERVATION]
      expected_obs = tf.nest.map_structure(
          lambda obs, shift, scale: (obs + shift) * scale,
          sample[rlds_types.OBSERVATION], shift, scale)
      for k in expected_obs:
        self.assertEqual(expected_obs[k].shape, normalized_obs[k].shape)
        self.assertTrue(
            tf.reduce_all(tf.equal(expected_obs[k], normalized_obs[k])))

  @parameterized.parameters((1,), (2,), (flexible_batch.BATCH_AUTO_TUNE,))
  def test_map_steps(self, batch_size):
    shift = tf.nest.map_structure(lambda x: -tf.cast(x, tf.float32),
                                  self.obs_mean)
    scale = tf.nest.map_structure(
        lambda x: tf.cast(1.0 / np.maximum(x, 1e-3), tf.float32), self.obs_std)

    def normalize_step(step: Dict[str, Any]) -> Dict[str, Any]:
      step[rlds_types.OBSERVATION] = tf.nest.map_structure(
          lambda x, x_offset, x_scale: (x + x_offset) * x_scale,
          step[rlds_types.OBSERVATION], shift, scale)
      return step

    steps_dataset = self.episodes_dataset.flat_map(
        lambda x: x[rlds_types.STEPS])
    normalized_ds = nested_ops.map_steps(
        steps_dataset,
        normalize_step,
        optimization_batch_size=batch_size)

    two_ds = tf.data.Dataset.zip((steps_dataset, normalized_ds))
    for (sample, normalized) in two_ds:
      normalized_obs = normalized[rlds_types.OBSERVATION]
      expected_obs = tf.nest.map_structure(
          lambda obs, shift, scale: (obs + shift) * scale,
          sample[rlds_types.OBSERVATION], shift, scale)
      for k in expected_obs:
        self.assertEqual(expected_obs[k].shape, normalized_obs[k].shape)
        self.assertTrue(
            tf.reduce_all(tf.equal(expected_obs[k], normalized_obs[k])))

  def test_nested_apply(self):
    def truncate_episode(steps):
      return steps.take(2)

    dataset = nested_ops.apply_nested_steps(self.episodes_dataset,
                                            truncate_episode)
    for episode in dataset:
      steps = episode[rlds_types.STEPS]
      episode_length = steps.reduce(0, lambda x, step: x + 1)
      self.assertEqual(episode_length, 2)

  @parameterized.parameters((1,), (2,), (flexible_batch.BATCH_AUTO_TUNE,))
  def test_total_sum_tfdata(self, batch_size):
    expected_sum = {
        rlds_types.OBSERVATION: {
            'field0': [6., 6.],
            'field1': [6., 12.],
        },
        rlds_types.ACTION: (60, 84, 126),
    }
    def data_to_sum(step):
      return {
          rlds_types.OBSERVATION: step[rlds_types.OBSERVATION],
          rlds_types.ACTION: step[rlds_types.ACTION]
      }

    total_sum = nested_ops.sum_nested_steps(
        self.episodes_dataset, data_to_sum, optimization_batch_size=batch_size)

    self.expect_nested_dict_equality(total_sum, expected_sum)

  @parameterized.parameters((1,), (2,), (flexible_batch.BATCH_AUTO_TUNE,))
  def test_sum_episode_tfdata(self, batch_size):
    expected_sum = {
        rlds_types.OBSERVATION: {
            'field0': [1., 3.],
            'field1': [3., 3.],
        },
        rlds_types.ACTION: (30, 42, 63),
    }
    def data_to_sum(step):
      return {
          rlds_types.OBSERVATION: step[rlds_types.OBSERVATION],
          rlds_types.ACTION: step[rlds_types.ACTION]
      }

    obs_action = self.steps1_dataset.map(
        lambda step:
        {k: step[k] for k in [rlds_types.OBSERVATION, rlds_types.ACTION]})


    total_sum = nested_ops.sum_dataset(
        obs_action, data_to_sum,
        optimization_batch_size=batch_size)

    self.expect_nested_dict_equality(total_sum, expected_sum)

  def test_final_step_tfdata(self):
    step = nested_ops.final_step(self.steps1_dataset)
    self.assertTrue(
        tf.reduce_all(tf.equal(step[rlds_types.OBSERVATION]['field0'],
                               [0., 2.])))

  def test_episode_length_tfdata(self):
    result = nested_ops.episode_length(
        self.steps1_dataset, optimization_batch_size=0)
    self.assertTrue(result, 3)

  def test_episode_length_batched(self):
    result = nested_ops.episode_length(self.steps1_dataset)
    self.assertTrue(result, 3)


if __name__ == '__main__':
  absltest.main()
