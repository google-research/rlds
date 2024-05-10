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
"""Tests for alignment."""
from absl.testing import absltest
from rlds import rlds_types
from rlds.transformations import alignment
from rlds.transformations import transformations_testlib
import tensorflow.compat.v2 as tf


class AlignmentTest(transformations_testlib.TransformationsTest):

  def setUp(self):
    super().setUp()
    self.steps = {
        rlds_types.OBSERVATION: {
            'field0': [[0, 0], [0, 1], [0, 2]],
            'field1': [[1, 0], [1, 1], [1, 2]]
        },
        rlds_types.ACTION: ([0, 10, 20], [10, 11, 21], [20, 21, 22]),
        rlds_types.REWARD: [0.0, 1.0, 2.0],
        rlds_types.IS_TERMINAL: [False, False, True],
        rlds_types.IS_FIRST: [True, False, False],
    }

  def test_shift_observation_optimized(self):
    steps_ds = tf.data.Dataset.from_tensor_slices(self.steps)
    shifted_ds = alignment.shift_keys(steps_ds, [rlds_types.OBSERVATION], -1)

    expected_steps = {
        rlds_types.OBSERVATION: {
            'field0': [[0, 0], [0, 1]],
            'field1': [[1, 0], [1, 1]]
        },
        rlds_types.ACTION: ([10, 20], [11, 21], [21, 22]),
        rlds_types.REWARD: [1.0, 2.0],
        rlds_types.IS_TERMINAL: [False, True],
        rlds_types.IS_FIRST: [False, False],
    }

    expected_ds = tf.data.Dataset.from_tensor_slices(expected_steps)

    self.expect_equal_datasets(shifted_ds, expected_ds)

  def test_shift_observation_optimization_disabled(self):
    steps_ds = tf.data.Dataset.from_tensor_slices(self.steps)
    shifted_ds = alignment.shift_keys(
        steps_ds, [rlds_types.OBSERVATION], -1)

    expected_steps = {
        rlds_types.OBSERVATION: {
            'field0': [[0, 0], [0, 1]],
            'field1': [[1, 0], [1, 1]]
        },
        rlds_types.ACTION: ([10, 20], [11, 21], [21, 22]),
        rlds_types.REWARD: [1.0, 2.0],
        rlds_types.IS_TERMINAL: [False, True],
        rlds_types.IS_FIRST: [False, False],
    }

    expected_ds = tf.data.Dataset.from_tensor_slices(expected_steps)

    self.expect_equal_datasets(shifted_ds, expected_ds)

  def test_shift_observation_custom_batch(self):
    steps_ds = tf.data.Dataset.from_tensor_slices(self.steps)
    shifted_ds = alignment.shift_keys(
        steps_ds, [rlds_types.OBSERVATION],
        -1,
        batch_size=2)

    expected_steps = {
        rlds_types.OBSERVATION: {
            'field0': [[0, 0], [0, 1]],
            'field1': [[1, 0], [1, 1]]
        },
        rlds_types.ACTION: ([10, 20], [11, 21], [21, 22]),
        rlds_types.REWARD: [1.0, 2.0],
        rlds_types.IS_TERMINAL: [False, True],
        rlds_types.IS_FIRST: [False, False],
    }

    for e in shifted_ds:
      print(e)
    expected_ds = tf.data.Dataset.from_tensor_slices(expected_steps)

    self.expect_equal_datasets(shifted_ds, expected_ds)

  def test_wrong_batch_size_fails(self):
    steps_ds = tf.data.Dataset.from_tensor_slices(self.steps)

    with self.assertRaises(ValueError):
      alignment.shift_keys(steps_ds, [rlds_types.OBSERVATION], -3, 2)


if __name__ == '__main__':
  absltest.main()
