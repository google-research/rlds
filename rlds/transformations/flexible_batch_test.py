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
"""Tests for flexible_batch."""
from absl.testing import absltest
from rlds import rlds_types
from rlds.transformations import flexible_batch
from rlds.transformations import transformations_testlib
import tensorflow as tf


# If any of the tests fail for strange reasons, make sure that TF didn't remove
# or change internal 'sliding_window_dataset' OP.
class StepBatchTest(transformations_testlib.TransformationsTest):

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

  def test_window_batch_drop_reminder(self):
    expected_steps = {
        rlds_types.OBSERVATION: {
            'field0': [[[0, 0], [0, 1]], [[0, 1], [0, 2]]],
            'field1': [[[1, 0], [1, 1]], [[1, 1], [1, 2]]]
        },
        rlds_types.ACTION: ([[0, 10], [10, 20]], [[10, 11], [11,
                                                             21]], [[20, 21],
                                                                    [21, 22]]),
        rlds_types.REWARD: [[0.0, 1.0], [1.0, 2.0]],
        rlds_types.IS_TERMINAL: [[False, False], [False, True]],
        rlds_types.IS_FIRST: [[True, False], [False, False]],
    }
    steps_dataset = flexible_batch.batch(
        tf.data.Dataset.from_tensor_slices(self.steps),
        size=2,
        shift=1,
        stride=1,
        drop_remainder=True)
    expected_dataset = tf.data.Dataset.from_tensor_slices(expected_steps)

    self.expect_equal_datasets(steps_dataset, expected_dataset)

  def test_window_batch_dont_drop_reminder(self):
    steps_dataset = flexible_batch.batch(
        tf.data.Dataset.from_tensor_slices(self.steps),
        size=2,
        shift=1,
        stride=1,
        drop_remainder=False)
    length = steps_dataset.reduce(0, lambda count, episode: count+1)
    self.assertEqual(length, 3)

  def test_batch_default_equivalent_to_batch(self):
    steps_dataset = flexible_batch.batch(
        tf.data.Dataset.from_tensor_slices(self.steps),
        size=2,
        drop_remainder=True)
    expected_dataset = tf.data.Dataset.from_tensor_slices(self.steps).batch(
        batch_size=2, drop_remainder=True)
    self.assertLen(steps_dataset, 1)

    self.expect_equal_datasets(steps_dataset, expected_dataset)

  def test_window_for_simple_dataset(self):
    ds = tf.data.Dataset.from_tensors(42).repeat(100)

    generated_dataset = flexible_batch.batch(
        ds, size=2, shift=1, stride=1, drop_remainder=True)
    expected_dataset = ds.window(
        size=2, shift=1, stride=1,
        drop_remainder=True).flat_map(lambda element: element.batch(3))

    for e1, e2 in zip(generated_dataset, expected_dataset):
      self.assertTrue(all(e1 == e2))

  def test_window_for_tuple_dataset(self):
    ds = tf.data.Dataset.from_tensors([1]).repeat(100)

    generated_dataset = flexible_batch.batch(
        ds, size=2, shift=1, stride=1, drop_remainder=True)
    expected_dataset = ds.window(
        size=2, shift=1, stride=1,
        drop_remainder=True).flat_map(lambda element: element.batch(3))

    for e1, e2 in zip(generated_dataset, expected_dataset):
      self.assertTrue(all(e1 == e2))


if __name__ == '__main__':
  absltest.main()
