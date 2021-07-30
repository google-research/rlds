# Copyright 2021 Google LLC.
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

  def test_batch_with_overlap_works(self):
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
    self.assertLen(steps_dataset, 2)

    self.expect_equal_datasets(steps_dataset, expected_dataset)

  def test_batch_default_equivalent_to_batch(self):
    steps_dataset = flexible_batch.batch(
        tf.data.Dataset.from_tensor_slices(self.steps),
        size=2,
        drop_remainder=True)
    expected_dataset = tf.data.Dataset.from_tensor_slices(self.steps).batch(
        batch_size=2, drop_remainder=True)
    self.assertLen(steps_dataset, 1)

    self.expect_equal_datasets(steps_dataset, expected_dataset)


if __name__ == '__main__':
  absltest.main()
