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
"""Tests for dataset_concat."""

from absl.testing import absltest
from rlds import rlds_types
from rlds.transformations import dataset_concat
from rlds.transformations import shape_ops
from rlds.transformations import transformations_testlib
import tensorflow as tf


class DatasetConcatTest(transformations_testlib.TransformationsTest):

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

  def test_concatenate_empty_step(self):
    ds = tf.data.Dataset.from_tensor_slices(self.steps)

    concatenated = dataset_concat.concatenate(
        shape_ops.zero_dataset_like(ds), ds)

    expected = shape_ops.zero_dataset_like(ds).concatenate(ds)

    self.expect_equal_datasets(concatenated, expected)

  def test_concatenate_extra_fields(self):
    steps1 = {
        rlds_types.OBSERVATION: [1, 2, 3],
        rlds_types.ACTION: [2, 3, 4],
    }

    steps2 = {
        rlds_types.REWARD: [1., 1., 1.],
        rlds_types.IS_TERMINAL: [False, False, True]
    }

    joined_steps = {
        rlds_types.OBSERVATION: [1, 2, 3, 0, 0, 0],
        rlds_types.ACTION: [2, 3, 4, 0, 0, 0],
        rlds_types.REWARD: [0., 0., 0., 1., 1., 1.],
        rlds_types.IS_TERMINAL: [False, False, False, False, False, True],
    }

    ds1 = tf.data.Dataset.from_tensor_slices(steps1)
    ds2 = tf.data.Dataset.from_tensor_slices(steps2)
    joined = tf.data.Dataset.from_tensor_slices(joined_steps)
    concatenated = dataset_concat.concatenate(ds1, ds2)

    self.expect_equal_datasets(concatenated, joined)

  def test_concatenate_extra_fields_with_intersection(self):
    steps1 = {
        rlds_types.OBSERVATION: [1, 2, 3],
        rlds_types.ACTION: [2, 3, 4],
        'extra_data': [4, 5, 6],
    }

    steps2 = {
        rlds_types.REWARD: [1., 1., 1.],
        rlds_types.IS_TERMINAL: [False, False, True],
        'extra_data': [7, 8, 9],
    }

    joined_steps = {
        rlds_types.OBSERVATION: [1, 2, 3, 0, 0, 0],
        rlds_types.ACTION: [2, 3, 4, 0, 0, 0],
        rlds_types.REWARD: [0., 0., 0., 1., 1., 1.],
        rlds_types.IS_TERMINAL: [False, False, False, False, False, True],
        'extra_data': [4, 5, 6, 7, 8, 9],
    }

    ds1 = tf.data.Dataset.from_tensor_slices(steps1)
    ds2 = tf.data.Dataset.from_tensor_slices(steps2)
    joined = tf.data.Dataset.from_tensor_slices(joined_steps)
    concatenated = dataset_concat.concatenate(ds1, ds2)

    self.expect_equal_datasets(concatenated, joined)

  def test_concat_if_terminal_tfdata(self):
    dataset = tf.data.Dataset.from_tensor_slices(self.steps)

    def make_extra_steps(_):
      return shape_ops.zero_dataset_like(dataset)

    steps_with_absorbing = dataset_concat.concat_if_terminal(
        dataset,
        make_extra_steps)

    self.expect_equal_datasets(steps_with_absorbing.take(3), dataset)

    self.expect_equal_datasets(
        steps_with_absorbing.skip(3).take(1),
        shape_ops.zero_dataset_like(dataset))

if __name__ == '__main__':
  absltest.main()
