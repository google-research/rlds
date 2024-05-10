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
"""Tests for shape_ops."""

from absl.testing import absltest
from rlds import rlds_types
from rlds.transformations import shape_ops
from rlds.transformations import transformations_testlib
import tensorflow as tf


class ShapeOpsTest(transformations_testlib.TransformationsTest):

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

  def test_zeros_from_nested_spec(self):
    ds = tf.data.Dataset.from_tensor_slices(self.steps)
    zero_step = shape_ops.zeros_from_spec(ds.element_spec)
    expected_result = tf.nest.map_structure(tf.zeros_like, next(iter(ds)))
    self.expect_equal_step(zero_step, expected_result)

  def test_zeros_from_batched_step(self):
    ds = tf.data.Dataset.from_tensor_slices(self.steps)
    batched_ds = ds.batch(2)
    zero_step = shape_ops.zeros_from_spec(batched_ds.element_spec)
    expected_result = tf.nest.map_structure(tf.zeros_like,
                                            next(iter(ds.batch(1))))
    self.expect_equal_step(zero_step, expected_result)

  def test_zeros_dataset_like(self):
    ds = tf.data.Dataset.from_tensor_slices(self.steps)
    zero_ds = shape_ops.zero_dataset_like(ds)

    num_elements = zero_ds.reduce(0, lambda x, y: x + 1)
    self.assertEqual(num_elements, 1)

    zero_step = tf.data.experimental.get_single_element(zero_ds)
    expected_result = tf.nest.map_structure(tf.zeros_like, next(iter(ds)))

    self.expect_equal_step(zero_step, expected_result)


if __name__ == '__main__':
  absltest.main()
