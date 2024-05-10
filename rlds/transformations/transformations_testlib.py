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
"""Tests for transformations."""

from absl.testing import parameterized
import tensorflow as tf


class TransformationsTest(parameterized.TestCase):
  """Library for testing RLDS transformations.

  It provides functions to check equality of datasets.
  """

  def expect_equal_step(self, step, expected_step):
    """Compares two steps and fails if they are not equal."""
    for k, expected in expected_step.items():
      if isinstance(step[k], tf.data.Dataset):
        self.expect_equal_datasets(step[k], expected_step[k])
      elif isinstance(step[k], dict):
        for inner_key in step[k]:
          self.assertEqual(step[k][inner_key].shape,
                           expected[inner_key].shape)
          self.assertTrue(
              tf.reduce_all(
                  tf.equal(step[k][inner_key], expected[inner_key])))
      elif isinstance(step[k], tuple):
        for v, expected_v in zip(step[k], expected):
          self.assertEqual(v.shape, expected_v.shape)
          self.assertTrue(tf.reduce_all(tf.equal(v, expected_v)))
      else:
        self.assertEqual(step[k].shape, expected.shape)
        self.assertTrue(tf.reduce_all(tf.equal(step[k], expected)))

  def expect_equal_datasets(self, ds, expected_ds):
    """Compares two datasets and fails if they are not equal."""
    ds_length = ds.reduce(0, lambda x, step: x + 1)
    expected_length = expected_ds.reduce(0, lambda x, step: x + 1)

    self.assertEqual(ds_length, expected_length)

    ds_iter = iter(ds)
    for expected_item in iter(expected_ds):
      dataset_item = next(ds_iter)
      self.expect_equal_step(dataset_item, expected_item)

  def expect_nested_dict_equality(self, obtained, expected, approximate=False):
    """Compares two nested dicts and fails if they are not equal.

    Args:
      obtained: result obtained in the test computation.
      expected: expected result.
      approximate: if True, checks for approximate equality (maximum difference
        accepted is 1e-5).
    """
    if isinstance(expected, dict):
      self.assertLen(list(obtained.keys()), len(list(expected.keys())))
      for k in expected:
        self.expect_nested_dict_equality(obtained[k], expected[k], approximate)
    else:
      self.assertLen(obtained, len(expected))
      for k, _ in enumerate(expected):
        if approximate:
          self.assertLess(
              abs(obtained[k] - expected[k]),
              1e-5,
              msg=f'{k}: result:{obtained} vs expected:{expected}')
        else:
          self.assertEqual(obtained[k], expected[k])
