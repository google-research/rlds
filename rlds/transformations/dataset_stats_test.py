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
"""Tests for dataset_stats."""

from absl.testing import absltest
import numpy as np
from rlds import rlds_types
from rlds.transformations import dataset_stats
from rlds.transformations import transformations_testlib
import tensorflow as tf


def _get_data(step):
  if step[rlds_types.IS_LAST]:
    mask = {
        rlds_types.OBSERVATION: True,
        rlds_types.ACTION: False,
    }
  else:
    mask = {
        rlds_types.OBSERVATION: True,
        rlds_types.ACTION: True,
    }
  data = {rlds_types.OBSERVATION: step[rlds_types.OBSERVATION],
          rlds_types.ACTION: step[rlds_types.ACTION]}
  return data, mask


class DatasetStatsTest(transformations_testlib.TransformationsTest):

  def setUp(self):
    super().setUp()
    steps1 = {
        rlds_types.OBSERVATION: {
            'field0': [[1, 0], [0, 1], [0, 2]],
            'field1': [[1, 0], [1, 1], [1, 2]]
        },
        rlds_types.ACTION: ([0, 10, 20], [10, 11, 21], [20, 21, 22]),
        rlds_types.REWARD: [0.0, 1.0, 2.0],
        rlds_types.IS_TERMINAL: [False, False, True],
        rlds_types.IS_FIRST: [True, False, False],
        rlds_types.IS_LAST: [False, False, False],
    }
    steps2 = {
        rlds_types.OBSERVATION: {
            'field0': [[3, 0], [1, 1], [1, 2]],
            'field1': [[1, 3], [1, 4], [1, 2]]
        },
        rlds_types.ACTION: ([0, 10, 20], [10, 13, 22], [20, 22, 22]),
        rlds_types.REWARD: [0.0, 1.0, 2.0],
        rlds_types.IS_TERMINAL: [False, False, True],
        rlds_types.IS_FIRST: [True, False, False],
        # the action should be ignored for the last step of the second episode.
        rlds_types.IS_LAST: [False, False, True],
    }

    self.expected_obs_mean = {
        'field0': [1., 1.],
        'field1': [1., 2.],
    }
    self.expected_obs_std = {
        'field0': [
            np.sqrt(np.float64(6) / np.float64(5)),
            np.sqrt(np.float64(4) / np.float64(5))
        ],
        'field1': [0., np.sqrt(np.float64(10) / np.float64(5))],
    }
    self.expected_action_mean = (8., 13., 21)
    self.expected_action_std = (np.sqrt(np.float64(70)),
                                np.sqrt(np.float64(21.5)), 1.)

    self.steps1_dataset = tf.data.Dataset.from_tensor_slices(steps1)
    self.steps2_dataset = tf.data.Dataset.from_tensor_slices(steps2)
    self.episodes_dataset = tf.data.Dataset.from_tensor_slices({
        rlds_types.STEPS: [self.steps1_dataset, self.steps2_dataset],
    })

  def test_mean_std_observations_tfdata_batch1(self):
    mean, std = dataset_stats.mean_and_std(
        self.episodes_dataset, _get_data, optimization_batch_size=1)

    self.expect_nested_dict_equality(mean[rlds_types.OBSERVATION],
                                     self.expected_obs_mean)
    self.expect_nested_dict_equality(
        std[rlds_types.OBSERVATION], self.expected_obs_std, approximate=True)

    self.expect_nested_dict_equality(mean[rlds_types.ACTION],
                                     self.expected_action_mean)
    self.expect_nested_dict_equality(
        std[rlds_types.ACTION], self.expected_action_std, approximate=True)

  def test_mean_std_observations_batched(self):

    mean, std = dataset_stats.mean_and_std(self.episodes_dataset, _get_data)

    self.expect_nested_dict_equality(mean[rlds_types.OBSERVATION],
                                     self.expected_obs_mean)
    self.expect_nested_dict_equality(
        std[rlds_types.OBSERVATION], self.expected_obs_std, approximate=True)

    self.expect_nested_dict_equality(mean[rlds_types.ACTION],
                                     self.expected_action_mean)
    self.expect_nested_dict_equality(
        std[rlds_types.ACTION], self.expected_action_std, approximate=True)

  def test_sar_fields_mask(self):
    step = self.steps1_dataset.take(1).get_single_element()
    fields, mask = dataset_stats.sar_fields_mask(step)

    expected_fields = {
        rlds_types.OBSERVATION: step[rlds_types.OBSERVATION],
        rlds_types.ACTION: step[rlds_types.ACTION],
        rlds_types.REWARD: step[rlds_types.REWARD],
    }

    self.expect_equal_step(fields, expected_fields)

    self.assertTrue(mask[rlds_types.OBSERVATION])
    self.assertTrue(mask[rlds_types.ACTION])
    self.assertTrue(mask[rlds_types.REWARD])

  def test_sar_fields_mask_last_step(self):
    step = self.steps2_dataset.skip(2).take(1).get_single_element()
    fields, mask = dataset_stats.sar_fields_mask(step)

    expected_fields = {
        rlds_types.OBSERVATION: step[rlds_types.OBSERVATION],
        rlds_types.ACTION: step[rlds_types.ACTION],
        rlds_types.REWARD: step[rlds_types.REWARD],
    }

    self.expect_equal_step(fields, expected_fields)

    self.assertTrue(mask[rlds_types.OBSERVATION])
    self.assertFalse(mask[rlds_types.ACTION])
    self.assertFalse(mask[rlds_types.REWARD])


if __name__ == '__main__':
  absltest.main()
