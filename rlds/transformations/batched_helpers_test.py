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
"""Tests for batched_helpers."""

from absl.testing import absltest
from rlds import rlds_types
from rlds.transformations import batched_helpers
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

  def test_batched_map(self):
    steps_dataset = tf.data.Dataset.from_tensor_slices(self.steps)

    def increase_reward(step):
      step[rlds_types.REWARD] = step[rlds_types.REWARD] + 1.0
      return step

    expected_result = steps_dataset.map(increase_reward)
    result = batched_helpers.batched_map(steps_dataset, increase_reward)

    self.expect_equal_datasets(result, expected_result)


if __name__ == '__main__':
  absltest.main()
