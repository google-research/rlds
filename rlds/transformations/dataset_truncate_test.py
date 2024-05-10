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
"""Tests for dataset_truncate."""

from absl.testing import absltest
from rlds import rlds_types
from rlds.transformations import dataset_truncate
from rlds.transformations import transformations_testlib
import tensorflow as tf


class DatasetTruncateTest(transformations_testlib.TransformationsTest):

  def test_conditional_truncation(self):
    steps = {
        rlds_types.OBSERVATION: {
            'field0': [[1, 0], [0, 1], [0, 2], [4, 5]],
            'field1': [[1, 0], [1, 1], [1, 2], [6, 7]],
        },
        rlds_types.IS_TERMINAL: [False, False, True, False],
    }
    ds = tf.data.Dataset.from_tensor_slices(steps)

    truncated = dataset_truncate.truncate_after_condition(
        ds, lambda step: step[rlds_types.IS_TERMINAL])

    self.assertEqual(truncated.reduce(0, lambda count, x: count+1), 3)
    for i, step in enumerate(truncated):
      self.assertEqual(step[rlds_types.IS_TERMINAL], i == 2)


if __name__ == '__main__':
  absltest.main()
