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
"""Internal helpers to perform batched operations."""

from typing import Callable, Optional

from rlds import rlds_types
from rlds.transformations import flexible_batch
import tensorflow as tf


def batched_map(steps: tf.data.Dataset,
                map_fn: Callable[[rlds_types.BatchedStep],
                                 rlds_types.BatchedStep],
                size: int = flexible_batch.BATCH_AUTO_TUNE,
                shift: Optional[int] = None,
                stride: int = 1) -> tf.data.Dataset:
  """Applies a function to a dataset of steps.

  Args:
    steps: dataset of steps.
    map_fn: function to apply to a batch of steps.
    size: Size of the batch. By default we use a large batch size so many
      episodes will have only one batch element.
    shift: increment to compute the index to start the next batch (shift=1 means
      that we create a batch for each element in the input dataset). Must be
      positive.
    stride: increment to compute the index to select the next element of each
      batch (stride=1 means that we include consecutive elements in the batch).
      Must be positive.

  Returns:
    The unbatched dataset of steps after applying the operation.

  """
  size = flexible_batch.get_batch_size(steps, size)
  options = tf.data.Options()
  # We disable the map & batch fusion because this function uses a batch size
  # larger than the size of the episode (sometimes, much larger). Because of
  # this, merging map & batch operations assumes that there are more elements in
  # the episode than the real size and it's very inefficient in memory
  # consumption.
  options.experimental_optimization.map_and_batch_fusion = False
  steps = steps.with_options(options)
  steps = flexible_batch.batch(steps, size, shift, stride)
  steps = steps.map(map_fn)
  return steps.unbatch()
