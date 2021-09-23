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
"""Library functions for flexible batching."""

from typing import Any, Dict, Optional, Tuple, Union

from rlds import rlds_types
from rlds.transformations import shape_ops
import tensorflow as tf

# Used by the methods accepting optimization_batch_size to specify that batch
# size is to be selected automatically trying to optimize the pipeline
# execution.
BATCH_AUTO_TUNE = -1

# Batch size (in bytes) to target when batching map / reduce calls. Value was
# selected experimentally to balance between the cost of batching vs cost of
# map/reduce calls.
DEFAULT_BATCH_SIZE_IN_BYTES = 200_000


def get_batch_size(dataset: tf.data.Dataset,
                   optimization_batch_size: int = BATCH_AUTO_TUNE) -> int:
  """Computes optimization batch size for processing steps in batches.

  Args:
    dataset: episodes or steps dataset.
    optimization_batch_size: user-provided hint for the batch size.

  Returns:
    Size of the optimization batch.
  """
  if optimization_batch_size != BATCH_AUTO_TUNE:
    return optimization_batch_size
  if rlds_types.STEPS in dataset.element_spec:
    spec = dataset.element_spec[rlds_types.STEPS].element_spec
  else:
    spec = dataset.element_spec
  step_size_in_bytes = shape_ops.size_from_spec(spec)
  return  max(1, DEFAULT_BATCH_SIZE_IN_BYTES // step_size_in_bytes)


def _windowed_to_batched_dataset(
    nested_dataset: Union[Dict[str, Any], Tuple[Any], tf.data.Dataset],
    batch_size: int) -> Union[Dict[str, Any], Tuple[Any], Any]:
  """Converts a nested windowed dataset into a batch.

  Args:
    nested_dataset: nested dataset that has been generated with a window
      transformation.
    batch_size: desired batch size (it has to correspond to the size used for
      the window transformation).

  Returns:
    Batch that respects the nested structure of the given dataset.

  """
  return tf.nest.map_structure(
      lambda ds: tf.data.experimental.get_single_element(
          ds.batch(batch_size)),
      nested_dataset)


def batch(dataset: tf.data.Dataset,
          size: int = BATCH_AUTO_TUNE,
          shift: Optional[int] = None,
          stride: int = 1,
          drop_remainder: bool = False) -> tf.data.Dataset:
  """Batches dataset elements using tf.data.Dataset window interface.

  It is equivalent to tf.data.Dataset.window but flattens the nested datasets.

  Args:
    dataset: dataset to be batched.
    size: number of elements per batch. The default value (BATCH_AUTO_TUNE)
      makes size selection automatic to target efficient execution.
    shift: increment to compute the index to start the next batch (shift=1 means
      that we create a batch for each element in the input dataset). Must be
      positive. If not specified, shift is defaults to size.
    stride: increment to compute the index to select the next element of each
      batch (stride=1 means that we include consecutive elements in the batch).
      Must be positive.
    drop_remainder: whether the last batches should be dropped if their size is
      smaller than size.

  Returns:
    A dataset where each element has been batched according to the
    configuration parameters.

  Examples:

  If dataset ds contains (1, 2, 3, 4):

   * batch(ds, size=2, shift=1, stride=1, False)->([1, 2], [2, 3], [3, 4], [4])
   * batch(ds, size=2, shift=1, stride=1, True) -> ([1, 2], [2, 3], [3, 4])
   * batch(ds, size=2, shift=2, stride=1, False)->([1, 2], [3, 4])
   * batch(ds, size=2, shift=1, stride=2, False)->([1, 3], [2, 4], [3], [4])

  """
  size = get_batch_size(dataset, size)
  if (not shift or shift == size) and stride == 1:
    return dataset.batch(batch_size=size, drop_remainder=drop_remainder)
  windowed = dataset.window(
      size=size, shift=shift, stride=stride, drop_remainder=drop_remainder)
  return windowed.map(
      lambda windowed_ds: _windowed_to_batched_dataset(windowed_ds, size))
