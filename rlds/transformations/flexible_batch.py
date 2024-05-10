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
"""Library functions for flexible batching."""

from typing import Any, Dict, Optional, Tuple, Union

from rlds import rlds_types
from rlds.transformations import shape_ops
import tensorflow as tf


from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops



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


def _windowed_to_batch(
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
      lambda ds: ds.batch(batch_size), nested_dataset)


def _windowed_map_to_batch(
    nested_dataset: Dict[str, Any],
    batch_size: int) -> Dict[str, Any]:
  """Converts a map of nested windowed datasets into a batch.

  Args:
    nested_dataset: map of nested datasets that have been generated with a
      window transformation.
    batch_size: desired batch size (it has to correspond to the size used for
      the window transformation).

  Returns:
    Batch that respects the nested structure of the given dataset.

  """
  keys = nested_dataset.keys()
  ds = tf.data.Dataset.zip(
      tuple([_windowed_to_batch(nested_dataset[k], batch_size) for k in keys]))
  return ds.map(lambda *args: {k: v for k, v in zip(keys, args)})


class _SlideDataset(dataset_ops.UnaryDataset):
  """Copy of deprecated sliding dataset transformations from tf.contrib.data."""

  def __init__(self, input_dataset, window_size, window_shift, window_stride,
               drop_remainder):
    """See `sliding_window_batch` for details."""
    self._input_dataset = input_dataset
    self._window_size = ops.convert_to_tensor(
        window_size, dtype=dtypes.int64, name='window_size')
    self._window_stride = ops.convert_to_tensor(
        window_stride, dtype=dtypes.int64, name='window_stride')
    self._window_shift = ops.convert_to_tensor(
        window_shift, dtype=dtypes.int64, name='window_shift')
    self._drop_remainder = drop_remainder

    input_structure = dataset_ops.get_structure(input_dataset)
    self._element_spec = nest.map_structure(
        lambda component_spec: component_spec._batch(None), input_structure)
    variant_tensor = ged_ops.sliding_window_dataset(
        self._input_dataset._variant_tensor,
        window_size=self._window_size,
        window_shift=self._window_shift,
        window_stride=self._window_stride,
        drop_remainder=self._drop_remainder,
        **self._flat_structure)
    super(_SlideDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


def _sliding_window_batch(size, shift, stride, drop_remainder):
  """A sliding window over a dataset.

  This transformation passes a sliding window over this dataset. The window size
  is `size`, the stride of the input elements is `stride`, and the shift between
  consecutive windows is `shift`. If the remaining elements cannot fill up the
  sliding window, this transformation will drop the final smaller element.
  For example:

  ```python
  # NOTE: The following examples use `{ ... }` to represent the
  # contents of a dataset.
  a = { [1], [2], [3], [4], [5], [6] }

  a.apply(_sliding_window_batch(size=3)) ==
  { [[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]], [[4], [5], [6]] }

  a.apply(_sliding_window_batch(size=3, shift=2)) ==
  { [[1], [2], [3]], [[3], [4], [5]] }

  a.apply(_sliding_window_batch(size=3, stride=2)) ==
  { [[1], [3], [5]], [[2], [4], [6]] }
  ```

  Args:
    size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      elements in the sliding window. It must be positive.
    shift: A `tf.int64` scalar `tf.Tensor`, representing the
      forward shift of the sliding window in each iteration.
      It must be positive.
    stride: A `tf.int64` scalar `tf.Tensor`, representing the
      stride of the input elements in the sliding window.
      It must be positive.
    drop_remainder: A boolean representing whether the last batch should be
      dropped if its size is smaller than size.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if invalid arguments are provided.
  """
  def _apply_fn(dataset):
    return _SlideDataset(dataset, size, shift, stride, drop_remainder)

  return _apply_fn


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
      positive. If not specified, shift defaults to size.
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
  if not shift:
    shift = size
  if shift == size and stride == 1:
    return dataset.batch(batch_size=size, drop_remainder=drop_remainder)
  try:
    return dataset.apply(
        _sliding_window_batch(
            size=size,
            stride=stride,
            shift=shift,
            drop_remainder=drop_remainder))
  except TypeError:
    # Old version of TF doesn't support drop_remainder parameter.
    windowed = dataset.window(
        size=size, shift=shift, stride=stride, drop_remainder=drop_remainder)
    if isinstance(dataset.element_spec, dict):
      return windowed.flat_map(
          lambda windowed_ds: _windowed_map_to_batch(windowed_ds, size))
    return windowed.flat_map(lambda windowed_ds: windowed_ds.batch(size))
