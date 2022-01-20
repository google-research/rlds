# Copyright 2022 Google LLC.
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
"""Library functions to operate on dataset shapes."""

import sys
from typing import Any, Tuple

from rlds import rlds_types
import tensorflow as tf


def _zeros_shape(element_shape: tf.TensorShape) -> Tuple[Any, ...]:
  if not element_shape:
    return ()
  return tuple([1 if dim is None else dim for dim in element_shape])


def size_from_spec(spec: tf.TensorSpec) -> int:
  """Computes size in bytes of an element with `spec` structure.

  Args:
    spec: TensorSpec that specifies the shape and types of the element.

  Returns:
    Size of the element represented by `spec`. If the size can not be
    determined, returns `sys.maxsize`.
  """
  try:
    return sum(
        tf.nest.flatten(
            tf.nest.map_structure(
                lambda t: t.dtype.size * t.shape.num_elements(),
                spec).values()))
  except TypeError:
    return sys.maxsize


def zeros_from_spec(spec: tf.TensorSpec) -> rlds_types.Step:
  """Builds a tensor of zeros with the given spec.

  If the spec has been obtained from a batch of steps where the first
  dimension is `None`, it will create a zero step with a batch dimension of 1.

  Args:
    spec: TensorSpec that specifies the shape and types of the output.

  Returns:
    tensor with `spec` as TensorSpec, and with all the fields initialized to
    zeros.
  """
  return tf.nest.map_structure(
      lambda t: tf.zeros(_zeros_shape(t.shape), t.dtype), spec)


def zero_dataset_like(ds: tf.data.Dataset) -> tf.data.Dataset:
  """Creates a one element dataset with the spec of ds containing zeros.

  Args:
    ds: Dataset of steps.

  Returns:
    Dataset of one element that has the same `element_spec` as `ds`.
  """

  return tf.data.Dataset.from_tensors(zeros_from_spec(ds.element_spec))
