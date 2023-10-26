# Copyright 2023 Google LLC.
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


def uniform_from_spec(
    spec: tf.TensorSpec, minval: int, maxval: int, seed: int
) -> rlds_types.Step:
  """Builds a tensor of random values with the given spec.

  If the spec has been obtained from a batch of steps where the first
  dimension is `None`, it will create a zero step with a batch dimension of 1.

  Args:
    spec: TensorSpec that specifies the shape and types of the output.
    minval: A Tensor or Python value of type dtype, broadcastable with shape
      (for integer types, broadcasting is not supported, so it needs to be a
      scalar). The lower bound on the range of random values to generate
      (inclusive). Defaults to 0.
    maxval: A Tensor or Python value of type dtype, broadcastable with shape
      (for integer types, broadcasting is not supported, so it needs to be a
      scalar). The upper bound on the range of random values to generate
      (exclusive). Defaults to 1 if dtype is floating point.
    seed: A Python integer. Used in combination with tf.random.set_seed to
      create a reproducible sequence of tensors across multiple calls.

  Returns:
    tensor with `spec` as TensorSpec, and with all the fields randomly
    sampled.
  """

  def uniform(t):
    return tf.random.uniform(
        _zeros_shape(t.shape),
        minval=minval,
        maxval=maxval,
        seed=seed,
        dtype=t.dtype,
    )

  return tf.nest.map_structure(
      uniform,
      spec,
  )


def zero_dataset_like(ds: tf.data.Dataset) -> tf.data.Dataset:
  """Creates a one element dataset with the spec of ds containing zeros.

  Args:
    ds: Dataset of steps.

  Returns:
    Dataset of one element that has the same `element_spec` as `ds`.
  """

  return tf.data.Dataset.from_tensors(zeros_from_spec(ds.element_spec))
