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
"""Functionality related to the re-alignment of step fields."""

import enum
from typing import List

from rlds import rlds_types
from rlds.transformations import batched_helpers
from rlds.transformations import flexible_batch
import tensorflow.compat.v2 as tf


def shift_keys(
    steps: tf.data.Dataset,
    keys: List[str],
    shift: int,
    batch_size: int = flexible_batch.BATCH_AUTO_TUNE) -> tf.data.Dataset:
  """Shifts `key` from every element in `keys` with `shift` positions.

  If step_i has keys A_i and B_i and we shift A -1 positions, step_i will
  contain A_{i-1} and B_i. B_0 will be discarded, as well as A_n (where n
  is the last step)

  Args:
    steps: `tf.data.Dataset` of steps.
    keys: list of step keys that have to be shifted.
    shift: number of positions to shift. Note that the shift is expressed as
      an index. So, in step_i, the indicated keys will be now correspond
      to the values of step_{i+shift}.
    batch_size: batch size to use when applying the transformation. It should be
      larger than `shift`. The default value (BATCH_AUTO_TUNE) makes batch size
      selection automatic.

  Returns:
    `tf.data.Dataset` of steps where the elements with key in `keys` are
    shifted. The resulting dataset is of size `steps.cardinality() - shift`
  """
  if batch_size == flexible_batch.BATCH_AUTO_TUNE:
    batch_size = max(4*shift, flexible_batch.get_batch_size(steps, batch_size))

  def shift_field(field, shift):
    if shift > 0:
      return tf.nest.map_structure(lambda x: x[shift:], field)
    else:
      return tf.nest.map_structure(lambda x: x[:shift], field)

  def realign_batch(batch):
    result = {}
    for k in batch:
      if k not in keys:
        result[k] = shift_field(batch[k], -shift)
      else:
        result[k] = shift_field(batch[k], shift)
    return result

  if abs(shift) >= batch_size:
    raise ValueError(f'Trying to shift {shift} positions using a batch size '
                     'of {batch_size} (abs(shift) should be smaller than the '
                     'batch size).')
  batch_shift = batch_size - abs(shift)

  return batched_helpers.batched_map(
      steps, map_fn=realign_batch, size=batch_size, shift=batch_shift)


@enum.unique
class AlignmentType(enum.Enum):
  UNKNOWN = enum.auto()
  SAR = enum.auto()
  ARS = enum.auto()
  RSA = enum.auto()


def add_alignment_to_step(step: rlds_types.Step, alignment: AlignmentType):
  step[rlds_types.ALIGNMENT] = alignment.value
  return step
