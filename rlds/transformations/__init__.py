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
"""rlds.transformations."""

from rlds.transformations.alignment import add_alignment_to_step
from rlds.transformations.alignment import AlignmentType
from rlds.transformations.alignment import shift_keys

from rlds.transformations.dataset_concat import concat_if_terminal
from rlds.transformations.dataset_concat import concatenate

from rlds.transformations.dataset_stats import mean_and_std
from rlds.transformations.dataset_stats import sar_fields_mask

from rlds.transformations.dataset_truncate import truncate_after_condition

from rlds.transformations.flexible_batch import batch
from rlds.transformations.flexible_batch import BATCH_AUTO_TUNE

from rlds.transformations.nested_ops import apply_nested_steps
from rlds.transformations.nested_ops import episode_length
from rlds.transformations.nested_ops import final_step
from rlds.transformations.nested_ops import map_nested_steps
from rlds.transformations.nested_ops import map_steps
from rlds.transformations.nested_ops import sum_dataset
from rlds.transformations.nested_ops import sum_nested_steps

from rlds.transformations.pattern import pattern_map
from rlds.transformations.pattern import pattern_map_from_transform
from rlds.transformations.pattern import step_spec

from rlds.transformations.shape_ops import uniform_from_spec
from rlds.transformations.shape_ops import zero_dataset_like
from rlds.transformations.shape_ops import zeros_from_spec
