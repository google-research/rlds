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
"""TFDS episode writer."""
from typing import Optional

from absl import logging
from rlds import rlds_types
import tensorflow_datasets as tfds


DatasetConfig = tfds.rlds.rlds_base.DatasetConfig


class EpisodeWriter():
  """Class that writes trajectory data in TFDS format (and RLDS structure)."""

  def __init__(self,
               data_directory: str,
               ds_config: DatasetConfig,
               max_episodes_per_file: int = 1000,
               split_name: Optional[str] = 'train',
               version: str = '0.0.1',
               overwrite: bool = True):
    """Constructor.

    Args:
      data_directory: Directory to store the data
      ds_config: Dataset Configuration.
      max_episodes_per_file: Number of episodes to store per shard.
      split_name: Name to be used by the split. If None, the name of the parent
        directory will be used.
      version: version (major.minor.patch) of the dataset.
      overwrite: if False, and there is an existing dataset, it will append to
        it.
    """

    self._data_directory = data_directory
    ds_identity = tfds.core.dataset_info.DatasetIdentity(
        name=ds_config.name,
        version=tfds.core.Version(version),
        data_dir=data_directory,
        module_name='')
    self._ds_info = tfds.rlds.rlds_base.build_info(ds_config, ds_identity)
    self._ds_info.set_file_format('tfrecord')

    self._sequential_writer = tfds.core.SequentialWriter(
        self._ds_info, max_episodes_per_file, overwrite=overwrite)
    self._split_name = split_name
    self._sequential_writer.initialize_splits([split_name],
                                              fail_if_exists=overwrite)
    logging.info('Creating dataset in: %r', self._data_directory)

  def add_episode(self, episode: rlds_types.Episode) -> None:
    """Adds the episode to the dataset.

    Args:
      episode: episode to add to the dataset.
    """
    self._sequential_writer.add_examples({self._split_name: [episode]})

  def close(self) -> None:
    self._sequential_writer.close_all()
