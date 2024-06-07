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
"""Tests for generate_config."""
import dataclasses
import os

from absl import flags
from absl.testing import absltest
import numpy as np
from rlds.tfds import config_generator
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


class ConfigGeneratorTest(absltest.TestCase):

  def _fake_image(self):
    empty_array = tf.zeros((100, 100, 4), 'uint8')
    return tf.io.encode_png(empty_array)


  def test_extract_feature_image(self):
    data = {'Image': tf.zeros(shape=(2, 2, 1), dtype=np.uint8)}
    data_type = config_generator.extract_feature_from_data(
        data, use_images=True, image_encoding='png')
    self.assertEqual(type(data_type['Image']), tfds.features.Image)

  def test_extract_feature_tensor(self):
    data = {'Image': tf.zeros(shape=(2, 2, 1), dtype=np.uint8)}
    data_type = config_generator.extract_feature_from_data(
        data, use_images=False, image_encoding='png')
    self.assertNotEqual(type(data_type['Image']), tfds.features.Image)

  def test_extract_feature_scalar(self):
    data = {'Image': tf.cast(1, tf.uint8)}
    data_type = config_generator.extract_feature_from_data(
        data, use_images=True, image_encoding='png')
    self.assertNotEqual(type(data_type['Image']), tfds.features.Image)

  def test_extract_feature_shape_tensor(self):
    data = {'Image': tf.zeros(shape=(5, 4), dtype=np.uint8)}
    data_type = config_generator.extract_feature_from_data(
        data, use_images=True, image_encoding='png')
    self.assertNotEqual(type(data_type['Image']), tfds.features.Image)

  def test_extract_feature_string_image(self):
    data = {'Image': self._fake_image()}
    data_type = config_generator.extract_feature_from_data(
        data, use_images=True, image_encoding='png')
    self.assertEqual(type(data_type['Image']), tfds.features.Image)

  def test_extract_feature_string_scalar(self):
    data = {'Image': self._fake_image()}
    data_type = config_generator.extract_feature_from_data(
        data, use_images=False, image_encoding='png')
    self.assertNotEqual(type(data_type['Image']), tfds.features.Image)

  def test_extract_feature_string_name_scalar(self):
    data = {'Field': self._fake_image()}
    data_type = config_generator.extract_feature_from_data(
        data, use_images=False, image_encoding='png')
    self.assertNotEqual(type(data_type['Field']), tfds.features.Image)

  def test_extract_feature_fails_with_tuples(self):
    data = {'Tuple': (tf.constant(1), tf.constant(2), tf.constant(3))}
    with self.assertRaises(ValueError):
      _ = config_generator.extract_feature_from_data(
          data, use_images=False, image_encoding='png')

  def test_extract_feature_encodes_list_as_sequence(self):
    data = {'List': [tf.constant(1), tf.constant(2), tf.constant(3)]}
    data_type = config_generator.extract_feature_from_data(
        data, use_images=False, image_encoding='png')
    self.assertEqual(type(data_type['List']), tfds.features.Sequence)
    self.assertEqual(type(data_type['List'].feature), tfds.features.Tensor)

  def test_extract_feature_empty_list_fails(self):
    data = {'List': []}
    with self.assertRaises(ValueError):
      _ = config_generator.extract_feature_from_data(
          data, use_images=False, image_encoding='png')

  def test_extract_feature_numpy(self):
    data = {'Field': np.zeros((2, 2))}
    data_type = config_generator.extract_feature_from_data(
        data, use_images=True, image_encoding='png')
    self.assertEqual(type(data_type['Field']), tfds.features.Tensor)
    self.assertEqual(data_type['Field'].shape, (2, 2))
    self.assertEqual(data_type['Field'].dtype, np.float64)

  def test_extract_feature_numpy_squeeze_scalar(self):
    data = {'Field': np.ndarray(1)}
    print(data['Field'].shape)
    data_type = config_generator.extract_feature_from_data(
        data, use_images=True, image_encoding='png', squeeze_scalars=True)
    self.assertEqual(type(data_type['Field']), tfds.features.Tensor)
    self.assertEqual(data_type['Field'].shape, ())
    self.assertEqual(data_type['Field'].dtype, np.float64)

  def test_extract_feature_numpy_scalar(self):
    data = {'Field': np.ndarray(1)}
    print(data['Field'].shape)
    data_type = config_generator.extract_feature_from_data(
        data, use_images=True, image_encoding='png', squeeze_scalars=False)
    self.assertEqual(type(data_type['Field']), tfds.features.Tensor)
    self.assertEqual(data_type['Field'].shape, (1,))
    self.assertEqual(data_type['Field'].dtype, np.float64)


if __name__ == '__main__':
  absltest.main()
