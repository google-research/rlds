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
"""Library to generate a TFDS config."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from rlds import rlds_types
import tensorflow as tf
import tensorflow_datasets as tfds

_STEP_KEYS = [
    rlds_types.OBSERVATION, rlds_types.ACTION, rlds_types.DISCOUNT,
    rlds_types.REWARD, rlds_types.IS_LAST, rlds_types.IS_FIRST,
    rlds_types.IS_TERMINAL
]


def _is_image(data: tf.Tensor, field_name: Optional[str],
              image_encoding: str) -> bool:
  """Checks if data corresponds to an image."""
  if not field_name or not('image' in field_name or 'Image' in field_name):
    return False
  try:
    _ = tfds.features.image_feature.get_and_validate_encoding(image_encoding)
  except ValueError:
    return False
  if data.shape == tf.TensorShape([]):
    # Scalars are only considered images if they are encoded as strings.
    if data.dtype is tf.string or data.dtype is str:
      return True
    return False
  if len(data.shape) > 4:
    # GIF images have 4 dims, the rest have 3.
    return False
  try:
    _ = tfds.features.image_feature.get_and_validate_shape(
        data.shape, image_encoding)
    _ = tfds.features.image_feature.get_and_validate_dtype(
        data.dtype, image_encoding)
  except ValueError:
    return False
  # Extra check for float32 images
  if data.shape[-1] != 1:
    return False
  return True


def _is_scalar(data: Union[tf.Tensor, np.ndarray],
               squeeze: bool = True) -> bool:
  """Checks if the data is a scalar.

  Args:
    data: data to check.
    squeeze: if True, considers shape (1,) as a scalar.

  Returns:
    True if data is a scalar.
  """
  if not hasattr(data, 'shape'):
    return True
  # Some numpy arrays will still be treated as Tensors of shape=(). It's not
  # very relevant because that's how TFDS treats all scalars internally anyway.
  if squeeze:
    return data.shape == (1,) or data.shape == tf.TensorShape(
        []) or not data.shape
  else:
    # Note that numpy arrays with one element have shape (1,),so they will
    # be treated as Tensors of shape (1,)
    return data.shape == tf.TensorShape([]) or not data.shape


def extract_feature_from_data(
    data: Union[Dict[str, Any], List[Any], Union[tf.Tensor, np.ndarray, Any]],
    use_images: bool,
    image_encoding: Optional[str],
    field_name: Optional[str] = None,
    squeeze_scalars: bool = True,
    convert_tuple_to_list: bool = False,
    tensor_encoding=tfds.features.Encoding.ZLIB,
) -> Union[Dict[str, Any], tfds.features.FeatureConnector]:
  """Returns the data type of providing data.

  Args:
    data: supports data of the following types: nested dictionary/union/list of
      tf.Tensor, np.Arrays, scalars or types that have 'shape' and 'dtype' args.
      Lists have to contain uniform elements.
    use_images: if True, encodes uint8 tensors and string scalars with a field
      name that includes `image` or `Image` as images.
    image_encoding: if `use_images`, uses this encoding for the detected images.
    field_name: if present, is used to decide if data of tf.string type should
      be encoded as an image.
    squeeze_scalars: if True, it will treat arrays of shape (1,) as
      `tfds.features.Scalar`.
    convert_tuple_to_list: if True, converts tuple input data to a list, since
      tuples are not supported in TFDS.
    tensor_encoding: Internal encoding of a `tfds.features.Tensor`.

  Returns:
    the same nested data structure with the data expressed as TFDS Features.

  Raises:
    ValueError for data that is not supported by TFDS.
  """
  if isinstance(data, dict):
    return tfds.features.FeaturesDict({
        k: extract_feature_from_data(data[k], use_images, image_encoding, k,
                                     squeeze_scalars, convert_tuple_to_list)
        for k in data
    })
  elif isinstance(data, tuple):
    if not convert_tuple_to_list:
      raise ValueError('Tuples are not supported in TFDS. '
                       'Use dictionaries or lists instead.')
    if not data:
      raise ValueError('Trying to extract the type of an empty tuple.')
    # Elements of a list are expected to have the same types. We don't check all
    # the elements of the list the same way that we don't check all the steps in
    # an episode.
    feature = extract_feature_from_data(data[0], use_images, image_encoding,
                                        field_name, squeeze_scalars,
                                        convert_tuple_to_list,
                                        tensor_encoding=tensor_encoding)
    return tfds.features.Sequence(feature=feature)
  elif isinstance(data, list):
    if not data:
      raise ValueError('Trying to extract the type of an empty list.')
    # Elements of a list are expected to have the same types. We don't check all
    # the elements of the list the same way that we don't check all the steps in
    # an episode.
    feature = extract_feature_from_data(data[0], use_images, image_encoding,
                                        field_name, squeeze_scalars,
                                        convert_tuple_to_list,
                                        tensor_encoding=tensor_encoding)
    return tfds.features.Sequence(feature=feature)
  elif use_images and _is_image(data, field_name, image_encoding):
    if not image_encoding:
      raise ValueError('Image encoding is not defined.')
    if _is_scalar(data, squeeze_scalars):
      return tfds.features.Image(encoding_format=image_encoding)
    else:
      return tfds.features.Image(
          shape=data.shape,
          dtype=tf.as_dtype(data.dtype),
          encoding_format=image_encoding)
    return data.dtype
  elif _is_scalar(data, squeeze_scalars):
    return tf.as_dtype(data.dtype)
  elif tfds.core.utils.dtype_utils.is_string(data.dtype):
    return tfds.features.Text()
  else:
    return tfds.features.Tensor(
        shape=data.shape,
        dtype=tf.as_dtype(data.dtype),
        encoding=tensor_encoding)




def generate_config_from_spec(
    episode_spec: tf.TensorSpec,
    name: str = 'default_config',
    use_images: bool = True,
    image_encoding: str = 'png') -> tfds.rlds.rlds_base.DatasetConfig:
  """Generates a config for a dataset.

  Args:
    episode_spec: RLDS episode spec in terms of tensor specs. This can be an
      environment spec or a tf.data.Dataset.element_spec.
    name: name of the config to generate
    use_images: if True (default), encodes uint8 tensors and string scalars with
      a field name that includes `image` or `Image` as images.
    image_encoding: if `use_images`, uses this encoding for the detected images.
      Defaults to `png`. See `tfds.features.Image` for valid values for this
      argument.

  Returns:
    a dictionary contains a config for the dataset.
  """

  episode_metadata = {
      k: extract_feature_from_data(episode_spec[k], use_images, image_encoding,
                                   k)
      for k in episode_spec
      if k != rlds_types.STEPS
  }


  step_spec = episode_spec[rlds_types.STEPS].element_spec
  step_metadata = {
      k: extract_feature_from_data(step_spec[k], use_images, image_encoding, k)
      for k in step_spec
      if k not in _STEP_KEYS
  }

  if rlds_types.DISCOUNT in step_spec:
    discount_info = extract_feature_from_data(step_spec[rlds_types.DISCOUNT],
                                              use_images, image_encoding)
  else:
    discount_info = None

  # pytype: disable=wrong-keyword-args
  return tfds.rlds.rlds_base.DatasetConfig(
      name=name,
      observation_info=extract_feature_from_data(
          step_spec[rlds_types.OBSERVATION], use_images, image_encoding),
      action_info=extract_feature_from_data(step_spec[rlds_types.ACTION],
                                            use_images, image_encoding),
      reward_info=extract_feature_from_data(step_spec[rlds_types.REWARD],
                                            use_images, image_encoding),
      discount_info=discount_info,
      episode_metadata_info=episode_metadata,
      step_metadata_info=step_metadata,
  )
  # pytype: enable=wrong-keyword-args
