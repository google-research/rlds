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

#!/bin/bash
# Designed to work with ./docker/release.dockerfile to build RLDS.

# Exit if any process returns non-zero status.
set -e
set -o pipefail
cd "$(dirname "$0")"

# Flags
PYTHON_VERSIONS=3.8 # Options 3.8 (default)
CLEAN=false # Set to true to run bazel clean.
OUTPUT_DIR=/tmp/rlds/dist/
INSTALL=true # Should the built package be installed.
PYTHON_TESTS=true

PIP_PKG_EXTRA_ARGS="" # Extra args passed to `build_pip_package`.

while [[ $# -gt -0 ]]; do
  key="$1"
  case $key in
      --python)
      PYTHON_VERSIONS="$2" # Python versions to build against.
      shift
      ;;
      --clean)
      CLEAN="$2" # `true` to run bazel clean. False otherwise.
      shift
      ;;
      --install)
      INSTALL="$2" # `true` to install built package. False otherwise.
      shift
      ;;
      --output_dir)
      OUTPUT_DIR="$2"
      shift
      ;;
      --python_tests)
      PYTHON_TESTS="$2"
      shift
      ;;
    *)
      echo "Unknown flag: $key"
      echo "Usage:"
      echo "--python  [3.8(default)]"
      echo "--clean   [true to run bazel clean]"
      echo "--install [true to install built package]"
      echo "--output_dir  [location to copy .whl file.]"
      echo "--python_tests  [true (default) to run python tests.]"
      exit 1
      ;;
  esac
  shift # past argument or value
done

for python_version in $PYTHON_VERSIONS; do

  # Cleans the environment.
  if [ "$CLEAN" = "true" ]; then
    bazel clean
  fi

  if [ "$python_version" = "3.8" ]; then
    export PYTHON_BIN_PATH=/usr/bin/python3.8 && export PYTHON_LIB_PATH=/usr/local/lib/python3.8/dist-packages
    ABI=cp38
  else
    echo "Error unknown --python. Only [3.8]"
    exit 1
  fi

  # Configures Bazel environment for selected Python version.
  $PYTHON_BIN_PATH configure.py

  # Build RLDS and run all bazel Python tests.
  bazel test -c opt --copt=-mavx --test_output=errors //...

  # Builds RLDS and creates the wheel package.
  bazel build -c opt pip_package:build_pip_package
  ./bazel-bin/pip_package/build_pip_package --dst $OUTPUT_DIR/fresh $PIP_PKG_EXTRA_ARGS

  echo "Wheel package created."
  # Install built package.
  if [ "$INSTALL" = "true" ]; then
    echo "Installing"
    $PYTHON_BIN_PATH -mpip install --upgrade $OUTPUT_DIR/fresh/*
  fi

  chmod 666 $OUTPUT_DIR/fresh/*
  mv $OUTPUT_DIR/fresh/* $OUTPUT_DIR/
  rm -r $OUTPUT_DIR/fresh

  if [ "$PYTHON_TESTS" = "true" ]; then
    echo "Run Python tests..."
    set +e

    bash run_python_tests.sh |& tee ./unittest_log.txt
    UNIT_TEST_ERROR_CODE=$?
    set -e
    if [[ $UNIT_TEST_ERROR_CODE != 0 ]]; then
      echo -e "\n\n\n===========Error Summary============"
      grep -E 'ERROR:|FAIL:' ./unittest_log.txt
      exit $UNIT_TEST_ERROR_CODE
    else
      echo "Python tests successful!!!"
    fi
  fi

done
