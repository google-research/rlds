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

#!/bin/bash
set -e

function build_wheel() {
  TMPDIR="$1"
  DESTDIR="$2"

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  if [[ -e python_bin_path.sh ]]; then
    echo $(date)  "Setting PYTHON_BIN_PATH equal to what was set with configure.py."
    source python_bin_path.sh
  fi
  PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-$(which python3)}

  pushd ${TMPDIR} > /dev/null

  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH}" setup.py bdist_wheel ${PKG_NAME_FLAG} --plat manylinux2010_x86_64 > /dev/null
  DEST=${TMPDIR}/dist/
  if [[ ! "$TMPDIR" -ef "$DESTDIR" ]]; then
    mkdir -p ${DESTDIR}
    cp dist/* ${DESTDIR}
    DEST=${DESTDIR}
  fi
  popd > /dev/null
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

function prepare_src() {
  TMPDIR="${1%/}"
  mkdir -p "$TMPDIR"

  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"

  if [ ! -d bazel-bin/rlds ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  cp LICENSE ${TMPDIR}

  # Copy all Python files.
  cp --parents `find -name \*.py*` ${TMPDIR}
  mv ${TMPDIR}/pip_package/setup.py ${TMPDIR}
  cp ./pip_package/MANIFEST.in ${TMPDIR}
  mv ${TMPDIR}/pip_package/rlds_version.py ${TMPDIR}
}

function usage() {
  echo "Usage:"
  echo "$0 [options]"
  echo "  Options:"
  echo "    --dst  path to copy the .whl into."
  echo ""
  exit 1
}

function main() {
  # This is where the source code is copied and where the whl will be built.
  DST_DIR=""

  while true; do
    if [[ "$1" == "--help" ]]; then
      usage
      exit 1
    elif [[ "$1" == "--dst" ]]; then
      shift
      DST_DIR=$1
    fi

    if [[ -z "$1" ]]; then
      break
    fi
    shift
  done

  TMPDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
  if [[ -z "$DST_DIR" ]]; then
    DST_DIR=${TMPDIR}
  fi

  prepare_src "$TMPDIR"
  build_wheel "$TMPDIR" "$DST_DIR"
}

main "$@"
