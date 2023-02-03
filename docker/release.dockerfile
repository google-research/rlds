ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image
FROM $base_image


# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image
# TODO(sabela): support other python versions.
ARG python_version="python3.8"
ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"

# Otherwise it gets stuck asking for a timezone
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata


RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y --no-install-recommends \
        software-properties-common \
        aria2 \
        build-essential \
        curl \
        git \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        python3-dev \
        python3.8-dev \
        # python >= 3.8 needs distutils for packaging.
        python3.8-distutils \
        rename \
        rsync \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Use get-pip to install pip, setuptools and wheel.
RUN curl -O https://bootstrap.pypa.io/get-pip.py


ARG bazel_version=3.7.2
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Add support for Bazel autocomplete, see
# https://docs.bazel.build/versions/master/completion.html for instructions.
RUN cp /usr/local/lib/bazel/bin/bazel-complete.bash /etc/bash_completion.d


ARG pip_dependencies=' \
      contextlib2 \
      dm-tree>=0.1.5 \
      dataclasses \
      google-api-python-client \
      h5py \
      numpy \
      oauth2client \
      pandas \
      portpicker \
      dm-reverb-nightly \
      tensorflow-datasets'

ARG tensorflow_pip='tf-nightly'

# So dependencies are installed for the supported Python versions
RUN for python in ${python_version}; do \
    $python get-pip.py && \
    $python -mpip uninstall -y tensorflow tensorflow-gpu tf-nightly tf-nightly-gpu && \
    $python -mpip --no-cache-dir install ${tensorflow_pip} --upgrade && \
    $python -mpip --no-cache-dir install $pip_dependencies; \
  done

RUN rm get-pip.py

# bazel assumes the python executable is "python".
RUN ln -s /usr/bin/python3 /usr/bin/python


ADD . /rlds/
WORKDIR /rlds

CMD ["/bin/bash"]
