# RLDS

RLDS stands for Reinforcement Learning Datasets and it is an ecosystem of tools
to store, retrieve, manipulate episodic data in in the context of Sequential
Decision Making including Reinforcement Learning (RL), Learning for
Demonstrations, Offline RL or Imitation Learning.

This repo includes a library for manipulating RLDS compliant datasets. Please
refer to:

*   [EnvLogger](http://github.com/deepmind/envlogger) to create synthetic datasets
*   [RLDS Creator](http://github.com/google-research/rlds-creator) to create datasets where a human interacts with an
    environment.
*   [TFDS](http://www.tensorflow.org/datasets/catalog/overview) for existing RL datasets.

### QuickStart

See how to use RLDS in this [tutorial].

[tutorial]: http://github.com/google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb

### Dataset Format

The dataset is retrieved as a `tf.data.Dataset` of Episodes.

*   **Episode**: dictionary that contains a `tf.data.Dataset` of Steps, and
    metadata.

*   **Step**: dictionary that contains:

    *   `observation`: current observation
    *   `action`: action taken in the current observation
    *   `reward`: return after appyling the action to the current observation
    *   `is_terminal`: if this is a terminal step
    *   `is_first`: if this is the first step of an episode that contains the
        initial state.
    *   `is_last`: if this is the last step of an episode, that contains the
        last observation. When true, `action`, `reward` and `discount`, and
        other cutom fields subsequent to the observation are considered invalid.
    *   `discount`: discount factor at this step.
    *   extra metadata

    When `is_terminal = True`, the `observation` corresponds to a final state,
    so `reward`, `discount` and `action` are meaningless. Depending on the
    environment, the final `observation` may also be meaningless.

    If an episode ends in a step where `is_terminal = False`, it means that this
    episode has been truncated. In this case, depending on the environment, the
    action, reward and discount might be empty as well.

You can then apply regular `tf.data.Dataset` transformations to this format.

## How to load your dataset

#### Datasets in the TFDS catalog

These datasets can be loaded directly with:

```py
tfds.load('dataset_name').as_dataset()['train']
```

See the full documentation and the catalog in https://www.tensorflow.org/datasets.

#### Datasets in your own repository

Datasets can be implemented with TFDS both inside and outside of the TFDS
repository. See examples about how to load them
[here](https://www.tensorflow.org/datasets/external_tfrecord?hl=en#load_dataset_with_tfds).

## How to add your dataset to TFDS

### Using the TFDS catalog

You can add your dataset directly to TFDS
following the instructions at https://www.tensorflow.org/datasets.

* If your data has been generated with Envlogger or the RLDS Creator, you can just use the rlds helpers in TFDS (see [here](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/rlds/robosuite_panda_pick_place_can/robosuite_panda_pick_place_can.py) an example).
* Otherwise, make sure your `generate_examples` implementation provides the same structure
  and keys as RLDS loaders if you want your dataset to be compatible with RLDS
  pipelines
  ([example](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/d4rl/dataset_utils.py)).


Note that you can follow the same steps to add the data to your own repository
(see more details in the [TFDS documentation](https://www.tensorflow.org/datasets/add_dataset?hl=en)).

## Acknowledgements

We greatly appreciate all the support from the
[TF-Agents](https://github.com/tensorflow/agents) team in setting up building
and testing for EnvLogger.

## Disclaimer

This is not an officially supported Google product.
