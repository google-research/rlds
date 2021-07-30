# RL Datasets

RL Datasets is a system to store, retrieve and filter RL trajectories in order
to use them in RL workflows.

### Dataset Format{#dataset_format}

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

### Using TFDS to load a dataset {#tfds-load}


#### Datasets in the TFDS catalog

These datasets can be loaded directly with:

```py
tfds.load('dataset_name').as_dataset()['train']
```

See the full documentation and the catalog in https://www.tensorflow.org/datasets.

## How to add your dataset to TFDS

### Using the TFDS catalog

You can add your dataset directly to TFDS
following the instructions at https://www.tensorflow.org/datasets ([example](http://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/rl_unplugged/rlu_atari)).

## Acknowledgements

We greatly appreciate all the support from the
[TF-Agents](https://github.com/tensorflow/agents) team in setting up building
and testing for EnvLogger.

## Disclaimer

This is not an officially supported Google product.
