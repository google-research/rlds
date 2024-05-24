# Add your dataset to TFDS

## Scenarios

### My data is neither in RLDS nor in TFDS format (general case)

In the general case, adding a dataset to TFDS involves two steps:

1.  Implement a python class that provides a dataset builder with the specs of
    the data (e.g., what is the shape of the observations, actions, etc.) and
    how to read your dataset files.

    NOTE: If you want your dataset to be compatible with RLDS pipelines, make
    sure your implementation provides the same structure and keys as an RLDS
    dataset.

2.  Run a `download_and_prepare` pipeline that converts the data to the TFDS
    intermediate format.

You can follow the [instructions to add a dataset] in the TFDS site.

There are situations in which it might be preferrable to re-write the raw data
into an RLDS/TFDS compatible format before adding it to TFDS (for example if
your data uses a format that cannot be shared). You can use the [Envlogger] or
the [EpisodeWriter] directly to do so. To use the EpisodeWriter, you can create
your DatasetConfig with the [ConfigGenerator] tool.

[instructions to add a dataset]: https://www.tensorflow.org/datasets/add_dataset?hl=en
[EpisodeWiter]: https://github.com/google-research/rlds/blob/main/rlds/tfds/episode_writer.py
[Envlogger]:https://github.com/google-research/rlds#how-to-create-a-dataset
[ConfigGenerator]: https://github.com/google-research/rlds/blob/main/rlds/tfds/config_generator.py

### My RLDS dataset is already in TFDS format

Even if your data is already in TFDS format, you may want to create a TFDS
builder if you want to:

*   **Reshuffle**: When you want to re-generate the data to ensure that episodes
    are shuffled on disk (otherwise, they are stored as they were generated with
    Envlogger).
*   **Share**: Consider if you want to add the dataset to the TFDS catalog or if
    you just want to share it in your own repository (note that users will still
    be able to load your data directly with `tfds.builder_from_directory`).

Most of the steps to follow in this case will be either no-ops or very simple.
You can find an example
[here](https://www.tensorflow.org/datasets/catalog/mt_opt).

### My RLDS dataset is not in TFDS format

If you have generated your RLDS dataset with the Envlogger Riegeli backend and
you want to convert it to TFDS, you can take a look at the instructions to
create a TFDS builder using the RLDS helpers.

## Creating a TFDS builder

You can add your dataset directly to TFDS
following the instructions at https://www.tensorflow.org/datasets.

* If your data has been generated with Envlogger or the RLDS Creator, you can just use the rlds helpers in TFDS (see [here](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/rlds/datasets/robosuite_panda_pick_place_can/robosuite_panda_pick_place_can.py) an example, or
    [here](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/robotics/mt_opt/mt_opy.py)
    if you used the TFDS Envlogger backend).
* Otherwise, make sure your `generate_examples` implementation provides the same structure
  and keys as RLDS loaders if you want your dataset to be compatible with RLDS
  pipelines
  ([example](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/d4rl/dataset_utils.py)).


Note that you can follow the same steps to add the data to your own repository
(see more details in the [TFDS documentation](https://www.tensorflow.org/datasets/add_dataset?hl=en)).
