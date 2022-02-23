# Transformations

RLDS provides a library of tranformations to efficiently manipulate the datasets
of steps and episodes. These transformations use optimizations based on the best
practices described in
[this colab](https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_performance.ipynb).

See usage examples in the
[tutorial](https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb)
and in the [examples page](docs/examples.md).

This document only includes an overview of a set of the available functions, see
the full documentation of the API in the code.

## Batch

*   `rlds.transformations.batch`: batches a dataset of steps using the
    `tf.data.Dataset` window interface (i.e., it allows to batch with overlap).
    It is particularly useful to generate a dataset of transitions or
    overlapping trajectories. See an example
    [here](https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb#scrollTo=TGT3YfzFOrBm).

## Manipulating nested datasets

*   `rlds.transformations.episode_length`: obtains the lenght of an episode
    (i.e., a dataset of steps).

*   `rlds.transformations.sum_dataset`: accumulates the values of a dataset of
    steps. It can be used to gather the accumulated reward of each episode
    ([example](https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb#scrollTo=nblDjpJ6M1-u)).

*   `rlds.transformations.sum_nested_steps`: accumulates the required values
    accross all steps in a dataset of episodes.

*   `rlds.transformations.final_step`: returns the final step of a dataset of
    steps.

*   `rlds.transformations.map_nested_steps`: Applies a transformation to all the
    steps of a dataset of episodes. The transformation is applied to each step
    individually.

*   `rlds.transformations.apply_nested_steps`: Applies a transformation to all
    the steps dataset of a dataset of episodes. The transformation is applied to
    each dataset of steps (not to each step separately).

## Concatenation

*   `rlds.transformations.concatenate`: concatenatest two datasets of steps. If
    the steps contain different fields, the missing fields are added and
    initialized to zeros.

*   `rlds.transformations.concat_if_terminal`: concats two datases of steps only
    if the first one ends ina terminal step. It is used in the
    [examples](https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_examples.ipynb#scrollTo=pWNhxwJzOUJv)
    to add absorbing states to an episode.

## Dataset stats

*   `rlds.transformations.mean_and_std`: calculates the mean and the standard
    deviation across a dataset of episodes for the given fields. It is used in
    the
    [examples](https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb#scrollTo=Z0TITfo_4oZr)
    to apply normalization.

*   `rlds.transformations.sar_field_mask`: can be used as a parameter to
    `mean_and_std` to obtain the mean and standard deviation of the default sar
    fields (observation, action, reward)

## Truncation

*   `rlds.transformations.truncate_after_condition`: truncates a dataset of
    steps after the first step that satisfies a condition.

## Alignment

*   `rlds.transformations.shift_keys`: shifts elements of the step. For example,
    to change from SAR to RSA alignment. It is applied to a steps dataset.

*   `rlds.transformations.add_alingment_to_step`: adds an extra field to the
    step indicating what is the alignment.

## Zero initialization

*   `rlds.transformations.zeros_from_spec`: Builds a tensor of zeros with the
    given spec. If the spec was obtained from a batch of steps where the first
    dimension is `None`, it creates a zero step with a batch dimension of 1.

*   `rlds.transformations.zero_dataset_like`: Creates a dataset of one element
    with the same spec as the given dataset and initialized to zeros.
