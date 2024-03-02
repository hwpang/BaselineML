from sklearn.model_selection import train_test_split

SPLIT_METHODS = ["random", "scaffold", "cluster"]

def split_data(
    datapoints,
    method="random",
    return_inds=False,
    split_sizes=(0.8, 0.1, 0.1),
    seed=0,
    **kwargs,
):
    """
    Split data into training, validation, and test sets.
    """

    match method:

        case "random":

            split_sizes_tmp = list(split_sizes)
            for i, split_size in enumerate(split_sizes):
                if split_size == 0.0:
                    split_sizes_tmp[i] = 1e-10
            split_sizes = split_sizes_tmp

            train_val_inds, test_inds = train_test_split(
                range(len(datapoints)),
                test_size=split_sizes[2],
                random_state=seed,
                **kwargs,
            )
            train_inds, val_inds = train_test_split(
                train_val_inds,
                test_size=split_sizes[1] / (split_sizes[0] + split_sizes[1]),
                random_state=seed,
                **kwargs,
            )

        case "scaffold":

            pass

        case "cluster":

            pass

    if return_inds:
        return train_inds, val_inds, test_inds

    train_mols = [datapoints[i] for i in train_inds]
    val_mols = [datapoints[i] for i in val_inds]
    test_mols = [datapoints[i] for i in test_inds]

    return train_mols, val_mols, test_mols
