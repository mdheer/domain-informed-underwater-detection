# config.py

from src.tools.enums import (
    DataSplit,
    DataVariant,
    WaterCurrentFilter,
    InputDataPath,
)

# Configuration settings using enums and direct values
input_data_path = InputDataPath.MATHIEU_MERGED
dataset_name = "6_sup_train_ideal-test_ideal"
num_samples_per_fold = 100
k = 10
min_train_samples = 20
step = 100
k_folds_sets = [
    DataSplit.UNLABELLED,
    DataSplit.TRAIN,
    DataSplit.VALIDATE,
    DataSplit.TEST,
]  # Remove the set that has a filter based on the current

configs = {
    "unlabelled_data": {
        "data_split": DataSplit.UNLABELLED,
        "number_of_datapoints": 400,
        "data_variant": DataVariant.IDEAL,
        "filter_based_on_current": None,
    },
    "train_dataset_input": {
        "data_split": DataSplit.TRAIN,
        "number_of_datapoints": 600,
        "data_variant": DataVariant.IDEAL,
        "filter_based_on_current": None,
    },
    "validate_dataset_input": {
        "data_split": DataSplit.VALIDATE,
        "number_of_datapoints": 400,
        "data_variant": DataVariant.IDEAL,
        "filter_based_on_current": None,
    },
    "test_dataset_input": {
        "data_split": DataSplit.TEST,
        "number_of_datapoints": 400,
        "data_variant": DataVariant.IDEAL,
        "filter_based_on_current": None,
    },
}
