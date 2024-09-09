from enum import Enum


class HomePath(Enum):
    MATHIEU_LOCAL = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu"
    MATHIEU_CLUSTER = r"/home/mdheer/thesis_mathieu"


class ClassName(Enum):
    """
    Enum for classifying objects in the dataset.

    Attributes:
        PLASTIC: Represents plastic objects.
        FISH: Represents fish objects.
    """

    PLASTIC = "plastic"
    FISH = "fish"


class DataSplit(Enum):
    """
    Enum for different data splits used in training and evaluation.

    Attributes:
        UNLABELLED: Represents the unlabelled training set
        TRAIN: Represents the training dataset.
        VALIDATE: Represents the validation dataset.
        TEST: Represents the test dataset.
    """

    UNLABELLED = "unlabelled"
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"


class OpticalFlowAccuracy(Enum):
    LOW = "low_optical_flow_accuracy"
    HIGH = "high_optical_flow_accuracy"


class InvertedSimilarity(Enum):
    """
    Enum for types of inverted similarity measures.

    Attributes:
        AVERAGE_MSE: Represents the average mean squared error.
        WEIGHTED_MSE: Represents a weighted mean squared error.
    """

    AVERAGE_MSE = "default_mse"
    WEIGHTED_MSE = "weighted_mse"
    WEIGHTED_MAD = "weighted_mad"
    WEIGHTED_HAUSDORFF = "weighted_hausdorff"
    WEIGHTED_DTW = "weighted_dtw"
    WEIGHTED_DFD = "weighted_dfd"
    WEIGHTED_VARIANCE = "weighted_variance"


class Similarity(Enum):
    """
    Enum for different types of similarity measures.

    Attributes:
        PEARSON_CORR: Represents a weighted Pearson correlation.
        SPEARMAN_CORR: Represents a weighted Spearman correlation.
    """

    PEARSON_CORR = "weighted_pearson_correlation"
    SPEARMAN_CORR = "weighted_spearman_correlation"


class WaterCurrentFilter(Enum):
    """
    Enum for filtering the data

    Attributes:
        LIGHT_CURRENT: Represents trajectories with a light current
        STRONG_CURRENT: Represents trajectories with a strong current
    """

    LIGHT_CURRENT = "low_current"
    STRONG_CURRENT = "strong_current"


class InputDataPath(Enum):
    """
    Enum for specifying different data paths.

    Attributes:
        MATHIEU_LOCAL: Data path specific to Mathieu's local setup.
        CLUSTER: Data path for cluster setup.
        ATHINA_LOCAL: Data path specific to Athina's local setup.
        DEFAULT: Represents a default data path.
        MATHIEU_24_07: Data path for Mathieu's setup as of 24/10.
    """

    MATHIEU_LOCAL = "mathieu_data_path"
    CLUSTER = "cluster_data_path"
    ATHINA_LOCAL = "athina_data_path"
    DEFAULT = "default"
    MATHIEU_24_07 = "mathieu_data_path_24/10"
    CLUSTER_11_02 = "cluster_data_path_11/02"
    CLUSTER_MERGED = "cluster_merged_data_path"
    MATHIEU_MERGED = "mathieu_merged"


class ParameterEstimation(Enum):
    """
    Enum for different types of inputs used in mathematical modeling.

    Attributes:
        OFF: Use Unity settings as inputs.
        LOW: Use of limited parameter estimation for model inputs.
        MEDIUM: Use of parameter estimation for half of the model inputs.
        HIGH: Use of parameter estimation for the majority of model inputs.
        FULL: Use of parameter estimation for all model inputs.
    """

    OFF = "unity_settings"
    LOW = "low_parameter_estimation"
    MEDIUM = "medium_parameter_estimation"
    HIGH = "high_parameter_estimation"
    FULL = "full_parameter_estimation"


class DataVariant(Enum):
    """
    Enum for different types of neural network inputs.

    Attributes:
        IDEAL: Ideal dataset without any noise or distortion.
        OPTICAL_FLOW: Data obtained from optical flow analysis.
        GAUSSIAN_NOISE: Data with added Gaussian noise.
        JITTER_NOISE: Data with added jitter noise.
    """

    IDEAL = "ideal_data"
    OPTICAL_FLOW = "optical_flow_data"
    GAUSSIAN_NOISE = "gaussian_noise"
    JITTER_NOISE = "jitter_noise"


class DictKeys(Enum):
    """
    Enum for different keys used in dictionary structures.

    Attributes:
        PROCESSED_DATA: Key for processed data.
        UNITY_ANNOTATIONS: Key for Unity annotations.
        UNITY_SETTINGS: Key for Unity settings.
        UNITY_POSITIONS: Key for Unity positions.
        UNITY_VELOCITIES: Key for Unity velocities.
        UNITY_ANGLES: Key for angular positions of an object.
        UNITY_ANGULAR_VELOCITIES: Key for angular velocities of an object.
        OPTICAL_FLOW_DATA: Key for optical flow data.
        OPTICAL_FLOW_POSITIONS: Key for positions in optical flow data.
        OPTICAL_FLOW_VELOCITIES: Key for velocities in optical flow data.
        NOISY_DATA: Key for noisy data.
        NOISY_DATA_POSITIONS: Key for positions in noisy data.
        NOISY_DATA_VELOCITIES: Key for velocities in noisy data.
    """

    PROCESSED_DATA = "processed_data"

    UNITY_ANNOTATIONS = "unity_annotations"
    UNITY_SETTINGS = "unity_settings"
    UNITY_POSITIONS = "position"
    UNITY_VELOCITIES = "velocity"
    UNITY_ANGLES = "objectAngularAngle"
    UNITY_ANGULAR_VELOCITIES = "objectAngularSpeed"

    OPTICAL_FLOW_DATA = "optical_flow_data"
    OPTICAL_FLOW_POSITIONS = "projected_positions"
    OPTICAL_FLOW_VELOCITIES = "projected_velocities"

    NOISY_DATA = "gaussian_noise"
    NOISY_DATA_POSITIONS = "position"
    NOISY_DATA_VELOCITIES = "velocity"


class LogName(Enum):
    """
    Enum for different types of log names used in the logging process.

    Attributes:
        ACCURACY: Log name for accuracy.
        LOSS_TRAIN_TOT: Log name for total training loss.
        LOSS_TRAIN_CE: Log name for cross-entropy training loss.
        LOSS_TRAIN_KL: Log name for KL-divergence training loss.
    """

    ACCURACY = "ACCURACY"
    LOSS_TRAIN_TOT = "LOSS_TRAIN_TOT"
    LOSS_TRAIN_CE = "LOSS_TRAIN_CE"
    LOSS_TRAIN_KL = "LOSS_TRAIN_KL"
