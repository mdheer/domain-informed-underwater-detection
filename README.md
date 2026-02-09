# Domain-Informed Underwater Detection

End-to-end research code for building underwater debris detectors that combine synthetic Unity trajectories, domain knowledge derived from a mathematical model, and modern neural detectors (YOLO + a lightweight trajectory classifier).

- Generate reproducible datasets from large Unity exports (`parsed_data.json` plus raw frame folders).
- Convert Unity annotations into YOLO labels and spin up Ultralytics training runs.
- Train a trajectory classifier that can optionally distill supervision from the mathematical model.
- Inspect data quality, log experiments, and post-process trajectories with the provided utilities.

## Repository Layout

```
.
├── config.json                 # Maps enum identifiers to local/cluster data roots
├── dataset_configs/            # Splitting strategies consumed by dataset_generator.py
├── src/
│   ├── data_preprocessing/     # Dataset curation, filtering, YOLO label conversion
│   ├── data_inspection/        # Plotting + logging helpers
│   ├── domain_knowledge/       # Mathematical model + classifier (teacher model)
│   ├── neural_network/         # Student classifier, dataset + training loop
│   ├── tools/                  # General utilities, enums, plotting, conversions
│   ├── train_trajectory.py     # Experiment runner for the trajectory model
│   └── train_yolo.py           # Ultralytics YOLO training/inference helper
├── unity_data/                 # Example Unity export with parsed_data.json + images
└── dataset_generator.py        # Builds datasets/<name>/... folders from configs
```

## Getting Started

### 1. Install dependencies

Use Python 3.10+ and create an isolated environment:

```bash
pip install -r requirements.txt
```


### 2. Point the code to your data

`src/tools.general_functions.read_input_path` reads from `config.json`. Update the file so every `InputDataPath` enum (see `src/tools/enums.py`) resolves to a valid folder. Each folder is expected to contain:

```
<data_root>/
├── parsed_data.json              # Aggregated trajectories exported from Unity
└── <sequence_id>/
    ├── *.jpg                     # Rendered frames
    ├── annotations.json          # Bounding boxes per frame
    └── settings.json             # Unity environment metadata
```

The sample `unity_data/` directory follows this convention and can be used for smoke tests. 

### 3. Verify YOLO data directories

YOLO training symlinks labels/images into `./yolo_data/{images,labels}/{train,val,test}`. Create those folders once:

```bash
mkdir -p yolo_data/images/{train,val,test} yolo_data/labels/{train,val,test}
```

## Typical Workflow
1. **Download the full dataset**
   The full dataset can be found here: `https://data.4tu.nl/datasets/09c93995-2d5b-4e44-9de6-b117c87b4704`
2. **Preprocess Unity annotations (optional).**  
   Use `python src/preprocess_yolo.py` to convert Unity JSON annotations into YOLO-format `.txt` files and preview random examples with bounding boxes. Adjust `InputDataPath` inside the script to choose the source dataset.

3. **Generate curated datasets for experiments.**  
   - Duplicate one of the templates in `dataset_configs/` (e.g., `small_test_set.py`).  
   - Update `input_data_path`, desired `DataVariant` per split, the number of samples, and whether any `WaterCurrentFilter` should be applied.  
   - Run `python src/dataset_generator.py` after changing `dataset_file_name` to your new config.  
   - The script creates `./datasets/<dataset_name>/k_fold_<id>/...` along with JSON distributions, YOLO YAML files, and optional unlabelled folds.

4. **Train the trajectory classifier (with or without domain knowledge).**  
   - Configure the experiment header inside `src/train_trajectory.py` (dataset name, experiment slug, knowledge temperature/alpha, epochs, batch size, etc.).  
   - Run `python src/train_trajectory.py`.  
   - For every fold and distribution the runner toggles `embed_domain_knowledge` on/off, logs to `./training_logs/<experiment>/<k_fold>/`, and stores configs as JSON for reproducibility.  
   - The training loop lives in `src/neural_network/main.py`. It loads the dataset JSON, maps `DataVariant` to the right parameter estimation regime, builds `CustomDataset` instances, and distills teacher predictions from `DomainKnowledgeClassifier` when enabled.

5. **Train a YOLO detector on the same data splits.**  
   - Ensure `dataset_name` (inside `src/train_yolo.py`) matches the dataset folder generated earlier.  
   - For each distribution JSON the script:
     1. Calls `prepare_dataset` to clear `yolo_data/` and create symlinks that match the Ultralytics YAML.
     2. Trains `ultralytics.YOLO()` with the specified number of epochs.
     3. Evaluates on the held-out test set via `model(test_data_path)` and stores detections under `runs/detect/<run_name>/test_detections.json`.

6. **Inspect, validate, and post-process.**  
   - `src/data_inspection/plotting.py`, `src/data_inspection/training_data_analysis.py`, and `src/validation/domain_knowledge.py` contain helpers for sanity checks, visualizations, or comparing the mathematical model to Unity ground truth.  
   - `src/postprocess_trajectory.py` and `src/tools/*.py` add utilities for aggregating predictions, creating videos, and exporting plots.

## Configuration Reference

- **DatasetConfig (`src/data_preprocessing/data_curation.py`):**
  - `data_split`: `DataSplit.UNLABELLED | TRAIN | VALIDATE | TEST`
  - `number_of_datapoints`: samples drawn for the split (per distribution)
  - `data_variant`: `DataVariant.IDEAL | OPTICAL_FLOW | GAUSSIAN_NOISE | JITTER_NOISE`
  - `filter_based_on_current`: `WaterCurrentFilter.LIGHT_CURRENT | STRONG_CURRENT | None`
  - `k_folds_sets` (in the module) states which splits participate in k-fold reshuffling.

- **Domain knowledge toggles:** `TrainNeuralNetwork` receives `embed_domain_knowledge` plus `temperature` and `alpha` (see `ExperimentRunner`). Teacher predictions come from `DomainKnowledgeClassifier`, which queries pre-computed mathematical model outputs in `unity_data/mathematical_model_outputs_*`.

- **Logging:** `src/data_inspection/logger.MainLogger` writes per-epoch metrics (accuracy, CE/KL loss, etc.) into `training_logs/<experiment>/<k_fold>/<distribution>/`. YOLO logs follow Ultralytics’ `runs/` convention.

## Utilities & Tips

- **Data filtering:** `src/data_preprocessing/preprocessing.py` provides optical-flow accuracy checks, current filters, dataset directory builders, and YOLO YAML emitters.  
- **General helpers:** `src/tools/general_functions.py` exposes list merging, timestamp synchronization, and dataset sorting.  
- **Visualization:** `src/tools/create_video.py`, `src/tools/process_plot.py`, `src/tools/gradually_plotter.py`, and `src/data_inspection/plotting.py` simplify debugging new datasets.  
- **Slack notifications:** `train_trajectory.py` can ping Slack via `send_slack_message`; fill in a token and channel before enabling.

## Re-running Experiments

1. Update or add a dataset config under `dataset_configs/`.
2. Run `python src/dataset_generator.py` (answer prompts if a dataset already exists).
3. Launch `python src/train_trajectory.py` with matching `dataset_name`.
4. Launch `python src/train_yolo.py` to train the detector on identical folds.
5. Inspect results under `training_logs/` and `runs/detect/`.
