"""Central configuration for training and testing."""

from pathlib import Path
import os
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('AMPNET_DATA_DIR', str(PROJECT_ROOT / 'datasets')))
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', (PROJECT_ROOT / 'mlruns').as_uri())
MLFLOW_EXPERIMENT_NAME = 'sta_rppg'

TRAIN_FEATURES_PATH = DATA_ROOT / 'ibvp_train_features.pth'
TRAIN_LABELS_PATH = DATA_ROOT / 'ibvp_train_labels.pth'
TEST_FEATURES_PATH = DATA_ROOT / 'ibvp_test_features.pth'
TEST_LABELS_PATH = DATA_ROOT / 'ibvp_test_labels.pth'
MODEL_DIR = PROJECT_ROOT / 'model_paths'

SEGMENT_LENGTH = 128
SESSION_LENGTH = 1792
SAMPLING_RATE = 28
DATASET_DIVISION = 'subject-wise'  # or 'random'
BATCH_SIZE = 4
K_FOLDS = 2
PATH_FOLDS = 1

MODALITY = 'multimodal'  # 'rgb', 'thermal', or 'multimodal'
DEMOGRAPHIES = ['asian', 'black', 'caucasian', 'mixed']
NUM_DEMOGRAPHY_GROUPS = 4

PERTURBATION_PLAN = {
    None: ['clean'],
    'resolution': ['mild', 'moderate', 'severe'],
    'shift': ['mild', 'moderate', 'severe'],
}

RGB_MODEL_NAMES = ['R3EDSAN', 'PhysNet', 'iBVPNet', 'RTrPPG']
THERMAL_MODEL_NAMES = ['T3ED', 'T3EDSAN-TAM', 'T3EDSAN-CBAM']
MULTIMODAL_MODEL_NAMES = ['AMPNet']
TRAIN_RGB_MODEL_NAMES = ['R3EDSAN-CBAM', 'R3EDSAN-TAM', 'R3ED']
TRAIN_THERMAL_MODEL_NAMES = ['T3EDSAN-CBAM', 'T3EDSAN-TAM', 'T3ED']

