from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parents[3]


@dataclass
class Config:

    # ------raw-------
    train_data_path: Path = BASE_DIR / "data/machine-learning-medium-digit-recognizer/train.csv"
    test_data_path: Path = BASE_DIR / "data/machine-learning-medium-digit-recognizer/test.csv"
    simple_data_path: Path = BASE_DIR / "data/machine-learning-medium-digit-recognizer/simple_data.csv"

    # -------processed------
    train_data_npy_path: Path = BASE_DIR / "data/train.npy"
    test_data_npy_path: Path = BASE_DIR / "data/test.npy"

    # --------weight---------
    epochs: int = 50
    # batch_size: int = 32
    batch_size:int = 1
    learning_rate: float = 0.001

    # ------------data---------
    data_rate = 0.8
    model_save_path: Path = BASE_DIR / "data/model.pth"
