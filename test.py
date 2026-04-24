from pathlib import Path

from src.model_training import train_and_select_model
from utils.helper import create_directories, load_config


def main():
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "config.yaml")
    create_directories(config["paths"])

    artifact = train_and_select_model(str(project_root / "config.yaml"))
    print("Setup successful")
    print(f"Best model: {artifact['model_name']}")
    print(f"Metrics: {artifact['metrics']}")


if __name__ == "__main__":
    main()
