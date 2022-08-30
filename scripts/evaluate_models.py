from pathlib import Path
import torch
from log_analyzer.evaluator import Evaluator
from log_analyzer.train_loop import eval_model, init_from_config_files
import csv
import pandas as pd


model_dir = Path("trained_models")
names = ["jakob", "simon", "yeongwoo"]

data_dir = Path("data")

log_data_dir = data_dir / "sample_data"
counts_file = data_dir / "counts678.json"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_single(run_dir: Path):
    name_parts = run_dir.name.split("_")
    run_id = name_parts[0]
    model_type = name_parts[1]
    tokenization = name_parts[2]
    model_conf: Path = run_dir / "model_config.json"
    trainer_conf: Path = run_dir / "trainer_config.json"
    evaluator: Evaluator
    _, evaluator, _, _, test_loader = init_from_config_files(
        model_type=model_type,
        bidirectional=False,
        model_config_file=model_conf,
        tokenization=tokenization,
        trainer_config_file=trainer_conf,
        data_folder=log_data_dir,
        counts_file=counts_file,
    )
    with open(run_dir / "model_best.pt", "rb") as f:
        state_dict = torch.load(f, map_location=device)
    evaluator.model.load_state_dict(state_dict)
    eval_model(evaluator, test_loader, True)
    eval_data = evaluator.run_all()
    eval_data["tokenization"] = tokenization
    eval_data["run_id"] = run_id
    eval_data["model_type"] = model_type
    return eval_data


def write_to_csv(eval_data: dict):
    with open("eval_data.csv", "a") as f:
        writer = csv.DictWriter(f, fieldnames=eval_data.keys())
        writer.writerow(eval_data)


def evaluate_set_of_models(folder: Path, first_entry: bool):

    eval_data = [evaluate_single(run_dir) for run_dir in folder.iterdir()]

    df = pd.DataFrame(eval_data)
    write_mode = "w" if first_entry else "a"
    df.to_csv("eval_data.csv", mode=write_mode, header=first_entry, index=False)
    df.groupby(["model_type", "tokenization"]).mean().to_csv(
        "eval_data_mean.csv", mode=write_mode, header=first_entry, index=True
    )
    
    return False


def main():
    first = True
    for name in names:
        first = evaluate_set_of_models(model_dir / name, first)


if __name__ == "__main__":
    main()
