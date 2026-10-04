"""Resource-safe Helpdesk smoke run for the U-ED-LSTM pipeline."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from event_log_loader.new_event_log_loader_v2 import EventLogLoader
from evaluation_v2.probabilistic_evaluation import ProbabilisticEvaluation
from loss.losses import Loss
from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import (
    DropoutUncertaintyEncoderDecoderLSTM,
)
from trainer.trainer import Trainer


SOURCE_COLUMNS = {
    "Case ID",
    "Activity",
    "Resource",
    "Complete Timestamp",
    "Variant index",
    "seriousness",
    "customer",
    "product",
    "responsible_section",
    "seriousness_2",
    "service_level",
    "service_type",
    "support_section",
    "workgroup",
}

STATIC_CATEGORICAL_COLUMNS = [
    "VariantIndex",
    "seriousness",
    "customer",
    "product",
    "responsible_section",
    "seriousness_2",
    "service_level",
    "service_type",
    "support_section",
    "workgroup",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small end-to-end smoke test on the Helpdesk dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "finale.csv",
        help="Path to the original, enriched Helpdesk CSV.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=200,
        help="Maximum number of cases to use; 0 uses the complete dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=0,
        help="Training epochs for a pipeline test; zero skips training.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=32,
        help="Maximum training samples used by the pipeline test.",
    )
    parser.add_argument(
        "--evaluation-samples",
        type=int,
        default=2,
        help="MC suffix samples generated for one test prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optionally save the encoded train/validation/test datasets.",
    )
    return parser.parse_args()


def helpdesk_properties() -> dict[str, object]:
    return {
        "case_name": "CaseID",
        "concept_name": "Activity",
        "timestamp_name": "CompleteTimestamp",
        "date_format": "%Y/%m/%d %H:%M:%S.%f",
        "time_since_case_start_column": "case_elapsed_time",
        "time_since_last_event_column": "event_elapsed_time",
        "day_in_week_column": "day_in_week",
        "seconds_in_day_column": "seconds_in_day",
        "min_suffix_size": 5,
        "train_validation_size": 0.15,
        "test_validation_size": 0.2,
        "window_size": "auto",
        "categorical_columns": ["Activity", "Resource"],
        "continuous_columns": [
            "case_elapsed_time",
            "event_elapsed_time",
            "day_in_week",
            "seconds_in_day",
        ],
        "continuous_positive_columns": [],
        "static_categorical_columns": STATIC_CATEGORICAL_COLUMNS,
        "static_continuous_columns": [],
    }


def prepare_input(data_path: Path, max_cases: int, temp_dir: Path) -> Path:
    dataframe = pd.read_csv(data_path)
    missing = SOURCE_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing Helpdesk columns: {', '.join(sorted(missing))}")

    dataframe = dataframe.rename(
        columns={
            "Case ID": "CaseID",
            "Complete Timestamp": "CompleteTimestamp",
            "Variant index": "VariantIndex",
        }
    )

    if max_cases < 0:
        raise ValueError("--max-cases must be zero or positive")
    if max_cases and dataframe["CaseID"].nunique() > max_cases:
        case_ids = dataframe["CaseID"].drop_duplicates().iloc[:max_cases]
        dataframe = dataframe[dataframe["CaseID"].isin(case_ids)]

    prepared_path = temp_dir / "helpdesk.csv"
    dataframe.to_csv(prepared_path, index=False)
    print(
        f"Input: {len(dataframe)} events in "
        f"{dataframe['CaseID'].nunique()} cases"
    )
    return prepared_path


def build_model(dataset) -> DropoutUncertaintyEncoderDecoderLSTM:
    categories, numerics = dataset.all_categories
    encoder_features = [
        [feature[0] for feature in categories],
        [feature[0] for feature in numerics],
    ]
    decoder_features = [
        ["Activity"],
        ["case_elapsed_time", "event_elapsed_time"],
    ]
    static_categories, static_numerics = dataset.all_static_categories
    static_encoder_features = [
        [feature[0] for feature in static_categories],
        [feature[0] for feature in static_numerics],
    ]
    return DropoutUncertaintyEncoderDecoderLSTM(
        data_set_categories=dataset.all_categories,
        enc_feat=encoder_features,
        dec_feat=decoder_features,
        seq_len_pred=dataset.min_suffix_size,
        hidden_size=128,
        num_layers=4,
        dropout=0.1,
        static_data_set_categories=dataset.all_static_categories,
        static_enc_feat=static_encoder_features,
    )


def run_model_smoke_test(
    dataset, batch_size: int
) -> DropoutUncertaintyEncoderDecoderLSTM:
    if len(dataset) == 0:
        raise RuntimeError("The selected Helpdesk subset produced no training samples")

    model = build_model(dataset)
    model.eval()
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False)))
    _, categories, numerics, _, zero_mask, static_categories, static_numerics, _ = (
        batch
    )
    suffix_size = dataset.min_suffix_size
    prefixes = [
        [value[:, :-suffix_size] for value in categories],
        [value[:, :-suffix_size] for value in numerics],
    ]
    suffixes = [
        [value[:, -suffix_size:] for value in categories],
        [value[:, -suffix_size:] for value in numerics],
    ]

    with torch.no_grad():
        predictions, *_ = model(
            prefixes=prefixes,
            static_inputs=(static_categories, static_numerics),
            suffixes=suffixes,
            teacher_forcing_ratio=0.0,
            prefix_mask=zero_mask[:, :-suffix_size],
        )

    categorical_predictions, numerical_predictions = predictions
    output_names = sorted(categorical_predictions) + sorted(numerical_predictions)
    print(f"Model forward pass: OK ({', '.join(output_names)})")
    return model


class SilentWriter:
    """Minimal TensorBoard-compatible writer for smoke training."""

    def add_scalars(self, *_args, **_kwargs) -> None:
        pass


def run_training_test(
    model: DropoutUncertaintyEncoderDecoderLSTM,
    train_dataset,
    val_dataset,
    epochs: int,
    train_samples: int,
    batch_size: int,
    output_dir: Path,
) -> None:
    if epochs < 0:
        raise ValueError("--train-epochs must be zero or positive")
    if train_samples <= 0:
        raise ValueError("--train-samples must be positive")

    train_subset = train_dataset.subset(
        range(min(train_samples, len(train_dataset)))
    )
    val_subset = val_dataset.subset(
        range(min(max(batch_size, train_samples // 2), len(val_dataset)))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=15, min_lr=1e-8
    )
    trainer = Trainer(
        device=torch.device("cpu"),
        model=model,
        data_train=train_subset,
        data_val=val_subset,
        loss_obj=Loss(),
        log_normal_loss_num_feature=[],
        optimize_values={
            "regularization_term": 1e-4,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epochs": epochs,
            "mini_batches": batch_size,
            "shuffle": True,
            "min_teacher_forcing_value": 0.0,
            "max_teacher_forcing_value": 1.0,
        },
        suffix_data_split_value=train_dataset.min_suffix_size,
        writer=SilentWriter(),
        gradnorm_values={
            "use_gradnorm": True,
            "number_tasks": 3,
            "gn_alpha": 1.5,
            "gn_learning_rate": 1e-4,
        },
        save_model_n_th_epoch=1,
        saving_path=str(output_dir / "helpdesk_smoke_model.pkl"),
    )
    trainer.train_model(
        use_statics=True,
        use_zero_padd_masking=True,
        use_eos_padd_masking=True,
    )
    print(
        f"Training: OK ({epochs} epoch(s), {len(train_subset)} training samples)"
    )


def run_evaluation_test(
    model: DropoutUncertaintyEncoderDecoderLSTM,
    test_dataset,
    samples_per_case: int,
) -> None:
    if samples_per_case <= 0:
        raise ValueError("--evaluation-samples must be positive")

    model.eval()
    evaluator = ProbabilisticEvaluation(
        model=model,
        dataset=test_dataset,
        concept_name="Activity",
        growing_num_values=["case_elapsed_time"],
        positive_num_values=["event_elapsed_time"],
        decoder_cat=["Activity"],
        decoder_num=["case_elapsed_time", "event_elapsed_time"],
        num_processes=1,
        samples_per_case=samples_per_case,
    )
    result = next(evaluator.evaluate(random_order=False))
    case_name, prefix_length, _, predicted_suffixes, *_ = result
    if len(predicted_suffixes) != samples_per_case:
        raise RuntimeError("Evaluation returned an unexpected number of MC samples")
    print(
        "MC evaluation: OK "
        f"(case={case_name}, prefix={prefix_length}, samples={samples_per_case})"
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with tempfile.TemporaryDirectory(prefix="helpdesk-") as temp:
        input_path = prepare_input(args.data, args.max_cases, Path(temp))
        loader = EventLogLoader(str(input_path), helpdesk_properties())
        datasets = {
            split: loader.get_dataset(split) for split in ("train", "val", "test")
        }

    print(
        "Encoded samples: "
        + ", ".join(f"{name}={len(dataset)}" for name, dataset in datasets.items())
    )
    print(f"Window size: {datasets['train'].encoder_decoder.window_size}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for split, dataset in datasets.items():
            torch.save(dataset, args.output_dir / f"helpdesk_{split}.pkl")
        print(f"Saved encoded datasets to: {args.output_dir}")

    model = run_model_smoke_test(datasets["train"], args.batch_size)
    if args.train_epochs:
        with tempfile.TemporaryDirectory(prefix="helpdesk-training-") as temp:
            run_training_test(
                model=model,
                train_dataset=datasets["train"],
                val_dataset=datasets["val"],
                epochs=args.train_epochs,
                train_samples=args.train_samples,
                batch_size=args.batch_size,
                output_dir=Path(temp),
            )
        run_evaluation_test(
            model=model,
            test_dataset=datasets["test"],
            samples_per_case=args.evaluation_samples,
        )
    print("Helpdesk smoke test completed successfully.")


if __name__ == "__main__":
    main()
