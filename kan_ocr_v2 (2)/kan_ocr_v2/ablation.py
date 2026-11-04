import time
import multiprocessing

import torch
import pandas as pd

from train import run_training


CONFIGS = [
    dict(name="linear_baseline", head_type="linear"),
    dict(name="kan1_grid5", head_type="kan1", kan_grid=5),
    dict(name="kan1_grid8", head_type="kan1", kan_grid=8),
    dict(name="kan2_grid5", head_type="kan2", kan_grid=5, kan_hidden=128),
    dict(name="kan2_grid8", head_type="kan2", kan_grid=8, kan_hidden=128),
    dict(name="kan2_grid8_hidden64", head_type="kan2", kan_grid=8, kan_hidden=64),
]

EPOCHS = 15


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    results = []

    for cfg in CONFIGS:

        cfg = cfg.copy()
        name = cfg.pop("name")

        print(f"\n=== {name} ===")

        t0 = time.time()

        result = run_training(
            epochs=EPOCHS,
            patience=EPOCHS,
            num_workers=4,
            log_prefix=f"[{name}] ",
            verbose=True,
            **cfg,
        )

        elapsed = time.time() - t0

        results.append(
            {
                "name": name,
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "num_params": result["num_params"],
                "elapsed_s": elapsed,
                "final_train_acc": result["history"]["train_acc"].iloc[-1],
            }
        )

        print(
            f"--- {name}: "
            f"best_val_acc={result['best_val_acc']:.2f}% "
            f"test_acc={result['test_acc']:.2f}% "
            f"params={result['num_params']:,} "
            f"time={elapsed:.1f}s ---"
        )

    print("\n\n=== ABLATION SUMMARY ===")

    print(
        f"{'name':<24}"
        f"{'val_acc':>10}"
        f"{'test_acc':>10}"
        f"{'train_acc':>11}"
        f"{'params':>12}"
        f"{'time_s':>9}"
    )

    for r in sorted(results, key=lambda x: -x["best_val_acc"]):

        print(
            f"{r['name']:<24}"
            f"{r['best_val_acc']:>9.2f}%"
            f"{r['test_acc']:>9.2f}%"
            f"{r['final_train_acc']:>10.2f}%"
            f"{r['num_params']:>12,}"
            f"{r['elapsed_s']:>8.1f}s"
        )

    pd.DataFrame(results).to_csv(
        "ablation_results.csv",
        index=False
    )

    print("\nSaved ablation_results.csv")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
