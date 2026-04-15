import argparse
from pathlib import Path

from cmeaa_xai.data.iu_xray import build_iu_xray_dataframe_from_kaggle_layout, make_splits, save_splits
from cmeaa_xai.data.observation_miner import mine_observation_ngrams, save_phrase_bank


def main():
    parser = argparse.ArgumentParser(description="Prepare IU-Xray from the Kaggle layout.")
    parser.add_argument("--dataset-root", default="", help="Root directory returned by kagglehub.dataset_download.")
    parser.add_argument("--download-kaggle", action="store_true", help="Download raddar/chest-xrays-indiana-university with kagglehub.")
    parser.add_argument("--output-dir", required=True, help="Directory to save CSV splits and phrase bank.")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if args.download_kaggle:
        import kagglehub

        dataset_root = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
    if not dataset_root:
        raise ValueError("Provide --dataset-root or use --download-kaggle.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_iu_xray_dataframe_from_kaggle_layout(dataset_root)
    splits = make_splits(frame)
    csv_paths = save_splits(splits, str(output_dir))

    train_df = splits["train"]
    reports = train_df["report"].tolist()
    labels = [
        {col.replace("obs_", ""): int(row[col]) for col in train_df.columns if col.startswith("obs_")}
        for _, row in train_df.iterrows()
    ]
    phrase_bank = mine_observation_ngrams(reports, labels)
    save_phrase_bank(phrase_bank, str(output_dir / "phrase_bank.json"))

    print("Saved:")
    for split_name, path in csv_paths.items():
        print(f"  {split_name}: {path}")
    print(f"  phrase_bank: {output_dir / 'phrase_bank.json'}")


if __name__ == "__main__":
    main()
