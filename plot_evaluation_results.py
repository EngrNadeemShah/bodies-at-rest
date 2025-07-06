import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

def plot_histograms(per_sample, key, out_dir):
    for group in ["overall", "male", "female"]:
        values = per_sample.get(key, {}).get(group, [])
        if values:
            plt.figure(figsize=(6, 4))
            sns.histplot(values, kde=True, bins=20, stat="density", alpha=0.6)
            plt.title(f"{key.upper()} Distribution ({group.capitalize()})")
            plt.xlabel(f"{key.upper()} (cm)")
            plt.ylabel("Density")
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{key}_histogram_{group}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"✅ Saved {key} histogram for {group} to: {out_path}")
        else:
            print(f"⚠️ No data to plot {key} histogram for {group}.")

def plot_jointwise_bar(stats, out_dir):
    x = np.arange(len(JOINT_NAMES))
    width = 0.25

    overall = stats["overall"]["jointwise_mpjpe_cm"]
    male    = stats["male"]["jointwise_mpjpe_cm"]
    female  = stats["female"]["jointwise_mpjpe_cm"]

    plt.figure(figsize=(18, 6))
    if male: plt.bar(x - width, male, width=width, label="Male")
    plt.bar(x, overall, width=width, label="Overall")
    if female: plt.bar(x + width, female, width=width, label="Female")

    plt.xticks(ticks=x, labels=JOINT_NAMES, rotation=45, ha="right")
    plt.ylabel("MPJPE (cm)")
    plt.title("Joint-wise MPJPE by Gender")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis="y")

    out_path = os.path.join(out_dir, "jointwise_mpjpe_barplot.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved jointwise bar plot to: {out_path}")

def plot_jointwise_boxplots(per_sample, out_dir):
    for group in ["overall", "male", "female"]:
        matrix = per_sample.get("jointwise_mpjpe_cm", {}).get(group, [])
        if matrix:
            matrix = np.array(matrix)
            plt.figure(figsize=(16, 5))
            sns.boxplot(data=matrix, width=0.5, fliersize=2)
            plt.xticks(np.arange(24), JOINT_NAMES, rotation=45, ha="right")
            plt.ylabel("MPJPE (cm)")
            plt.title(f"Joint-wise MPJPE Boxplot ({group.capitalize()})")
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"jointwise_mpjpe_boxplot_{group}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"✅ Saved jointwise box plot for {group} to: {out_path}")
        else:
            print(f"⚠️ No data for jointwise box plot: {group}")

def main(json_path):
    with open(json_path, "r") as f:
        stats = json.load(f)

    out_dir = os.path.dirname(json_path)
    per_sample = stats.get("per_sample", {})

    print("\n📊 Generating plots from:", json_path)

    plot_histograms(per_sample, "mpjpe_cm", out_dir)
    plot_histograms(per_sample, "v2v_cm", out_dir)
    plot_jointwise_bar(stats, out_dir)
    plot_jointwise_boxplots(per_sample, out_dir)

    print("\n✅ All plots saved to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results from final_test_metrics.json")
    parser.add_argument("--input", type=str, required=False, help="Path to final_test_metrics.json", default="/home/nashah/projects/bodies-at-rest/runs/20250704_175521/final_test_metrics.json")
    args = parser.parse_args()
    main(args.input)
