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
PROXIMAL_JOINTS = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17]
DISTAL_JOINTS = [i for i in range(24) if i not in PROXIMAL_JOINTS]


def compute_means(jointwise_errors):
    jointwise_errors = np.array(jointwise_errors)
    return {
        "overall": jointwise_errors.mean(),
        "proximal": jointwise_errors[PROXIMAL_JOINTS].mean(),
        "distal": jointwise_errors[DISTAL_JOINTS].mean()
    }


def plot_histograms(per_sample, key, out_dir):
    for group in ["overall", "male", "female"]:
        values = per_sample.get(key, {}).get(group, [])
        if not values:
            print(f"⚠️ No data to plot {key} histogram for {group}.")
            continue

        plt.figure(figsize=(6, 4))
        sns.histplot(values, kde=True, bins=20, stat="density", alpha=0.6)
        plt.title(f"{key.upper()} Distribution ({group.capitalize()})")
        plt.xlabel(f"{key.upper()} (cm)")
        plt.ylabel("Density")
        plt.tight_layout()
        path = os.path.join(out_dir, f"{key}_histogram_{group}.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Saved {key} histogram for {group} to: {path}")


def plot_jointwise_bar(stats, out_dir):
    x = np.arange(len(JOINT_NAMES))
    width = 0.25

    overall = stats["overall"]["jointwise_mpjpe_cm"]
    male = stats["male"]["jointwise_mpjpe_cm"]
    female = stats["female"]["jointwise_mpjpe_cm"]

    plt.figure(figsize=(18, 6))
    plt.bar(x - width, male, width=width, label="Male")
    plt.bar(x, overall, width=width, label="Overall")
    plt.bar(x + width, female, width=width, label="Female")

    plt.xticks(ticks=x, labels=JOINT_NAMES, rotation=45, ha="right")
    plt.ylabel("MPJPE (cm)")
    plt.title("Joint-wise MPJPE by Gender")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "jointwise_mpjpe_barplot.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved jointwise bar plot to: {path}")


def plot_jointwise_boxplots(per_sample, out_dir):
    for group in ["overall", "male", "female"]:
        matrix = per_sample.get("jointwise_mpjpe_cm", {}).get(group, [])
        if not matrix:
            print(f"⚠️ No data for jointwise box plot: {group}")
            continue

        matrix = np.array(matrix)
        plt.figure(figsize=(16, 5))
        sns.boxplot(data=matrix, width=0.5, fliersize=2)
        plt.xticks(np.arange(24), JOINT_NAMES, rotation=45, ha="right")
        plt.ylabel("MPJPE (cm)")
        plt.title(f"Joint-wise MPJPE Boxplot ({group.capitalize()})")
        plt.tight_layout()
        path = os.path.join(out_dir, f"jointwise_mpjpe_boxplot_{group}.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Saved jointwise box plot for {group} to: {path}")


def plot_grouped_joint_error(stats, out_dir):
    for group in ["overall", "male", "female"]:
        jointwise = stats.get(group, {}).get("jointwise_mpjpe_cm", [])
        if not jointwise:
            print(f"⚠️ No jointwise data for {group}")
            continue

        means = compute_means(jointwise)
        labels = list(means.keys())
        values = list(means.values())

        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, values)
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, y + 0.1, f"{y:.2f}", ha='center')
        plt.ylabel("MPJPE (cm)")
        plt.title(f"Mean MPJPE: {group.capitalize()}")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        path = os.path.join(out_dir, f"grouped_joint_error_barplot_{group}.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Saved grouped joint error bar plot for {group} to: {path}")


def plot_jointwise_colored_bar(stats, out_dir):
    for group in ["overall", "male", "female"]:
        jointwise = stats.get(group, {}).get("jointwise_mpjpe_cm", [])
        if not jointwise:
            print(f"⚠️ No data for colored jointwise bar plot: {group}")
            continue

        jointwise = np.array(jointwise)
        means = compute_means(jointwise)
        colors = ['red' if i in PROXIMAL_JOINTS else 'blue' for i in range(24)]
        x = np.arange(24)

        plt.figure(figsize=(12, 6))
        plt.bar(x, jointwise, color=colors)
        plt.axhline(means["overall"], color='blue', linestyle='--', label=f'Mean Overall = {means["overall"]:.2f}')
        plt.axhline(means["distal"], color='blue', linestyle=':', label=f'Mean Distal = {means["distal"]:.2f}')
        plt.axhline(means["proximal"], color='red', linestyle=':', label=f'Mean Proximal = {means["proximal"]:.2f}')

        plt.xticks(ticks=x, labels=[f"{i+1}. {name}" for i, name in enumerate(JOINT_NAMES)], rotation=45, ha="right")
        plt.ylabel("MPJPE (cm)")
        plt.title(f"Bar Graph - Joint-wise Error ({group.capitalize()})")
        plt.legend(loc="upper right", frameon=True)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        path = os.path.join(out_dir, f"jointwise_mpjpe_colored_barplot_{group}.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Saved colored jointwise MPJPE bar plot for {group} to: {path}")


def main(json_path):
    with open(json_path, "r") as f:
        stats = json.load(f)

    out_dir = os.path.join(os.path.dirname(json_path), "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    per_sample = stats.get("per_sample", {})

    print(f"\n📊 Generating plots from: {json_path}")
    plot_histograms(per_sample, "mpjpe_cm", out_dir)
    plot_histograms(per_sample, "v2v_cm", out_dir)
    plot_jointwise_bar(stats, out_dir)
    plot_jointwise_boxplots(per_sample, out_dir)
    plot_grouped_joint_error(stats, out_dir)
    plot_jointwise_colored_bar(stats, out_dir)
    print(f"\n✅ All plots saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results from final_test_metrics.json")
    parser.add_argument("--input", type=str, default="/home/nashah/projects/bodies-at-rest/runs/20250704_175521/final_test_metrics.json", help="Path to final_test_metrics.json")
    args = parser.parse_args()
    main(args.input)
