from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_FILE = Path(__file__).resolve().parent / "output" / "cleaned_data.csv"
MULTI_CHOICE_INPUT_FILE = Path(__file__).resolve().parent / "output" / "multi_choice_binary.csv"
Q6_INPUT_FILE = Path(__file__).resolve().parent / "output" / "q6_ranked.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_OUTPUT_FILE = OUTPUT_DIR / "descriptive_summary.csv"
SINGLE_CHOICE_PREFIXES = ["1、", "2、", "4、", "5、"]
MULTI_CHOICE_PREFIXES = ["3、", "7、", "8、", "11、", "12、", "13、"]
Q6_WEIGHTS = {"Q6_第1选": 3, "Q6_第2选": 2, "Q6_第3选": 1}
Q9_PREFIX = "9、"
Q10_PREFIX = "10、"


def configure_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        palette="muted",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "KaiTi",
                "FangSong",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        },
    )


def find_column_by_prefix(dataframe: pd.DataFrame, prefix: str) -> str:
    return next(column for column in dataframe.columns if str(column).startswith(prefix))


def clean_text_value(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text == "(空)":
        return None
    return text


def simplify_option_label(label: str) -> str:
    return label.replace("__〖", "（").replace("〗", "）")


def summarize_single_choice(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    valid_answers = dataframe[column_name].dropna().astype(str).str.strip()
    valid_answers = valid_answers[valid_answers != ""]
    counts = valid_answers.value_counts()
    summary = counts.rename_axis("选项").reset_index(name="频数")
    summary["频率"] = summary["频数"] / int(summary["频数"].sum())
    return summary


def plot_single_choice_question(dataframe: pd.DataFrame, question_prefix: str) -> Path:
    column_name = find_column_by_prefix(dataframe, question_prefix)
    summary = summarize_single_choice(dataframe, column_name)
    valid_sample_size = int(summary["频数"].sum())
    question_number = question_prefix.replace("、", "")
    output_path = OUTPUT_DIR / f"desc_q{question_number}.png"

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    figure.suptitle(f"Q{question_number} 频率分布图", fontsize=16)

    bar_ax = axes[0]
    sns.barplot(data=summary, x="选项", y="频数", hue="选项", dodge=False, ax=bar_ax)
    if bar_ax.get_legend() is not None:
        bar_ax.get_legend().remove()
    bar_ax.set_title("条形图")
    bar_ax.set_xlabel("选项")
    bar_ax.set_ylabel("频数")
    bar_ax.tick_params(axis="x", rotation=20)
    for patch, count, rate in zip(bar_ax.patches, summary["频数"], summary["频率"]):
        bar_ax.annotate(
            f"{count} ({rate:.1%})",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    pie_ax = axes[1]
    pie_labels = [f"{option}\n{rate:.1%}" for option, rate in zip(summary["选项"], summary["频率"])]
    pie_ax.pie(
        summary["频数"],
        labels=pie_labels,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    pie_ax.set_title("饼图")

    figure.text(
        0.5,
        0.02,
        f"统计说明：已自动剔除缺失值；有效样本量 n={valid_sample_size}",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def summarize_multi_choice(dataframe: pd.DataFrame, question_prefix: str) -> tuple[pd.DataFrame, int]:
    question_columns = [column for column in dataframe.columns if str(column).startswith(question_prefix)]
    valid_sample_size = len(dataframe)
    counts = dataframe[question_columns].fillna(0).sum(axis=0).astype(int)
    summary = counts.rename_axis("原始列名").reset_index(name="选择人数")
    summary["选项"] = summary["原始列名"].map(
        lambda value: simplify_option_label(str(value).split("__", 1)[1].strip())
    )
    summary["选择率"] = summary["选择人数"] / valid_sample_size
    summary = summary.sort_values(by="选择率", ascending=False).reset_index(drop=True)
    return summary[["选项", "选择人数", "选择率"]], valid_sample_size


def plot_multi_choice_question(dataframe: pd.DataFrame, question_prefix: str) -> Path:
    summary, valid_sample_size = summarize_multi_choice(dataframe, question_prefix)
    question_number = question_prefix.replace("、", "")
    output_path = OUTPUT_DIR / f"desc_q{question_number}.png"

    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    sns.barplot(data=summary, x="选择率", y="选项", hue="选项", dodge=False, ax=axis)
    if axis.get_legend() is not None:
        axis.get_legend().remove()
    axis.set_title(f"Q{question_number} 多选题选择率分布")
    axis.set_xlabel("选择率")
    axis.set_ylabel("选项")
    axis.set_xlim(0, max(summary["选择率"].max() * 1.15, 0.1))

    for patch, count, rate in zip(axis.patches, summary["选择人数"], summary["选择率"]):
        axis.annotate(
            f"{count} ({rate:.1%})",
            (patch.get_width(), patch.get_y() + patch.get_height() / 2),
            ha="left",
            va="center",
            fontsize=9,
            xytext=(6, 0),
            textcoords="offset points",
        )

    figure.text(
        0.5,
        0.02,
        f"统计说明：选择率=各选项选择人数/有效问卷总数；有效样本量 n={valid_sample_size}",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def simplify_scale_label(label: str, prefix: str) -> str:
    simplified = str(label).replace(prefix, "", 1).strip()
    return simplified.split("—")[-1].strip()


def summarize_q6_ranked(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    score_tracker: dict[str, int] = {}
    valid_sample_size = 0

    for _, row in dataframe[list(Q6_WEIGHTS.keys())].iterrows():
        row_has_valid_choice = False
        for column_name, weight in Q6_WEIGHTS.items():
            option = clean_text_value(row[column_name])
            if option is None:
                continue
            row_has_valid_choice = True
            score_tracker[option] = score_tracker.get(option, 0) + weight
        if row_has_valid_choice:
            valid_sample_size += 1

    summary = pd.DataFrame(
        {"选项": list(score_tracker.keys()), "加权得分": list(score_tracker.values())}
    ).sort_values(by="加权得分", ascending=False, ignore_index=True)
    return summary, valid_sample_size


def plot_q6_ranked_question(dataframe: pd.DataFrame) -> Path:
    summary, valid_sample_size = summarize_q6_ranked(dataframe)
    output_path = OUTPUT_DIR / "desc_q6.png"

    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    sns.barplot(data=summary, x="选项", y="加权得分", hue="选项", dodge=False, ax=axis)
    if axis.get_legend() is not None:
        axis.get_legend().remove()
    axis.set_title("Q6 排序题加权得分分布")
    axis.set_xlabel("选项")
    axis.set_ylabel("加权得分")
    axis.tick_params(axis="x", rotation=18)

    for patch, score in zip(axis.patches, summary["加权得分"]):
        axis.annotate(
            f"{score}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    figure.text(
        0.5,
        0.02,
        f"统计说明：按第1选×3、第2选×2、第3选×1计分；缺失顺位已自动跳过；有效样本量 n={valid_sample_size}",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def summarize_q9_likert(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    q9_columns = [column for column in dataframe.columns if str(column).startswith(Q9_PREFIX)]
    q9_dataframe = dataframe[q9_columns].apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame(
        {
            "维度": [simplify_scale_label(column, Q9_PREFIX) for column in q9_columns],
            "均值": q9_dataframe.mean().values,
            "有效样本量": q9_dataframe.count().values,
        }
    )
    valid_counts = summary["有效样本量"].astype(int)
    return summary, int(valid_counts.min()), int(valid_counts.max())


def plot_q9_likert_question(dataframe: pd.DataFrame) -> Path:
    q9_columns = [column for column in dataframe.columns if str(column).startswith(Q9_PREFIX)]
    q9_dataframe = dataframe[q9_columns].apply(pd.to_numeric, errors="coerce")
    summary, min_valid_sample_size, max_valid_sample_size = summarize_q9_likert(dataframe)
    output_path = OUTPUT_DIR / "desc_q9.png"

    figure = plt.figure(figsize=(16, 8), constrained_layout=True)
    axes = figure.subplots(1, 2)
    figure.suptitle("Q9 李克特量表维度分布", fontsize=16)

    radar_ax = figure.add_subplot(1, 2, 1, polar=True)
    figure.delaxes(axes[0])
    angles = np.linspace(0, 2 * np.pi, len(summary), endpoint=False)
    radar_values = summary["均值"].tolist()
    radar_ax.plot(np.append(angles, angles[0]), radar_values + [radar_values[0]], linewidth=2)
    radar_ax.fill(np.append(angles, angles[0]), radar_values + [radar_values[0]], alpha=0.25)
    radar_ax.set_xticks(angles)
    radar_ax.set_xticklabels(summary["维度"].tolist())
    radar_ax.set_ylim(1, 5)
    radar_ax.set_yticks([1, 2, 3, 4, 5])
    radar_ax.set_title("各维度均值雷达图", pad=20)

    box_ax = axes[1]
    melted = q9_dataframe.rename(columns=dict(zip(q9_columns, summary["维度"]))).melt(
        var_name="维度", value_name="得分"
    )
    melted = melted.dropna(subset=["得分"])
    sns.boxplot(data=melted, x="维度", y="得分", hue="维度", dodge=False, ax=box_ax)
    if box_ax.get_legend() is not None:
        box_ax.get_legend().remove()
    box_ax.set_title("各维度得分箱线图")
    box_ax.set_xlabel("维度")
    box_ax.set_ylabel("得分")
    box_ax.set_ylim(1, 5)
    box_ax.tick_params(axis="x", rotation=20)

    figure.text(
        0.5,
        0.02,
        f"统计说明：各维度按非缺失值自动统计；有效样本量范围 n={min_valid_sample_size}-{max_valid_sample_size}",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def summarize_q10_weights(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    q10_columns = [column for column in dataframe.columns if str(column).startswith(Q10_PREFIX)]
    q10_dataframe = dataframe[q10_columns].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    summary = pd.DataFrame(
        {
            "主体": [simplify_scale_label(column, Q10_PREFIX) for column in q10_columns],
            "平均权重": q10_dataframe.mean().values,
        }
    )
    return summary, len(q10_dataframe)


def plot_q10_weight_question(dataframe: pd.DataFrame) -> Path:
    summary, valid_sample_size = summarize_q10_weights(dataframe)
    output_path = OUTPUT_DIR / "desc_q10.png"

    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    colors = sns.color_palette("muted", n_colors=len(summary))
    left = 0.0
    for (_, row), color in zip(summary.iterrows(), colors):
        axis.barh(["平均权重"], [row["平均权重"]], left=left, color=color, label=row["主体"])
        axis.text(left + row["平均权重"] / 2, 0, f"{row['平均权重']:.1f}%", ha="center", va="center", fontsize=9)
        left += float(row["平均权重"])

    axis.set_title("Q10 评价主体平均权重")
    axis.set_xlabel("平均权重（%）")
    axis.set_ylabel("")
    axis.set_xlim(0, max(100, left * 1.05))
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

    figure.text(
        0.5,
        0.02,
        f"统计说明：已自动剔除缺失行；有效样本量 n={valid_sample_size}；均值权重总和={left:.1f}%",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_descriptive_summary(
    dataframe: pd.DataFrame,
    multi_choice_dataframe: pd.DataFrame,
    q6_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary_frames: list[pd.DataFrame] = []

    for question_prefix in SINGLE_CHOICE_PREFIXES:
        question_number = question_prefix.replace("、", "")
        column_name = find_column_by_prefix(dataframe, question_prefix)
        summary = summarize_single_choice(dataframe, column_name).copy()
        summary.insert(0, "题号", f"Q{question_number}")
        summary.insert(1, "统计类型", "单选题频率")
        summary["有效样本量"] = int(summary["频数"].sum())
        summary = summary.rename(columns={"频率": "统计值", "选项": "项目"})
        summary["统计指标"] = "频率"
        summary_frames.append(summary[["题号", "统计类型", "项目", "统计指标", "统计值", "频数", "有效样本量"]])

    for question_prefix in MULTI_CHOICE_PREFIXES:
        question_number = question_prefix.replace("、", "")
        summary, valid_sample_size = summarize_multi_choice(multi_choice_dataframe, question_prefix)
        summary = summary.copy()
        summary.insert(0, "题号", f"Q{question_number}")
        summary.insert(1, "统计类型", "多选题选择率")
        summary["有效样本量"] = valid_sample_size
        summary = summary.rename(columns={"选项": "项目", "选择率": "统计值", "选择人数": "频数"})
        summary["统计指标"] = "选择率"
        summary_frames.append(summary[["题号", "统计类型", "项目", "统计指标", "统计值", "频数", "有效样本量"]])

    q6_summary, q6_valid_sample_size = summarize_q6_ranked(q6_dataframe)
    q6_summary = q6_summary.copy()
    q6_summary.insert(0, "题号", "Q6")
    q6_summary.insert(1, "统计类型", "排序题加权得分")
    q6_summary["统计指标"] = "加权得分"
    q6_summary["统计值"] = q6_summary["加权得分"]
    q6_summary["频数"] = pd.NA
    q6_summary["有效样本量"] = q6_valid_sample_size
    q6_summary = q6_summary.rename(columns={"选项": "项目"})
    summary_frames.append(q6_summary[["题号", "统计类型", "项目", "统计指标", "统计值", "频数", "有效样本量"]])

    q9_columns = [column for column in dataframe.columns if str(column).startswith(Q9_PREFIX)]
    q9_numeric = dataframe[q9_columns].apply(pd.to_numeric, errors="coerce")
    q9_summary, _, _ = summarize_q9_likert(dataframe)
    q9_distribution = []
    renamed_q9_columns = dict(zip(q9_columns, q9_summary["维度"]))
    for column_name in q9_columns:
        dimension_name = renamed_q9_columns[column_name]
        valid_count = int(q9_numeric[column_name].count())
        mean_value = float(q9_numeric[column_name].mean())
        q9_distribution.append(
            {
                "题号": "Q9",
                "统计类型": "李克特均值",
                "项目": dimension_name,
                "统计指标": "均值",
                "统计值": mean_value,
                "频数": pd.NA,
                "有效样本量": valid_count,
            }
        )
        score_counts = q9_numeric[column_name].value_counts(dropna=True).sort_index()
        for score, count in score_counts.items():
            q9_distribution.append(
                {
                    "题号": "Q9",
                    "统计类型": "李克特分布",
                    "项目": f"{dimension_name} | {int(score)}分",
                    "统计指标": "频率",
                    "统计值": float(count / valid_count),
                    "频数": int(count),
                    "有效样本量": valid_count,
                }
            )
    summary_frames.append(pd.DataFrame(q9_distribution))

    q10_summary, q10_valid_sample_size = summarize_q10_weights(dataframe)
    q10_summary = q10_summary.copy()
    q10_summary.insert(0, "题号", "Q10")
    q10_summary.insert(1, "统计类型", "权重均值")
    q10_summary["统计指标"] = "平均权重"
    q10_summary["统计值"] = q10_summary["平均权重"]
    q10_summary["频数"] = pd.NA
    q10_summary["有效样本量"] = q10_valid_sample_size
    q10_summary = q10_summary.rename(columns={"主体": "项目"})
    summary_frames.append(q10_summary[["题号", "统计类型", "项目", "统计指标", "统计值", "频数", "有效样本量"]])

    descriptive_summary = pd.concat(summary_frames, ignore_index=True)
    descriptive_summary["统计值"] = descriptive_summary["统计值"].astype(float)
    return descriptive_summary


def save_descriptive_summary(
    dataframe: pd.DataFrame,
    multi_choice_dataframe: pd.DataFrame,
    q6_dataframe: pd.DataFrame,
) -> Path:
    descriptive_summary = build_descriptive_summary(dataframe, multi_choice_dataframe, q6_dataframe)
    descriptive_summary.to_csv(SUMMARY_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    return SUMMARY_OUTPUT_FILE


def main() -> None:
    configure_plot_style()
    dataframe = pd.read_csv(INPUT_FILE)
    multi_choice_dataframe = pd.read_csv(MULTI_CHOICE_INPUT_FILE)
    q6_dataframe = pd.read_csv(Q6_INPUT_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for question_prefix in SINGLE_CHOICE_PREFIXES:
        output_path = plot_single_choice_question(dataframe, question_prefix)
        print(f"已生成: {output_path}")

    for question_prefix in MULTI_CHOICE_PREFIXES:
        output_path = plot_multi_choice_question(multi_choice_dataframe, question_prefix)
        print(f"已生成: {output_path}")

    q6_output_path = plot_q6_ranked_question(q6_dataframe)
    print(f"已生成: {q6_output_path}")

    q9_output_path = plot_q9_likert_question(dataframe)
    print(f"已生成: {q9_output_path}")

    q10_output_path = plot_q10_weight_question(dataframe)
    print(f"已生成: {q10_output_path}")

    summary_output_path = save_descriptive_summary(dataframe, multi_choice_dataframe, q6_dataframe)
    print(f"已生成: {summary_output_path}")


if __name__ == "__main__":
    main()