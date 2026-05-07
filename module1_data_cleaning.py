from typing import Optional
from pathlib import Path

import pandas as pd


FILE_PATH = r"d:\学生问卷172份.xlsx"
SHEET_NAME = "Sheet1"
TIME_COLUMN = "所用时间"
MULTI_SELECT_PREFIXES = ["3、", "7、", "8、", "11、", "12、", "13、"]
MULTI_SELECT_DELIMITER = "┋"
Q6_PREFIX = "6、"
RANK_DELIMITER = "→"
Q9_PREFIX = "9、"
Q10_PREFIX = "10、"
Q15_PREFIX = "15、"
Q15_VALID_VALUE = '基本同意 ←请选择此项'
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_duration_to_seconds(duration_text: object) -> Optional[int]:
    if pd.isna(duration_text):
        return None

    duration_text = str(duration_text).strip()
    if duration_text.endswith("秒"):
        return int(duration_text[:-1])

    return None


def split_multi_select_answers(answer_text: object) -> list:
    if pd.isna(answer_text):
        return []

    answer_text = str(answer_text).strip()
    if not answer_text:
        return []

    return [item.strip() for item in answer_text.split(MULTI_SELECT_DELIMITER) if item.strip()]


def split_ranked_answers(answer_text: object) -> list:
    if pd.isna(answer_text):
        return []

    answer_text = str(answer_text).strip()
    if not answer_text:
        return []

    return [item.strip() for item in answer_text.split(RANK_DELIMITER) if item.strip()]


def main() -> None:
    dataframe = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    q6_column = next(column for column in dataframe.columns if str(column).startswith(Q6_PREFIX))
    q9_columns = [column for column in dataframe.columns if str(column).startswith(Q9_PREFIX)]
    q10_columns = [column for column in dataframe.columns if str(column).startswith(Q10_PREFIX)]
    q15_column = next(column for column in dataframe.columns if str(column).startswith(Q15_PREFIX))
    multi_select_columns = [
        next(column for column in dataframe.columns if str(column).startswith(prefix))
        for prefix in MULTI_SELECT_PREFIXES
    ]
    filtered_dataframe = dataframe[dataframe[q15_column] == Q15_VALID_VALUE]
    removed_count = len(dataframe) - len(filtered_dataframe)
    duration_seconds = filtered_dataframe[TIME_COLUMN].apply(parse_duration_to_seconds)
    ranked_answers = filtered_dataframe[q6_column].apply(split_ranked_answers)
    ranked_counts = ranked_answers.apply(len)
    first_choices = ranked_answers.apply(lambda items: items[0] if len(items) > 0 else None)
    second_choices = ranked_answers.apply(lambda items: items[1] if len(items) > 1 else None)
    third_choices = ranked_answers.apply(lambda items: items[2] if len(items) > 2 else None)
    q9_row_sums = filtered_dataframe[q9_columns].sum(axis=1)
    total_score_matches = q9_row_sums == filtered_dataframe["总分"]
    total_score_match_rows = int(total_score_matches.sum())
    total_score_mismatch_rows = int((~total_score_matches).sum())
    q10_row_sums = filtered_dataframe[q10_columns].sum(axis=1)
    q10_valid_rows = int((q10_row_sums == 100).sum())
    q10_invalid_rows = int((q10_row_sums != 100).sum())
    cleaned_dataframe = filtered_dataframe.copy()
    cleaned_dataframe["所用时间_秒"] = duration_seconds

    q6_ranked_dataframe = pd.DataFrame(
        {
            "序号": filtered_dataframe["序号"].values,
            "Q6_第1选": first_choices.values,
            "Q6_第2选": second_choices.values,
            "Q6_第3选": third_choices.values,
        }
    )

    multi_choice_binary_dataframe = pd.DataFrame({"序号": filtered_dataframe["序号"].values})
    for column in multi_select_columns:
        expanded_answers = filtered_dataframe[column].apply(split_multi_select_answers)
        unique_options = sorted({option for answers in expanded_answers for option in answers})
        for option in unique_options:
            binary_column_name = f"{column}__{option}"
            multi_choice_binary_dataframe[binary_column_name] = expanded_answers.apply(
                lambda answers: int(option in answers)
            )

    invalid_reason_counts = {
        "Q15未选择指定选项": removed_count,
        "Q10权重和不为100": q10_invalid_rows,
        "总分与Q9求和不一致": total_score_mismatch_rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_data_path = OUTPUT_DIR / "cleaned_data.csv"
    q6_ranked_path = OUTPUT_DIR / "q6_ranked.csv"
    multi_choice_binary_path = OUTPUT_DIR / "multi_choice_binary.csv"
    cleaning_report_path = OUTPUT_DIR / "cleaning_report.txt"

    cleaned_dataframe.to_csv(cleaned_data_path, index=False, encoding="utf-8-sig")
    q6_ranked_dataframe.to_csv(q6_ranked_path, index=False, encoding="utf-8-sig")
    multi_choice_binary_dataframe.to_csv(multi_choice_binary_path, index=False, encoding="utf-8-sig")

    report_lines = [
        "清洗日志",
        f"原始行数: {len(dataframe)}",
        f"有效行数: {len(filtered_dataframe)}",
        f"剔除行数: {removed_count}",
        "各类无效原因统计:",
    ]
    report_lines.extend([f"- {reason}: {count}" for reason, count in invalid_reason_counts.items()])
    cleaning_report_path.write_text("\n".join(report_lines), encoding="utf-8-sig")

    print("1) 数据行列数")
    print(f"行数: {dataframe.shape[0]}, 列数: {dataframe.shape[1]}")
    print()

    print("2) 列名清单")
    print(dataframe.columns.tolist())
    print()

    print("3) Q15有效性过滤")
    print(f"过滤前行数: {len(dataframe)}")
    print(f"过滤后行数: {len(filtered_dataframe)}")
    print(f"剔除行数: {removed_count}")
    print()

    print("4) 所用时间转数字")
    print("本步服务交付目标: 为后续数据质量校验保留可计算的时长数值列")
    print(f"成功转换行数: {duration_seconds.notna().sum()}")
    print(f"转换失败行数: {duration_seconds.isna().sum()}")
    print(f"转换后样例(前5条): {duration_seconds.head(5).tolist()}")
    print()

    print("5) 多选题展开")
    print("本步服务交付目标: 为Q3、Q7、Q8、Q11、Q12、Q13的词频统计准备可拆分的选项列表")
    for column in multi_select_columns:
        expanded_answers = filtered_dataframe[column].apply(split_multi_select_answers)
        expanded_count = expanded_answers.apply(len)
        total_options = int(expanded_count.sum())
        non_empty_rows = int((expanded_count > 0).sum())
        sample_expanded = expanded_answers[expanded_count > 0].head(1).tolist()

        print(f"列名: {column}")
        print(f"非空展开行数: {non_empty_rows}")
        print(f"展开后总选项数: {total_options}")
        print(f"单行最多展开项数: {int(expanded_count.max())}")
        print(f"展开样例: {sample_expanded}")
        print()

    print("6) Q6拆分第1/2/3选")
    print("本步服务交付目标: 为排序题的加权偏好得分准备可直接使用的顺位列")
    print(f"非空排序行数: {int((ranked_counts > 0).sum())}")
    print(f"第1选数量: {int(first_choices.notna().sum())}")
    print(f"第2选数量: {int(second_choices.notna().sum())}")
    print(f"第3选数量: {int(third_choices.notna().sum())}")
    print(f"单行最多拆分项数: {int(ranked_counts.max())}")
    print(f"拆分样例(前3行): {ranked_answers.head(3).tolist()}")
    print()

    print("7) Q10权重校验")
    print("本步服务交付目标: 确认5个评价主体的权重分配满足逐行总和为100的统计前提")
    print(f"Q10列数: {len(q10_columns)}")
    print(f"逐行权重和=100的行数: {q10_valid_rows}")
    print(f"逐行权重和!=100的行数: {q10_invalid_rows}")
    print(f"逐行权重和最小值: {int(q10_row_sums.min())}")
    print(f"逐行权重和最大值: {int(q10_row_sums.max())}")
    print(f"逐行权重和样例(前5行): {q10_row_sums.head(5).tolist()}")
    print()

    print("8) 总分一致性校验")
    print("本步服务交付目标: 确认原始总分列与Q9十个维度求和一致，保证量表相关统计口径可靠")
    print(f"Q9列数: {len(q9_columns)}")
    print(f"总分一致行数: {total_score_match_rows}")
    print(f"总分不一致行数: {total_score_mismatch_rows}")
    print(f"Q9求和最小值: {int(q9_row_sums.min())}")
    print(f"Q9求和最大值: {int(q9_row_sums.max())}")
    print(
        "总分与Q9求和样例(前5行): "
        f"{list(zip(filtered_dataframe['总分'].head(5).tolist(), q9_row_sums.head(5).tolist()))}"
    )
    print()

    print(f"output/cleaned_data.csv 已保存")
    print(f"output/q6_ranked.csv 已保存")
    print(f"output/multi_choice_binary.csv 已保存")
    print(f"output/cleaning_report.txt 已保存")


if __name__ == "__main__":
    main()