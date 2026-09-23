import numpy as np
import pandas as pd


def get_terminal_cleavages(matches_df, seqLen):
    """提取终端裂解位点"""
    terminal_mask = (matches_df["Start AA"] == 1) | (matches_df["End AA"] == seqLen)
    terminal_df = matches_df[terminal_mask].copy()
    terminal_cleavages = set()

    for _, row in terminal_df.iterrows():
        start = int(row["Start AA"])
        end = int(row["End AA"])
        if start == 1:
            terminal_cleavages.add(end)
        if end == seqLen:
            terminal_cleavages.add(start - 1)

    return terminal_cleavages


def classify_internal_support(row, terminal_cleavages):
    """分类内部碎片支持类型"""
    start = int(row["Start AA"])
    end = int(row["End AA"])
    left_cleavage = start - 1
    right_cleavage = end

    left_support = left_cleavage in terminal_cleavages
    right_support = right_cleavage in terminal_cleavages

    if left_support and right_support:
        return "two_side"
    elif left_support or right_support:
        return "one_side"
    else:
        return "unsupported"


def calculate_internal_reliability_score(matches_df, seqLen, epsilon=1e-6):
    """计算内部可靠性得分 (IRS) """
    total_possible_cleavages = seqLen - 1

    # 终端碎片
    terminal_mask = (matches_df["Start AA"] == 1) | (matches_df["End AA"] == seqLen)
    terminal_df = matches_df[terminal_mask].copy()
    internal_df = matches_df[~terminal_mask].copy()

    # 终端裂解位点
    terminal_cleavages = get_terminal_cleavages(matches_df, seqLen)
    terminal_cleavage_coverage = len(terminal_cleavages) / total_possible_cleavages

    # 内部裂解位点
    internal_cleavages = set()
    for _, row in internal_df.iterrows():
        start = int(row["Start AA"])
        end = int(row["End AA"])
        internal_cleavages.add(start - 1)
        internal_cleavages.add(end)
    internal_cleavage_coverage = len(internal_cleavages) / total_possible_cleavages

    # 总裂解位点
    total_cleavages = terminal_cleavages | internal_cleavages
    total_cleavage_coverage = len(total_cleavages) / total_possible_cleavages

    total_internal = len(internal_df)
    if total_internal == 0:
        return {
            "terminal_cleavage_coverage": terminal_cleavage_coverage,
            "internal_cleavage_coverage": internal_cleavage_coverage,
            "total_cleavage_coverage": total_cleavage_coverage,
            "one_side_internal_ratio": 0,
            "two_side_internal_ratio": 0,
            "unsupported_internal_ratio": 0,
            "IRS": 0.0
        }

    # 统计支持情况
    one_side_supported = 0
    two_side_supported = 0
    unsupported_internal = 0

    for _, row in internal_df.iterrows():
        support_type = classify_internal_support(row, terminal_cleavages)
        if support_type == "two_side":
            two_side_supported += 1
        elif support_type == "one_side":
            one_side_supported += 1
        else:
            unsupported_internal += 1

    one_side_internal_ratio = one_side_supported / total_internal
    two_side_internal_ratio = two_side_supported / total_internal
    unsupported_internal_ratio = unsupported_internal / total_internal


    total_val = total_possible_cleavages

    irs_v7_base = terminal_cleavage_coverage + 1.0 * two_side_internal_ratio * (1 - terminal_cleavage_coverage)
    # irs_v7_base = np.sqrt(two_side_internal_ratio) * np.exp(
    #     -np.square(1 - terminal_cleavage_coverage)
    #     / (two_side_internal_ratio + epsilon)
    # )
    IRS = irs_v7_base * total_val
    # =============================================================

    return {
        "terminal_fragment_count": len(terminal_df),
        "internal_fragment_count": total_internal,
        "terminal_cleavage_count": len(terminal_cleavages),
        "internal_cleavage_count": len(internal_cleavages),
        "total_cleavage_count": len(total_cleavages),
        "terminal_cleavage_coverage": terminal_cleavage_coverage,
        "internal_cleavage_coverage": internal_cleavage_coverage,
        "total_cleavage_coverage": total_cleavage_coverage,
        "one_side_supported_internal": one_side_supported,
        "two_side_supported_internal": two_side_supported,
        "unsupported_internal": unsupported_internal,
        "one_side_internal_ratio": one_side_internal_ratio,
        "two_side_internal_ratio": two_side_internal_ratio,
        "unsupported_internal_ratio": unsupported_internal_ratio,
        "IRS": IRS
    }