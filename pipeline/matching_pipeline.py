"""
匹配流程管道 - 整合碎片匹配、后处理、可靠性分析和可视化
"""

import os
import pandas as pd
from typing import Optional, Dict, Tuple, List

from matching.protein_classes import Protein, NTermMod, CTermMod
from matching.fragment_finder import get_Nterminal_CM_output, get_Cterminal_CM_output, get_internal_CM_output
from matching.post_processing import post_process
from matching.reliability import calculate_internal_reliability_score
from matching.visualization import seg_map_plot_main, plot_internal_fragment_support_map
from matching.abundance_plot import fragment_abundance_plot_main
from data.mgf_parser import extract_ms_info_from_mgf


# 修饰类型映射表
MODIFICATION_MAP = {
    "n_acetylated": {"name": "Acetylated", "short_name": "ac-", "mass": 42.0105646837, "formula": "C2H2O", "position": "N-term"},
    "n_deaminated": {"name": "Deaminated", "short_name": "deam-", "mass": 0.984016, "formula": "H-1N-1O", "position": "N-term"},
    "n_methylated": {"name": "Methylated", "short_name": "me-", "mass": 14.01565, "formula": "CH2", "position": "N-term"},
    "n_dimethylated": {"name": "Dimethylated", "short_name": "dime-", "mass": 28.0313, "formula": "C2H4", "position": "N-term"},
    "n_formylated": {"name": "Formylated", "short_name": "form-", "mass": 27.9949, "formula": "CO", "position": "N-term"},
    "c_amided": {"name": "Amided", "short_name": "amide-", "mass": -0.984016, "formula": "HN1O-1", "position": "C-term"},
    "c_methylated": {"name": "Methylated (C-term)", "short_name": "cme-", "mass": 14.01565, "formula": "CH2", "position": "C-term"},
    "c_dehydrated": {"name": "Dehydrated", "short_name": "dehyd-", "mass": -18.010565, "formula": "H-2O-1", "position": "C-term"},
}


def get_modification_info(mod_type=None, custom_mod_name=None, custom_mod_mass=None,
                          custom_mod_formula=None, custom_mod_position="N-term"):
    """获取修饰信息"""
    if mod_type and mod_type in MODIFICATION_MAP:
        return MODIFICATION_MAP[mod_type].copy()
    elif custom_mod_name is not None and custom_mod_mass is not None:
        return {
            "name": custom_mod_name,
            "short_name": custom_mod_name,
            "mass": custom_mod_mass,
            "formula": custom_mod_formula,
            "position": custom_mod_position
        }
    return None


def process_single_deepuv_file(
    seq: str,
    final_df: pd.DataFrame,
    msalign_path: str,
    n_term_ion_types: List[str] = None,
    c_term_ion_types: List[str] = None,
    internal_ion_types: List[str] = None,
    ppm: int = 5,
    unloc_mod_df: Optional[pd.DataFrame] = None,
    modification: Optional[str] = None,
    custom_mod_name: Optional[str] = None,
    custom_mod_mass: Optional[float] = None,
    custom_mod_formula: Optional[str] = None,
    custom_mod_position: str = "N-term",
    return_matches: bool = True,
    return_reliability: bool = True,
    output_dir: Optional[str] = None,
    plot_maps: bool = True,
    plot_abundance: bool = True,
    filename_suffix: str = ""
) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    处理单个DeepUV预测结果文件，进行质谱碎片匹配分析

    Parameters:
    -----------
    seq : str
        蛋白质序列
    final_df : pd.DataFrame
        DeepUV预测结果DataFrame（包含pred_label列）
    msalign_path : str
        msalign文件路径
    n_term_ion_types : List[str]
        N端离子类型列表
    c_term_ion_types : List[str]
        C端离子类型列表
    internal_ion_types : List[str]
        内部离子类型列表
    ppm : int
        PPM容差
    unloc_mod_df : Optional[pd.DataFrame]
        非固定修饰DataFrame
    modification : Optional[str]
        修饰类型名称
    custom_mod_name : Optional[str]
        自定义修饰名称
    custom_mod_mass : Optional[float]
        自定义修饰质量
    custom_mod_formula : Optional[str]
        自定义修饰分子式
    return_matches : bool
        是否返回匹配结果
    return_reliability : bool
        是否返回可靠性得分
    output_dir : Optional[str]
        输出目录（用于保存可视化图谱）
    plot_maps : bool
        是否绘制裂解图谱
    plot_abundance : bool
        是否绘制丰度条形图
    filename_suffix : str
        文件名后缀

    Returns:
    --------
    Tuple[Optional[pd.DataFrame], Optional[Dict]]
        (匹配结果DataFrame, 可靠性得分字典)
    """

    # 设置默认离子类型
    if n_term_ion_types is None:
        from utils.constants import N_TERM_ION_TYPES
        n_term_ion_types = N_TERM_ION_TYPES
    if c_term_ion_types is None:
        from utils.constants import C_TERM_ION_TYPES
        c_term_ion_types = C_TERM_ION_TYPES
    if internal_ion_types is None:
        from utils.constants import INTERNAL_ION_TYPES
        internal_ion_types = INTERNAL_ION_TYPES

    # 获取修饰信息
    mod_info = get_modification_info(
        mod_type=modification,
        custom_mod_name=custom_mod_name,
        custom_mod_mass=custom_mod_mass,
        custom_mod_formula=custom_mod_formula,
        custom_mod_position=custom_mod_position  # 新增
    )

    # 构建蛋白质对象
    protein = Protein(seq)
    if mod_info:
        short_name = mod_info["short_name"]
        formula = mod_info.get("formula", None)
        actual_mass = mod_info.get("mass", None)

        if mod_info["position"] == "N-term":
            protein = protein + NTermMod(short_name, formula, actual_mass)
        elif mod_info["position"] == "C-term":
            protein = protein + CTermMod(short_name, formula, actual_mass)
        print(f"✓ Apply: {mod_info['name']} ({mod_info['position']})")

    seqLen = len(protein.seq)
    dataset_name = os.path.splitext(os.path.basename(msalign_path))[0]

    # 读取msalign
    print(f"read msalign: {os.path.basename(msalign_path)}")
    with open(msalign_path, 'r') as f:
        mono_arr, spectra_dfs = extract_ms_info_from_mgf(f)

    if not spectra_dfs:
        raise ValueError("no msalign data！")

    spec_df = spectra_dfs[0].copy()
    spec_df['label'] = final_df['pred_label'].values

    # 碎片匹配分析
    print("Perform fragment ion matching analysis ..")
    n_matches = get_Nterminal_CM_output(spec_df, protein, n_term_ion_types, ppm, unloc_mod_df)
    c_matches = get_Cterminal_CM_output(spec_df, protein, c_term_ion_types, ppm, unloc_mod_df)
    i_matches = get_internal_CM_output(spec_df, protein, internal_ion_types, ppm, unloc_mod_df)

    all_matches = pd.concat([n_matches, c_matches, i_matches], ignore_index=True)

    all_matches_label = all_matches.merge(
        spec_df[['Mass', 'label']],
        left_on='Observed Mass',
        right_on='Mass',
        how='left'
    )

    all_matches_label_fil = post_process(all_matches_label, seqLen)
    all_matches_label_fil['dataset'] = dataset_name

    print(f"✓ Done！ {len(all_matches_label_fil)} matches in total")

    # =========================================================
    # 可视化图谱
    # =========================================================
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # 准备可视化数据（移除额外的列）
        vis_df = all_matches_label_fil.copy()
        for col in ['label', 'dataset']:
            if col in vis_df.columns:
                vis_df = vis_df.drop(columns=[col])

        # 裂解图谱
        if plot_maps:
            print("\nGenerate fragmentation spectra ..")
            try:
                seg_map_plot_main(output_dir, vis_df, protein.seq)
                plot_internal_fragment_support_map(output_dir, vis_df, protein.seq)
                print("✓ Done！")
            except Exception as e:
                print(f"⚠ Error: {e}")

        # 丰度条形图
        if plot_abundance:
            print("\nGenerate abundance bar chart...")
            try:
                fragment_abundance_plot_main(
                    output_dir,
                    vis_df,
                    protein.seq,
                    filename_suffix=filename_suffix
                )
                print("✓ Done！")
            except Exception as e:
                print(f"⚠ Error: {e}")

    # =========================================================
    # 可靠性分析
    # =========================================================
    reliability_scores = calculate_internal_reliability_score(all_matches_label_fil, seqLen)
    reliability_scores["dataset"] = dataset_name

    # =========================================================
    # 返回结果
    # =========================================================
    result_matches = all_matches_label_fil if return_matches else None
    result_reliability = reliability_scores if return_reliability else None

    if result_matches is not None:
        cols_to_drop = ['label', 'dataset']
        result_matches = result_matches.drop(
            columns=[c for c in cols_to_drop if c in result_matches.columns]
        )

    return result_matches, result_reliability