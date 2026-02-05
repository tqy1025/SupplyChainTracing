import os
import pandas as pd
from tqdm import tqdm

# ================= 配置 =================
# Suite 候选（大组件）来自 4.2（最终 latent 流级表）
SUITE_FLOW_FILE = "./4.2.Clusters_Final_Latent_.csv"

# Sub-component 候选可以来自 3.1 / 4.1 / 4.2 任意文件（并集）
SUBCOMP_FILES = [
    "./3.1.Flows_Consistent_Strict_cap100_nopad.csv",
    "./4.1.Clusters_Validated_by_LLM_.csv",
    "./4.2.Clusters_Final_Latent_.csv",
]

# 输出（流级）
OUTPUT_FLOW_LEVEL = "./5.2.Network_Flows_Validated_Composite_MultiSource_FLOWLEVEL.csv"

# 输出中 Constituent_Clusters 最多保留多少个子组件 ID（避免列太长）
MAX_CONSTITUENTS = 30


# ================= 工具函数 =================
def read_csv_smart(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def build_component_device_map(df: pd.DataFrame, comp_col="Component_ID", dev_col="Device"):
    """
    从流级表构建 {Component_ID -> set(Device)} 映射
    """
    if comp_col not in df.columns or dev_col not in df.columns:
        raise ValueError(f"Missing columns: require [{comp_col}, {dev_col}] in a source file.")

    tmp = df[[comp_col, dev_col]].dropna().drop_duplicates().copy()
    tmp[comp_col] = tmp[comp_col].astype(str).str.strip()
    tmp[dev_col] = tmp[dev_col].astype(str).str.strip()

    comp_dev_map = tmp.groupby(comp_col)[dev_col].apply(set).to_dict()
    return comp_dev_map


def composite_validate(
    suite_map: dict,
    subcomp_map_union: dict,
    max_constituents: int = 30
) -> pd.DataFrame:
    """
    对 suite_map 中的每个组件做 Composite Suite 验证：
      - 找所有严格真子集 subcomponent：D_sub < D_suite
      - 判定 Union(D_subs) 是否 == D_suite
    subcomponent 来自 subcomp_map_union（多源并集）
    """
    # 为了加速：把 subcomp 项提取成 list
    sub_items = list(subcomp_map_union.items())

    # 套件按设备数从大到小遍历
    suites_sorted = sorted(suite_map.items(), key=lambda x: len(x[1]), reverse=True)

    results = []

    for suite_id, suite_devs in tqdm(suites_sorted, desc="Analyzing Composite Suites"):
        # 空/单设备不可能构成 composite（按你的定义）
        if not suite_devs or len(suite_devs) <= 1:
            results.append({
                "Component_ID": suite_id,
                "Is_Composite_Suite": False,
                "Sub_Cluster_Count": 0,
                "Reconstruction_Ratio": 0.0,
                "Constituent_Clusters": ""
            })
            continue

        sub_clusters = []
        sub_ids = []

        # 从多源子组件里找真子集
        for other_id, other_devs in sub_items:
            if other_id == suite_id:
                continue
            # 真子集：other_devs < suite_devs
            if other_devs and other_devs < suite_devs:
                sub_clusters.append(other_devs)
                sub_ids.append(other_id)

        if not sub_clusters:
            is_composite = False
            recon_ratio = 0.0
        else:
            union_of_subs = set().union(*sub_clusters)
            if union_of_subs == suite_devs:
                is_composite = True
                recon_ratio = 1.0
            else:
                is_composite = False
                recon_ratio = len(union_of_subs) / len(suite_devs)

        constituents_str = ""
        if is_composite:
            # 为了可读性：优先输出设备数较大的子组件（更“解释力强”）
            # 这里对子组件按 size 降序排序后截断
            sub_sorted = sorted(zip(sub_ids, sub_clusters), key=lambda x: len(x[1]), reverse=True)
            top_ids = [sid for sid, _ in sub_sorted[:max_constituents]]
            constituents_str = ";".join(top_ids)
            if len(sub_sorted) > max_constituents:
                constituents_str += "..."

        results.append({
            "Component_ID": suite_id,
            "Is_Composite_Suite": is_composite,
            "Sub_Cluster_Count": len(sub_clusters),
            "Reconstruction_Ratio": float(recon_ratio),
            "Constituent_Clusters": constituents_str
        })

    return pd.DataFrame(results)


# ================= 主流程 =================
def main():
    print("[-] Loading suite (flow-level) file for output...")
    df_suite_flows = read_csv_smart(SUITE_FLOW_FILE)

    # Suite 候选：来自 4.2 的 Component->Device set
    print("[-] Building suite component-device sets from 4.2...")
    suite_map = build_component_device_map(df_suite_flows, comp_col="Component_ID", dev_col="Device")
    print(f"    > Suite candidates: {len(suite_map)} components")

    # 多源子组件并集
    print("[-] Building union sub-component sets from 3 sources...")
    sub_union = {}
    for f in SUBCOMP_FILES:
        df_src = read_csv_smart(f)
        m = build_component_device_map(df_src, comp_col="Component_ID", dev_col="Device")

        # 合并到 union：同一个 Component_ID 如果在不同文件出现，取设备集合的并集（更保守）
        for cid, devs in m.items():
            if cid not in sub_union:
                sub_union[cid] = set(devs)
            else:
                sub_union[cid] |= set(devs)

    print(f"    > Union sub-components: {len(sub_union)} components")

    # Composite Suite 验证（组件级）
    print("[-] Validating composite suites (component-level)...")
    df_suite_val = composite_validate(suite_map, sub_union, max_constituents=MAX_CONSTITUENTS)

    num_composite = int(df_suite_val["Is_Composite_Suite"].sum())
    print(f"    > Composite suites found: {num_composite}/{len(df_suite_val)}")

    # 回灌到流级：按 Component_ID merge
    print("[-] Backfilling suite validation to each flow row...")
    df_out = df_suite_flows.merge(df_suite_val, on="Component_ID", how="left", validate="m:1")

    # 没匹配到的（理论上不该发生，因为 suite_map 来源于 df_suite_flows）
    for c in ["Is_Composite_Suite", "Sub_Cluster_Count", "Reconstruction_Ratio", "Constituent_Clusters"]:
        if c in df_out.columns:
            if c == "Is_Composite_Suite":
                df_out[c] = df_out[c].fillna(False)
            elif c == "Sub_Cluster_Count":
                df_out[c] = df_out[c].fillna(0).astype(int)
            elif c == "Reconstruction_Ratio":
                df_out[c] = df_out[c].fillna(0.0).astype(float)
            else:
                df_out[c] = df_out[c].fillna("")

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FLOW_LEVEL) or ".", exist_ok=True)
    df_out.to_csv(OUTPUT_FLOW_LEVEL, index=False, encoding="utf-8-sig")
    print(f"✅ Saved flow-level composite validation to: {OUTPUT_FLOW_LEVEL}")
    print(f"    Rows: {len(df_out)}  |  Cols: {len(df_out.columns)}")

    # 可选：输出一些统计
    print("-" * 60)
    print("Composite Suite Stats (flow-level backfilled, component-level computed)")
    print(f"Total suites: {len(df_suite_val)}")
    print(f"Composite suites: {num_composite}")
    if len(df_suite_val) > 0:
        print(f"Composite rate: {num_composite / len(df_suite_val) * 100:.2f}%")
    print("-" * 60)


if __name__ == "__main__":
    main()
