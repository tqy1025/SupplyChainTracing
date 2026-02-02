import pandas as pd
import os
from tqdm import tqdm


def validate_composite_pattern(input_file, output_file):
    print(f"[-] 正在读取文件: {input_file}")
    if not os.path.exists(input_file):
        print(f"[!] 错误: 文件不存在 - {input_file}")
        return

    # 1. 读取数据 (Filtered_Implicit_MultiVendor_Flows)
    df = pd.read_csv(input_file)

    # 提取 Component_ID 和 Device 用于集合运算
    # 确保去除重复项，保证集合纯度
    df_clean = df[['Component_ID', 'Device']].dropna().drop_duplicates()

    print("[-] 正在构建组件-设备集合映射...")

    # 2. 构建 D_k 集合
    # 格式: { 'Comp_1': {'Dev_A', 'Dev_B'}, ... }
    cluster_device_map = df_clean.groupby('Component_ID')['Device'].apply(set).to_dict()

    # 按设备数量从大到小排序，因为“Suite”通常包含更多设备
    # 这有助于逻辑理解，但对集合运算结果无影响
    sorted_clusters = sorted(cluster_device_map.items(), key=lambda x: len(x[1]), reverse=True)

    total_clusters = len(sorted_clusters)
    print(f"    > 待分析组件数: {total_clusters}")
    print("[-] 开始执行复合模式验证 (Composite Pattern Emergence)...")

    validation_results = []

    # 3. 遍历每个组件作为潜在的 "Suite" (C_suite)
    for suite_id, suite_devs in tqdm(sorted_clusters, desc="Analyzing Composites"):

        # 寻找潜在的子组件 (Sub-components)
        # 条件: D_sub 是 D_suite 的真子集 (D_sub < D_suite)
        # 且 D_sub 不为空
        sub_clusters = []
        sub_cluster_ids = []

        for other_id, other_devs in cluster_device_map.items():
            if suite_id == other_id:
                continue

            # 使用真子集判断 (<) 而不是子集 (<=)，防止自己重构自己
            if other_devs < suite_devs:
                sub_clusters.append(other_devs)
                sub_cluster_ids.append(other_id)

        # 4. 验证条件: Union(D_sub) == D_suite
        if not sub_clusters:
            is_composite = False
            reconstruction_ratio = 0.0
        else:
            # 计算所有子集的并集
            union_of_subs = set().union(*sub_clusters)

            # 检查是否完全相等
            if union_of_subs == suite_devs:
                is_composite = True
                reconstruction_ratio = 1.0
            else:
                is_composite = False
                # 计算覆盖率 (可选，用于分析)
                reconstruction_ratio = len(union_of_subs) / len(suite_devs)

        # 记录结果
        # 如果是复合模式，记录构成的子组件ID
        constituents_str = ""
        if is_composite:
            # 仅记录前10个 ID 避免 CSV 过长
            constituents_str = ";".join(sub_cluster_ids[:10])
            if len(sub_cluster_ids) > 10:
                constituents_str += "..."

        validation_results.append({
            'Component_ID': suite_id,
            'Device_Count': len(suite_devs),
            'Is_Composite_Suite': is_composite,
            'Sub_Cluster_Count': len(sub_clusters),
            'Reconstruction_Ratio': reconstruction_ratio,  # 1.0 表示完美重构
            'Constituent_Clusters': constituents_str
        })

    # 5. 合并结果
    print("[-] 正在生成最终报告...")
    df_val = pd.DataFrame(validation_results)

    # 统计
    num_composite = df_val['Is_Composite_Suite'].sum()
    print(f"    > 发现复合模式组件 (Suite): {num_composite}")

    # Merge 回原始数据 (左连接)
    # 这里的 df 是原始 filtered 文件
    df_final = pd.merge(df, df_val, on='Component_ID', how='left')

    # 6. 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"[-] 结果已保存至: {output_file}")

    # 7. 打印样本
    if num_composite > 0:
        print("-" * 60)
        print("样本: 复合模式组件 (Composite Suites)")
        sample = df_val[df_val['Is_Composite_Suite'] == True].head(3)
        for _, row in sample.iterrows():
            print(f"Suite ID: {row['Component_ID']} (Devices: {row['Device_Count']})")
            print(f"  Constructed from {row['Sub_Cluster_Count']} sub-clusters")
            print(f"  Constituents: {row['Constituent_Clusters']}")
        print("-" * 60)


# --- 执行配置 ---
if __name__ == "__main__":
    # base_dir = '../DTW_1_12'

    # 输入: 上一步生成的 Filtered 文件
    input_csv = f'../4.2.Clusters_Final_Latent.csv'

    # 输出: 包含复合模式验证结果的文件
    output_csv = f'../5.2.Network_Flows_Validated_Composite_v1.20.csv'

    validate_composite_pattern(input_file=input_csv, output_file=output_csv)

