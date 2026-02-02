import pandas as pd
import os
from tqdm import tqdm


def validate_multi_cluster_cooccurrence(input_file, output_file):
    print(f"[-] 正在读取文件: {input_file}")
    if not os.path.exists(input_file):
        print(f"[!] 错误: 文件不存在 - {input_file}")
        return

    # 1. 读取数据
    df = pd.read_csv(input_file)

    # 只需要 Component_ID 和 Device 列来构建集合
    df_clean = df[['Component_ID', 'Device']].dropna().drop_duplicates()

    print("[-] 正在构建聚类-设备集合映射...")

    # 2. 构建 D_k 集合
    cluster_device_map = df_clean.groupby('Component_ID')['Device'].apply(set).to_dict()

    cluster_ids = list(cluster_device_map.keys())
    total_clusters = len(cluster_ids)

    print(f"    > 识别到 {total_clusters} 个唯一的 Component_ID")
    print("[-] 开始执行共现验证 (Multi-Cluster Co-occurrence)...")

    # 存储验证结果
    validation_results = []

    # 3. 遍历每个聚类 C_k 进行验证
    for cid_k in tqdm(cluster_ids, desc="Validating Clusters"):
        devices_k = cluster_device_map[cid_k]

        match_count = 0
        matching_clusters = []

        # 遍历所有聚类 C_j
        for cid_j in cluster_ids:
            devices_j = cluster_device_map[cid_j]

            # 判断 D_k 是否是 D_j 的子集
            if devices_k.issubset(devices_j):
                match_count += 1
                if cid_j != cid_k:
                    matching_clusters.append(cid_j)

        # 4. 应用验证条件: count > 1
        is_validated = match_count > 1

        validation_results.append({
            'Component_ID': cid_k,
            'Device_Count': len(devices_k),
            'Is_Validated': is_validated,
            'Co_occurrence_Count': match_count,
            'Related_Clusters': ";".join(matching_clusters[:5]) + ("..." if len(matching_clusters) > 5 else "")
        })

    # 5. 将验证结果合并回原数据
    print("[-] 正在合并验证结果...")
    df_validation = pd.DataFrame(validation_results)

    num_validated = df_validation['Is_Validated'].sum()
    print(f"    > 验证通过的组件数: {num_validated} / {total_clusters} ({num_validated / total_clusters:.1%})")

    # Merge
    df_final = pd.merge(df, df_validation[['Component_ID', 'Is_Validated', 'Co_occurrence_Count', 'Related_Clusters']],
                        on='Component_ID', how='left')

    # 6. 保存结果 (修正了此处)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_final.to_csv(output_file, index=False)
    print(f"[-] 结果已保存至: {output_file}")

    if num_validated > 0:
        print("-" * 50)
        print("样本: 验证通过的高置信度组件 (High Confidence)")
        sample = df_validation[df_validation['Is_Validated'] == True].head(3)
        print(sample[['Component_ID', 'Device_Count', 'Co_occurrence_Count', 'Related_Clusters']])
        print("-" * 50)


# --- 执行配置 ---
if __name__ == "__main__":
    # 基础路径
    # base_dir = '../../DTW_1_12'  # 注意：根据您的报错日志，这里似乎是 ../../DTW_1_12

    # 输入文件
    input_csv = f'../5.1.Remaining_Unique_LLM_judge_False.csv'

    # 输出文件
    output_csv = f'../6.1.Network_Flows_Validated_Cooccurrence_v1.20.csv'


    validate_multi_cluster_cooccurrence(input_csv, output_csv)
