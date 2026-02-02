# import pandas as pd
# import pickle
# import os
# from itertools import combinations
# from tqdm import tqdm
#
#
# def calculate_known_relationship_reinforcement_corrected(csv_file, pkl_file, output_file):
#     print(f"[-] 正在读取流量数据: {csv_file}")
#     if not os.path.exists(csv_file) or not os.path.exists(pkl_file):
#         print("[!] 错误: 输入文件不存在。")
#         return
#
#     # 1. 加载 CSV 数据
#     df = pd.read_csv(csv_file)
#
#     # 预处理厂商列: 填充空值并转为字符串，防止比较出错
#     df['device_vendor'] = df['device_vendor'].fillna('unknown').astype(str).str.strip()
#
#     # 提取需要的列 (Component_ID, Device, device_vendor)
#     # drop_duplicates 确保每个设备在一个组件里只出现一次
#     df_clean = df[['Component_ID', 'Device', 'device_vendor']].drop_duplicates()
#
#     print(f"[-] 正在加载隐性关系库: {pkl_file}")
#     with open(pkl_file, 'rb') as f:
#         implicit_data = pickle.load(f)
#
#     # 2. 构建隐性关系对集合 (Implicit Pairs)
#     print("[-] 正在构建隐性关系对集合 (Implicit Pairs)...")
#     implicit_pairs_set = set()
#
#     # 遍历 Pickle 结构: Layer -> Target -> DeviceSet
#     for layer, targets in implicit_data.items():
#         for target, devices in targets.items():
#             if len(devices) < 2:
#                 continue
#
#             # 生成该目标下的所有设备对并存入集合
#             for d1, d2 in combinations(devices, 2):
#                 # 使用 sorted tuple 保证 (A, B) 和 (B, A) 一致
#                 pair = tuple(sorted((d1, d2)))
#                 implicit_pairs_set.add(pair)
#
#     print(f"    > 已构建 {len(implicit_pairs_set)} 对唯一的隐性关系对")
#
#     # 3. 遍历每个组件计算 Ratio
#     print("[-] 开始计算每个组件的已知关系重合度 (Include Explicit & Implicit)...")
#
#     results = []
#
#     # 按组件分组
#     grouped = df_clean.groupby('Component_ID')
#
#     for cid, group in tqdm(grouped, desc="Processing Clusters"):
#         # 获取该组件内的所有设备及其厂商
#         # 格式: [('Device_A', 'Amazon'), ('Device_B', 'Google'), ...]
#         dev_vendor_list = list(zip(group['Device'], group['device_vendor']))
#
#         total_pairs_count = 0  # 分母: 所有对
#         matched_pairs_count = 0  # 分子: 同厂商 或 隐性关联
#
#         # 只有1个设备的组件无法形成对，Ratio 为 0 或 N/A (这里设为0)
#         if len(dev_vendor_list) < 2:
#             results.append({
#                 'Component_ID': cid,
#                 'Device_Count': len(dev_vendor_list),
#                 'Total_Pairs': 0,
#                 'Matched_Pairs': 0,
#                 'Reinforcement_Ratio': 0.0
#             })
#             continue
#
#         # 生成组件内所有可能的两两组合
#         for i in range(len(dev_vendor_list)):
#             for j in range(i + 1, len(dev_vendor_list)):
#                 dev1, vendor1 = dev_vendor_list[i]
#                 dev2, vendor2 = dev_vendor_list[j]
#
#                 total_pairs_count += 1
#
#                 is_matched = False
#
#                 # --- 检查 1: 显性关系 (Explicit) - 同厂商 ---
#                 # 排除 'unknown' 厂商，如果两个都是 unknown 不算已知关系
#                 if vendor1 == vendor2 and vendor1.lower() != 'unknown' and vendor1 != '':
#                     is_matched = True
#
#                 # --- 检查 2: 隐性关系 (Implicit) - 共享第三方 ---
#                 # 如果还未匹配 (即不同厂商)，则检查隐性库
#                 if not is_matched:
#                     current_pair = tuple(sorted((dev1, dev2)))
#                     if current_pair in implicit_pairs_set:
#                         is_matched = True
#
#                 if is_matched:
#                     matched_pairs_count += 1
#
#         # 计算比率
#         ratio = matched_pairs_count / total_pairs_count if total_pairs_count > 0 else 0.0
#
#         results.append({
#             'Component_ID': cid,
#             'Device_Count': len(dev_vendor_list),
#             'Total_Pairs': total_pairs_count,
#             'Matched_Pairs': matched_pairs_count,
#             'Reinforcement_Ratio': ratio
#         })
#
#     # 4. 合并结果
#     print("[-] 正在合并结果...")
#     df_results = pd.DataFrame(results)
#
#     # Merge 回原始宽表 (左连接)
#     df_final = pd.merge(df, df_results[['Component_ID', 'Total_Pairs', 'Matched_Pairs', 'Reinforcement_Ratio']],
#                         on='Component_ID', how='left')
#
#     # 5. 保存
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     df_final.to_csv(output_file, index=False)
#
#     print("-" * 60)
#     print(f"计算完成！结果已保存至: {output_file}")
#
#     # 打印统计信息
#     # 这里的阈值 0.7 仅供参考打印，不影响文件保存
#     high_conf_df = df_results[df_results['Reinforcement_Ratio'] > 0.7]
#     print(f"    > Reinforcement Ratio > 0.7 的组件数: {len(high_conf_df)} / {len(df_results)}")
#
#     if not high_conf_df.empty:
#         print("\n样本 (Ratio > 0.7):")
#         print(high_conf_df[['Component_ID', 'Total_Pairs', 'Matched_Pairs', 'Reinforcement_Ratio']].head(3))
#     print("-" * 60)
#
#
# # --- 执行配置 ---
# if __name__ == "__main__":
#     # base_dir = '../DTW_1_12'
#
#     # 输入 1: 经过 Filtered 的多厂商流量表
#     # 注意：这里我们使用 "Filtered" 表，意味着所有组件已经是跨厂商的了
#     # 但这个逻辑同样适用于未过滤的表。
#     input_csv = f'./A1.Filtered_Implicit_MultiVendor_Flows_v1.12.csv'
#
#     # 输入 2: 隐性关系 Pickle
#     input_pkl = '../2_Implicit_Identification/Implicit_Relationships.pkl'
#
#     # 输出
#     output_csv = f'./A4.Network_Flows_Validated_Reinforcement_v1.12.csv'
#
#     calculate_known_relationship_reinforcement_corrected(input_csv, input_pkl, output_csv)

import pandas as pd
import pickle
import os
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict

# ================= 配置常量 =================
INVALID_TARGETS = {
    'no-data', 'no data', 'nodata',
    'nan', 'null', 'none', '',
    'unknown', 'n/a', '-',
    'private', 'reserved'
}


def is_valid_target(target):
    """判断目标字符串是否有效"""
    if not isinstance(target, str):
        return False
    t = target.strip().lower()
    return t not in INVALID_TARGETS


def calculate_reinforcement_fine_grained(csv_file, pkl_file, output_file):
    print(f"[-] 正在读取流量数据: {csv_file}")
    if not os.path.exists(csv_file) or not os.path.exists(pkl_file):
        print("[!] 错误: 输入文件不存在。")
        return

    # 1. 加载 CSV
    df = pd.read_csv(csv_file)
    df['device_vendor'] = df['device_vendor'].fillna('unknown').astype(str).str.strip()
    # 去重
    df_clean = df[['Component_ID', 'Device', 'device_vendor']].drop_duplicates()

    print(f"[-] 正在加载隐性关系库并过滤无效数据: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        implicit_data = pickle.load(f)

    # 2. 构建隐性证据映射 (细粒度)
    # implicit_pair_lookup: (DevA, DevB) -> Set of specific targets (e.g. "Service:time.google.com")
    implicit_pair_lookup = defaultdict(set)

    # 统计过滤情况
    filtered_count = 0
    valid_count = 0

    for layer, targets in implicit_data.items():
        for target, devices in targets.items():
            # --- 过滤逻辑 ---
            # 如果是 Service 或 Infrastructure 层，必须检查 target 是否有效
            if layer in ['Service', 'Infrastructure']:
                if not is_valid_target(target):
                    filtered_count += 1
                    continue

            valid_count += 1

            if len(devices) < 2:
                continue

            # 将该目标记录为这些设备对的证据
            for d1, d2 in combinations(devices, 2):
                pair = tuple(sorted((d1, d2)))
                evidence_str = f"{layer}:{target}"
                implicit_pair_lookup[pair].add(evidence_str)

    print(f"    > 已加载有效隐性目标: {valid_count} (过滤掉了 {filtered_count} 个无效目标 'no-data' 等)")
    print(f"    > 涉及设备对数量: {len(implicit_pair_lookup)}")

    # 3. 遍历 Cluster 进行验证
    print("[-] 开始细粒度验证...")
    results = []

    # 用于存储 detailed_report 的数据结构 (仅内存，用于最后打印)
    report_data = {}

    grouped = df_clean.groupby('Component_ID')

    for cid, group in tqdm(grouped, desc="Analyzing Clusters"):
        dev_vendor_list = list(zip(group['Device'], group['device_vendor']))
        num_devs = len(dev_vendor_list)

        total_pairs = 0
        matched_pairs = 0

        # 收集该 Cluster 内的所有证据详情
        # 格式: Evidence_String -> Set of Devices that share this evidence within this cluster
        cluster_evidence_breakdown = defaultdict(set)

        if num_devs < 2:
            results.append({
                'Component_ID': cid, 'Device_Count': num_devs,
                'Reinforcement_Ratio': 0.0, 'Match_Detail': "Single_Device"
            })
            continue

        for i in range(num_devs):
            for j in range(i + 1, num_devs):
                d1, v1 = dev_vendor_list[i]
                d2, v2 = dev_vendor_list[j]

                total_pairs += 1
                is_pair_matched = False

                # --- A. 显性关系 (同厂商) ---
                if v1 == v2 and v1.lower() not in INVALID_TARGETS:
                    is_pair_matched = True
                    evidence_key = f"Explicit_Vendor:{v1}"
                    cluster_evidence_breakdown[evidence_key].add(d1)
                    cluster_evidence_breakdown[evidence_key].add(d2)

                # --- B. 隐性关系 (查表) ---
                pair = tuple(sorted((d1, d2)))
                if pair in implicit_pair_lookup:
                    is_pair_matched = True
                    targets = implicit_pair_lookup[pair]
                    for t in targets:
                        # 记录证据: 这个 Target 将 d1 和 d2 连接在了一起
                        cluster_evidence_breakdown[t].add(d1)
                        cluster_evidence_breakdown[t].add(d2)

                if is_pair_matched:
                    matched_pairs += 1

        # 计算 Ratio
        ratio = matched_pairs / total_pairs if total_pairs > 0 else 0.0

        # 生成 CSV 用的摘要字符串 (Top 3 evidences by device coverage)
        # 按覆盖设备数量排序
        sorted_evidences = sorted(cluster_evidence_breakdown.items(), key=lambda x: len(x[1]), reverse=True)
        summary_parts = []
        for reason, devs in sorted_evidences[:5]:  # CSV只存前5个主要原因
            summary_parts.append(f"{reason}({len(devs)}devs)")
        match_detail_str = " | ".join(summary_parts) if summary_parts else "No_Evidence"

        results.append({
            'Component_ID': cid,
            'Device_Count': num_devs,
            'Total_Pairs': total_pairs,
            'Matched_Pairs': matched_pairs,
            'Reinforcement_Ratio': ratio,
            'Match_Detail': match_detail_str
        })

        # 如果置信度高，保存详细数据用于打印报告
        if ratio > 0.6 and matched_pairs > 0:
            report_data[cid] = {
                'stats': (num_devs, total_pairs, matched_pairs, ratio),
                'evidence': sorted_evidences  # 包含完整的 (Reason, DeviceSet)
            }

    # 4. 保存 CSV
    df_results = pd.DataFrame(results)
    df_final = pd.merge(df, df_results[
        ['Component_ID', 'Total_Pairs', 'Matched_Pairs', 'Reinforcement_Ratio', 'Match_Detail']],
                        on='Component_ID', how='left')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"[-] 结果已保存至: {output_file}")

    # 5. 打印细粒度可视化报告
    print_granular_report(report_data)


def print_granular_report(report_data):
    print("\n" + "=" * 100)
    print(" >>> 细粒度已知关系验证报告 (Fine-Grained Evidence Report) <<<")
    print(" 说明: 展示 Cluster 内部设备是如何通过 Explicit(厂商) 或 Implicit(公共服务) 连接在一起的。")
    print("=" * 100)

    # 按 Ratio 降序，然后按设备数降序
    sorted_cids = sorted(report_data.keys(),
                         key=lambda k: (report_data[k]['stats'][3], report_data[k]['stats'][0]),
                         reverse=True)

    count = 0
    for cid in sorted_cids:
        if count >= 5: break  # 展示前 5 个最典型的 Cluster

        data = report_data[cid]
        num_devs, total, matched, ratio = data['stats']
        evidence_list = data['evidence']

        print(f"\n[Cluster: {cid}]")
        print(f"  摘要: 共 {num_devs} 台设备形成 {total} 个对，其中 {matched} 对已知关联 (Ratio: {ratio:.2f})")
        print(f"  一致性细节 (Coherence Details):")

        # 打印覆盖度最高的证据 (最多 5 条)
        shown_evidences = 0
        for reason, dev_set in evidence_list:
            if shown_evidences >= 5:
                print(f"      ... (还有 {len(evidence_list) - 5} 条证据)")
                break

            # 格式化设备列表，如果太长就截断
            dev_list_str = list(dev_set)
            if len(dev_list_str) > 4:
                dev_display = f"[{', '.join(dev_list_str[:3])}, ...等{len(dev_list_str)}台]"
            else:
                dev_display = f"{dev_list_str}"

            print(f"    - {reason:<40} -> shared by {dev_display}")
            shown_evidences += 1

        count += 1
    print("\n" + "=" * 100)


# --- 执行配置 ---
if __name__ == "__main__":
    input_csv = f'../addition_pro/5.1.Remaining_Unique_LLM_judge_False.csv'

    # 输入 2: 隐性关系 Pickle
    input_pkl = '../2_Implicit_Identification/Implicit_Relationships.pkl'

    # 输出
    output_csv = f'../addition_pro/5.3.Network_Flows_Validated_Reinforcement_v1.20.csv'

    calculate_reinforcement_fine_grained(input_csv, input_pkl, output_csv)

# # --- 执行 ---
# if __name__ == "__main__":
#     # base_dir = '../DTW_1_12'
#     # input_csv = f'{base_dir}/4.2.5.Filtered_Implicit_MultiVendor_Flows_v1.12.csv'
#     # input_pkl = 'Implicit_Relationships.pkl'
#     # output_csv = f'{base_dir}/4.5.Network_Flows_Validated_Reinforcement_Evidence_v1.12.csv'
#     input_csv = f'./A1.Filtered_Implicit_MultiVendor_Flows_v1.12.csv'
#
#     # 输入 2: 隐性关系 Pickle
#     input_pkl = '../2_Implicit_Identification/Implicit_Relationships.pkl'
#
#     # 输出
#     output_csv = f'./A4.2.Network_Flows_Validated_Reinforcement_v1.12.csv'


    # calculate_reinforcement_with_evidence(input_csv, input_pkl, output_csv)
