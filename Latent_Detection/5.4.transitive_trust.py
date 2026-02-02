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
    if not isinstance(target, str): return False
    return target.strip().lower() not in INVALID_TARGETS


def validate_transitive_trust(csv_file, pkl_file, output_file):
    print(f"[-] 正在读取流量数据: {csv_file}")
    if not os.path.exists(csv_file) or not os.path.exists(pkl_file):
        print("[!] 错误: 输入文件不存在。")
        return

    # 1. 加载并预处理数据
    df = pd.read_csv(csv_file)
    df['device_vendor'] = df['device_vendor'].fillna('unknown').astype(str).str.strip()

    # 获取全局设备列表和厂商映射
    # 注意：这里我们构建全局 Explicit 集合，不仅限于 Cluster 内部
    # 假设 CSV 包含了所有出现的设备信息。如果 Implicit PKL 里有 CSV 里没有的设备，
    # 那些设备只能作为 Bridge 存在，不能作为被验证对象。

    # 构建全局 Vendor Map: {Vendor -> Set(Devices)}
    # 这对应公式中的 R_Explicit
    global_vendor_map = defaultdict(set)
    # 构建设备到厂商的反向映射
    device_to_vendor = {}

    # 遍历整个 DataFrame 构建全局知识
    for _, row in df.iterrows():
        d = row['Device']
        v = row['device_vendor']
        if v.lower() not in INVALID_TARGETS:
            global_vendor_map[v].add(d)
            device_to_vendor[d] = v

    print(f"    > 全局已知厂商数: {len(global_vendor_map)}")

    # 2. 构建隐性关系邻接表 (Implicit Graph)
    # 对应公式中的 E_Implicit
    # 结构: {Device -> Set(Neighbors)}
    print(f"[-] 正在构建全局隐性关系图谱: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        implicit_data = pickle.load(f)

    implicit_graph = defaultdict(set)
    # 同时记录连接原因用于可视化: (DevA, DevB) -> Reason
    implicit_connection_reason = {}

    for layer, targets in implicit_data.items():
        for target, devices in targets.items():
            # 过滤无效目标
            if layer in ['Service', 'Infrastructure'] and not is_valid_target(target):
                continue

            if len(devices) < 2:
                continue

            # 全连接: 组内设备两两互联
            dev_list = list(devices)
            for i in range(len(dev_list)):
                for j in range(i + 1, len(dev_list)):
                    d1, d2 = dev_list[i], dev_list[j]

                    # 添加无向边
                    implicit_graph[d1].add(d2)
                    implicit_graph[d2].add(d1)

                    # 记录原因 (为了性能，只存一个主要的即可，或者存列表)
                    # Key 必须排序以保证一致性
                    pair_key = tuple(sorted((d1, d2)))
                    if pair_key not in implicit_connection_reason:
                        implicit_connection_reason[pair_key] = f"{layer}:{target}"

    print(f"    > 隐性图谱节点数: {len(implicit_graph)}")

    # 3. 执行传递性验证
    print("[-] 开始执行传递信任验证 (Transitive Trust)...")

    results = []
    report_data = {}  # 用于打印报告

    # 只处理去重后的 Cluster 数据
    df_clusters = df[['Component_ID', 'Device', 'device_vendor']].drop_duplicates()
    grouped = df_clusters.groupby('Component_ID')

    for cid, group in tqdm(grouped, desc="Analyzing Clusters"):
        dev_list = group['Device'].tolist()
        vendor_list = group['device_vendor'].tolist()
        n = len(dev_list)

        # 统计变量
        pairs_to_validate = 0  # 分母: 潜在对 (跨厂商)
        validated_pairs = 0  # 分子: 满足传递性的对

        # 收集证据用于展示
        cluster_bridges = []  # List of strings

        if n < 2:
            results.append({
                'Component_ID': cid, 'Transitive_Ratio': 0.0, 'Bridge_Evidence': "Single_Device"
            })
            continue

        for i in range(n):
            for j in range(i + 1, n):
                dA, vA = dev_list[i], vendor_list[i]
                dB, vB = dev_list[j], vendor_list[j]

                # 关注 Latent Pair: 主要是跨厂商的
                # 如果是同厂商，本身就有强 Explicit 关系，通常不需要 Transitive 验证
                # 但根据公式定义，我们对 pair 进行验证。这里主要关注 Cross-Vendor。
                if vA == vB:
                    continue

                pairs_to_validate += 1

                # === 传递性检查逻辑 ===
                # 目标: 寻找 D_C
                # 条件 1: D_C 是 D_A 的同厂商兄弟 (D_C in global_vendor_map[vA])
                # 条件 2: D_C 与 D_B 有隐性连接 (D_C in implicit_graph[dB])

                bridge_found = False
                bridge_info = ""

                # 方向 1: A 的兄弟 -> 连接 -> B
                if vA in global_vendor_map:
                    siblings_A = global_vendor_map[vA]
                    # 获取 B 的所有邻居
                    neighbors_B = implicit_graph.get(dB, set())

                    # 求交集: 既是 A 的兄弟，又是 B 的邻居
                    # 排除 A 自己 (虽然 A 也是 A 的兄弟，但我们找的是第三方桥梁)
                    candidates = (siblings_A & neighbors_B) - {dA}

                    if candidates:
                        bridge_found = True
                        dC = next(iter(candidates))  # 取一个做代表
                        # 获取 C 和 B 连接的原因
                        reason = implicit_connection_reason.get(tuple(sorted((dC, dB))), "Unknown")
                        bridge_info = f"{dA} -(sib)-> {dC} -(imp:{reason})-> {dB}"

                # 方向 2: B 的兄弟 -> 连接 -> A (对称性)
                if not bridge_found and vB in global_vendor_map:
                    siblings_B = global_vendor_map[vB]
                    neighbors_A = implicit_graph.get(dA, set())
                    candidates = (siblings_B & neighbors_A) - {dB}

                    if candidates:
                        bridge_found = True
                        dC = next(iter(candidates))
                        reason = implicit_connection_reason.get(tuple(sorted((dC, dA))), "Unknown")
                        bridge_info = f"{dB} -(sib)-> {dC} -(imp:{reason})-> {dA}"

                if bridge_found:
                    validated_pairs += 1
                    # 收集证据 (只存前 3 个不同的桥梁路径)
                    if len(cluster_bridges) < 3:
                        cluster_bridges.append(bridge_info)

        # 计算比率
        ratio = validated_pairs / pairs_to_validate if pairs_to_validate > 0 else 0.0

        # 格式化证据字符串
        evidence_str = " | ".join(cluster_bridges) if cluster_bridges else "No_Transitive_Path"

        results.append({
            'Component_ID': cid,
            'Latent_Pairs_Count': pairs_to_validate,
            'Bridged_Pairs_Count': validated_pairs,
            'Transitive_Ratio': ratio,
            'Bridge_Evidence': evidence_str
        })

        if ratio > 0.6 and validated_pairs > 0:
            report_data[cid] = {
                'stats': (n, pairs_to_validate, validated_pairs, ratio),
                'bridges': cluster_bridges
            }

    # 4. 保存结果
    df_res = pd.DataFrame(results)
    # Merge 回主表
    df_final = pd.merge(df, df_res[['Component_ID', 'Latent_Pairs_Count', 'Transitive_Ratio', 'Bridge_Evidence']],
                        on='Component_ID', how='left')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"[-] 结果已保存至: {output_file}")

    # 5. 打印可视化报告
    print_transitive_report(report_data)


def print_transitive_report(report_data):
    print("\n" + "=" * 100)
    print(" >>> 传递信任验证报告 (Transitive Trust Report) <<<")
    print(" 逻辑: A与B无直接关联，但 A的兄弟C 与 B 有关联 (A -> C -> B)")
    print("=" * 100)

    sorted_cids = sorted(report_data.keys(),
                         key=lambda k: (report_data[k]['stats'][3], report_data[k]['stats'][1]),
                         reverse=True)

    count = 0
    for cid in sorted_cids:
        if count >= 5: break

        data = report_data[cid]
        n_devs, n_latent, n_bridged, ratio = data['stats']
        bridges = data['bridges']

        print(f"\n[Cluster: {cid}]")
        print(
            f"  摘要: 包含 {n_devs} 设备。跨厂商对 {n_latent} 个，其中 {n_bridged} 个通过传递性验证 (Ratio: {ratio:.2f})")
        print(f"  信任桥梁示例 (Bridge Path Examples):")

        for path in bridges:
            print(f"    * {path}")

        count += 1
    print("\n" + "=" * 100)


# --- 执行配置 ---
if __name__ == "__main__":
    # base_dir = '../DTW_1_12'
    # 输入: 使用多厂商过滤后的文件
    input_csv = f'./A1.Filtered_Implicit_MultiVendor_Flows_v1.12.csv'
    input_pkl = '../2_Implicit_Identification/Implicit_Relationships.pkl'
    # 输出
    output_csv = f'./A5.Network_Flows_Validated_Transitive_v1.12.csv'

    validate_transitive_trust(input_csv, input_pkl, output_csv)
