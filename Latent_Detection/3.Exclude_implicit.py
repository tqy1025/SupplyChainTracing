import pandas as pd
import os
import numpy as np

# ================= 配置部分 =================
# 1. 待分析的聚类文件 (输入)
CSV_FILE = '../../DTW_1_12/2.Extended_Network_Flows_vendor_type_big_than_1_diff_vendor_v1.12.csv'

# 2. 流量详情文件 (提供 Service 和 Infrastructure 的映射依据)
# 必须包含列: 'Remote_IP', 'Domain_A' (Service), 'Domain_PTR' (Infrastructure)
TRAFFIC_PKL = '../1_Data_Preprocessing/Traffic_Cleaned_v3_Final.pkl'

# 3. 输出文件
OUTPUT_CONSISTENT = '3.1.Clusters_Consistent_Implicit.csv'  # 证明有一致性的
OUTPUT_INCONSISTENT = '3.2.Clusters_Inconsistent_NoImplicit.csv'  # 没有任何一致性的


# ================= 主流程 =================

def main():
    print("[-] Loading datasets...")

    if not os.path.exists(CSV_FILE):
        print(f"[!] Error: {CSV_FILE} not found.")
        return
    try:
        df_flows = pd.read_csv(CSV_FILE, dtype={'Remote_IP': str})
    except UnicodeDecodeError:
        df_flows = pd.read_csv(CSV_FILE, encoding='gbk', dtype={'Remote_IP': str})

    if not os.path.exists(TRAFFIC_PKL):
        print(f"[!] Error: {TRAFFIC_PKL} not found.")
        return

    print(f"[-] Loading mapping data from {TRAFFIC_PKL}...")
    df_traffic = pd.read_pickle(TRAFFIC_PKL)

    # 标准化列名
    if 'Domain_A' not in df_traffic.columns and 'Remote_domain' in df_traffic.columns:
        df_traffic['Domain_A'] = df_traffic['Remote_domain']
    if 'Domain_PTR' not in df_traffic.columns and 'DNS' in df_traffic.columns:
        df_traffic['Domain_PTR'] = df_traffic['DNS']

    # 确保列存在，没有则补 None
    for col in ['Domain_A', 'Domain_PTR']:
        if col not in df_traffic.columns: df_traffic[col] = None

    # 构建 IP 映射表
    # 注意：这里我们要非常小心。一个 IP 在 Traffic 文件中可能有多条记录。
    # 有些记录可能有 Domain，有些可能没有（虽然理论上 IP 的 PTR 应该是固定的，但 Domain_A 取决于抓包时的 SNI/Host）。
    # 策略：对于同一个 IP，如果曾经出现过非空的 Domain_A，我们应该尽量采用非空值。
    # 因此，排序让非空值排在前面，然后 drop_duplicates

    # 辅助列用于排序：非空为0，空为1
    df_traffic['Service_Rank'] = df_traffic['Domain_A'].isna().astype(int)
    df_traffic['Infra_Rank'] = df_traffic['Domain_PTR'].isna().astype(int)

    # 1. 构建 IP -> Service (取最优值)
    df_svc = df_traffic.sort_values(['Remote_IP', 'Service_Rank'])
    map_svc = df_svc.drop_duplicates('Remote_IP')[['Remote_IP', 'Domain_A']]
    ip_to_service = dict(zip(map_svc['Remote_IP'], map_svc['Domain_A']))

    # 2. 构建 IP -> Infrastructure (取最优值)
    df_inf = df_traffic.sort_values(['Remote_IP', 'Infra_Rank'])
    map_inf = df_inf.drop_duplicates('Remote_IP')[['Remote_IP', 'Domain_PTR']]
    ip_to_infra = dict(zip(map_inf['Remote_IP'], map_inf['Domain_PTR']))

    print(f"    > Mapped {len(ip_to_service)} IPs.")

    # ==============================================================================
    # 映射到原始 DataFrame
    # ==============================================================================
    print("[-] Mapping Service and Infrastructure columns...")
    df_flows['Service'] = df_flows['Remote_IP'].map(ip_to_service)
    df_flows['Infrastructure'] = df_flows['Remote_IP'].map(ip_to_infra)

    # 填充 NaN 为 None 方便逻辑判断
    df_flows['Service'] = df_flows['Service'].where(pd.notnull(df_flows['Service']), None)
    df_flows['Infrastructure'] = df_flows['Infrastructure'].where(pd.notnull(df_flows['Infrastructure']), None)

    # 打印一下刚才那个有问题的 IP 映射结果，确保映射逻辑本身没问题（即确实是 None）
    check_ip = '42.157.165.144'
    print(f"    [Check] IP {check_ip} maps to Service: {ip_to_service.get(check_ip)}")

    # ==============================================================================
    # 聚类一致性检查 (Strict Logic)
    # ==============================================================================
    print(f"[-] Checking consistency for {len(df_flows['Component_ID'].unique())} clusters...")

    results = []
    grouped = df_flows.groupby('Component_ID')

    for comp_id, group in grouped:
        res = {
            'Component_ID': comp_id,
            'Is_Implicit': False,
            'Consistent_Layers': [],
            'Cluster_Val_IP': None,
            'Cluster_Val_Service': None,
            'Cluster_Val_Infra': None
        }

        # --- A. Check IP Consistency ---
        # 逻辑：所有流的 IP 必须完全一致
        unique_ips = group['Remote_IP'].unique()
        if len(unique_ips) == 1:
            res['Cluster_Val_IP'] = unique_ips[0]
            res['Consistent_Layers'].append('IP')
            res['Is_Implicit'] = True

        # --- B. Check Service Consistency (Strict) ---
        # 逻辑：
        # 1. 提取所有流的 Service 值
        # 2. 如果包含任何 None/NaN，直接判定为不一致 (因为无法证明那个 None 的 IP 连的是同一个服务)
        # 3. 如果不含 None，但包含 >1 个不同的值，不一致
        # 4. 只有当 Unique 值只有 1 个且非 None 时，才一致

        # 注意：这里要处理空字符串的情况
        services = group['Service'].tolist()

        # 检查是否全非空
        has_none = any((s is None or str(s).strip() == '') for s in services)

        if not has_none:
            unique_svcs = set(services)
            if len(unique_svcs) == 1:
                res['Cluster_Val_Service'] = list(unique_svcs)[0]
                res['Consistent_Layers'].append('Service')
                res['Is_Implicit'] = True

        # --- C. Check Infrastructure Consistency (Strict) ---
        infras = group['Infrastructure'].tolist()
        has_none_inf = any((i is None or str(i).strip() == '') for i in infras)

        if not has_none_inf:
            unique_infs = set(infras)
            if len(unique_infs) == 1:
                res['Cluster_Val_Infra'] = list(unique_infs)[0]
                res['Consistent_Layers'].append('Infrastructure')
                res['Is_Implicit'] = True

        res['Consistent_Layers'] = "|".join(res['Consistent_Layers'])
        results.append(res)

    # ==============================================================================
    # 合并与保存
    # ==============================================================================
    print("[-] Splitting results...")

    df_results = pd.DataFrame(results)
    df_final = df_flows.merge(df_results, on='Component_ID', how='left')

    df_consistent = df_final[df_final['Is_Implicit'] == True].copy()
    df_inconsistent = df_final[df_final['Is_Implicit'] == False].copy()

    # 清理列
    cols_to_drop_inc = ['Is_Implicit', 'Consistent_Layers', 'Cluster_Val_IP', 'Cluster_Val_Service',
                        'Cluster_Val_Infra']
    df_inconsistent.drop(columns=cols_to_drop_inc, errors='ignore', inplace=True)

    print(f"[-] Saving {len(df_consistent)} rows to {OUTPUT_CONSISTENT}...")
    df_consistent.to_csv(OUTPUT_CONSISTENT, index=False)

    print(f"[-] Saving {len(df_inconsistent)} rows to {OUTPUT_INCONSISTENT}...")
    df_inconsistent.to_csv(OUTPUT_INCONSISTENT, index=False)

    # 统计
    print("\n" + "=" * 40)
    print("STRICT CONSISTENCY STATISTICS")
    print("=" * 40)
    total_clusters = len(results)
    implicit_clusters = df_results['Is_Implicit'].sum()

    print(f"Total Clusters: {total_clusters}")
    print(f"Strict Implicit: {implicit_clusters} ({implicit_clusters / total_clusters * 100:.2f}%)")
    print(f"Inconsistent: {total_clusters - implicit_clusters}")

    cnt_ip = df_results['Consistent_Layers'].str.contains('IP').sum()
    cnt_svc = df_results['Consistent_Layers'].str.contains('Service').sum()
    cnt_inf = df_results['Consistent_Layers'].str.contains('Infrastructure').sum()

    print(f"  IP Consistent: {cnt_ip}")
    print(f"  Service Consistent: {cnt_svc}")
    print(f"  Infrastructure Consistent: {cnt_inf}")
    print("=" * 40)

    # 调试：查看原来的 0621 号 Cluster 现在被分到哪里了
    # 假设 Component_ID 是 Comp_0621
    target_id = 'Comp_0621'
    if target_id in df_final['Component_ID'].values:
        row = df_final[df_final['Component_ID'] == target_id].iloc[0]
        print(f"\n[Debug] {target_id} is now classified as: {'Implicit' if row['Is_Implicit'] else 'Inconsistent'}")
        print(f"        Layers: {row['Consistent_Layers']}")


if __name__ == '__main__':
    main()

