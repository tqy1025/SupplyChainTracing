import pandas as pd
import pickle
import os

# ================= 配置部分 =================
INPUT_FILE = '../1_Data_Preprocessing/Traffic_Cleaned_v3_Final.pkl'  # Tier-2 的输出文件
OUTPUT_CANDIDATES = 'Candidate_Intersections.pkl'  # 保存结果
MIN_DEVICES = 2  # 至少2个设备才算作"交集"


# ================= 辅助函数 =================

def generate_layer_candidates(df, target_col, layer_name):
    """
    针对指定列（IP/Domain_A/Domain_PTR）生成候选设备集合。
    返回: dict {Target: set(Device_ID, ...)}
    """
    print(f"[-] Generating candidates for {layer_name} (Col: {target_col})...")

    # 1. 过滤掉目标为空的行
    # 注意: Domain_A/PTR 可能为 None, 'None', nan
    valid_df = df[df[target_col].notna() & (df[target_col] != 'None')].copy()

    if valid_df.empty:
        print(f"    > No valid data for {layer_name}.")
        return {}

    # 2. 聚合: Target -> Unique Devices
    # 使用 groupby 快速聚合
    # Device 列名兼容性处理
    dev_col = 'Device' if 'Device' in df.columns else 'Device_ID'

    # 得到 Series: index=Target, values=Array of Devices
    grouped = valid_df.groupby(target_col)[dev_col].unique()

    # 3. 筛选: 只保留设备数 >= MIN_DEVICES 的目标
    candidates = {}
    count_ignored = 0

    for target, devices in grouped.items():
        if len(devices) >= MIN_DEVICES:
            candidates[target] = set(devices)
        else:
            count_ignored += 1

    print(f"    > Found {len(candidates)} targets shared by >= {MIN_DEVICES} devices.")
    print(f"    > Ignored {count_ignored} targets (single device usage).")

    return candidates


# ================= 主流程 =================

def main():
    # 1. 加载数据
    print(f"[-] Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"[!] Error: {INPUT_FILE} not found.")
        return

    df = pd.read_pickle(INPUT_FILE)
    print(f"    > Data Shape: {df.shape}")

    # 确保列存在
    required_cols = ['Remote_IP', 'Domain_A', 'Domain_PTR']
    if 'Device' not in df.columns and 'Device_ID' in df.columns:
        df.rename(columns={'Device_ID': 'Device'}, inplace=True)

    # 2. 生成三层候选集
    # 结果结构: results['IP'] = { '1.1.1.1': {'DevA', 'DevB'}, ... }
    results = {}

    # --- Level 1: IP Address (Physical Convergence) ---
    results['IP'] = generate_layer_candidates(df, 'Remote_IP', 'IP-Level')

    # --- Level 2: Service Endpoint (Tier-1 / Domain_A) ---
    results['Service'] = generate_layer_candidates(df, 'Domain_A', 'Service-Level')

    # --- Level 3: Infrastructure Endpoint (Tier-2 / Domain_PTR) ---
    results['Infrastructure'] = generate_layer_candidates(df, 'Domain_PTR', 'Infrastructure-Level')

    # 3. 统计与展示
    print("=" * 60)
    print("CANDIDATE INTERSECTION SUMMARY")
    print("=" * 60)

    for layer, data in results.items():
        print(f"Layer: {layer}")
        print(f"  Shared Targets Found: {len(data)}")

        # 打印 Top 3 最热门的目标（连接设备最多的）
        sorted_targets = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)
        for target, devs in sorted_targets[:3]:
            # 为了展示简洁，只打印前3个设备ID
            dev_list = list(devs)
            preview = f"{dev_list[:3]}..." if len(dev_list) > 3 else str(dev_list)
            print(f"    Top Target: {target:<30} | Device Count: {len(devs)} | Devs: {preview}")
        print("-" * 40)

    # 4. 保存结果
    print(f"[-] Saving candidates to {OUTPUT_CANDIDATES}...")
    with open(OUTPUT_CANDIDATES, 'wb') as f:
        pickle.dump(results, f)

    # 5. 补充：构建 Device -> Vendor 映射表供下一步(Surface Exclusion)使用
    # 下一步我们需要判断这些设备是否属于同一厂商，所以顺便保存一个 Vendor Map 很有用
    if 'Vendor' in df.columns:
        # 去重建立 Device -> Vendor 字典
        vendor_map = df[['Device', 'Vendor']].drop_duplicates().set_index('Device')['Vendor'].to_dict()
        with open('Device_Vendor_Map.pkl', 'wb') as f:
            pickle.dump(vendor_map, f)
        print("[-] Saved Device_Vendor_Map.pkl for next step (Surface Exclusion).")

    print("[-] Done.")


if __name__ == '__main__':
    main()