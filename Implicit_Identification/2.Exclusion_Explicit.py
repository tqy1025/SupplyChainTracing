import pandas as pd
import pickle
import os

# ================= 配置部分 =================
CANDIDATES_FILE = 'Candidate_Intersections.pkl'  # 候选集文件
VENDOR_MAP_FILE = 'Device_Vendor_Map.pkl'  # 设备-厂商映射文件
OUTPUT_FILE = 'Implicit_Relationships.pkl'  # 最终结果


# ================= 主流程 =================

def main():
    # 1. 加载数据
    print("[-] Loading data...")
    if not all(os.path.exists(f) for f in [CANDIDATES_FILE, VENDOR_MAP_FILE]):
        print(f"[!] Error: Input files not found. Please run previous steps.")
        return

    with open(CANDIDATES_FILE, 'rb') as f:
        candidate_data = pickle.load(f)

    with open(VENDOR_MAP_FILE, 'rb') as f:
        vendor_map = pickle.load(f)

    print(f"    > Loaded {sum(len(v) for v in candidate_data.values())} total candidate sets.")
    print(f"    > Loaded {len(vendor_map)} device-vendor mappings.")

    # 用于存储最终结果
    implicit_relationships = {
        'IP': {},
        'Service': {},
        'Infrastructure': {}
    }

    # 用于统计
    stats = {
        'IP': {'total': 0, 'explicit': 0, 'implicit': 0},
        'Service': {'total': 0, 'explicit': 0, 'implicit': 0},
        'Infrastructure': {'total': 0, 'explicit': 0, 'implicit': 0}
    }

    # 2. 遍历并过滤每一层
    print("\n[-] Filtering Explicit Relationships...")

    for layer, candidates in candidate_data.items():
        print(f"--- Processing Layer: {layer} ---")

        stats[layer]['total'] = len(candidates)

        for target, device_set in candidates.items():

            # 2.1 获取当前设备集合对应的所有厂商
            vendors_in_set = set()
            for device in device_set:
                # 使用 .get() 安全地获取厂商，如果设备不在 map 中则标记为 'unknown'
                vendor = vendor_map.get(device, 'unknown')
                vendors_in_set.add(vendor)

            # 2.2 判断关系类型
            # 如果集合中厂商数量大于1，则是跨厂商的隐性关系
            if len(vendors_in_set) > 1:
                implicit_relationships[layer][target] = device_set
                stats[layer]['implicit'] += 1
            else:
                # 否则，是同厂商的显性关系，被过滤掉
                stats[layer]['explicit'] += 1

        print(f"    > Total Candidates: {stats[layer]['total']}")
        print(f"    > Explicit (Filtered): {stats[layer]['explicit']}")
        print(f"    > Implicit (Retained): {stats[layer]['implicit']}")

    # 3. 结果总结与展示
    print("=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)

    for layer, data in implicit_relationships.items():
        print(f"Layer: {layer}")
        print(f"  Implicit Relationships Found: {len(data)}")

        if data:
            # 打印一个样本
            sample_target = next(iter(data))
            sample_devices = data[sample_target]
            sample_vendors = {vendor_map.get(d, 'N/A') for d in sample_devices}

            print(f"    Sample Target: {sample_target}")
            print(f"      Devices ({len(sample_devices)}): {list(sample_devices)[:5]}...")
            print(f"      Vendors ({len(sample_vendors)}): {sample_vendors}")
            print("-" * 40)

    # 4. 保存结果
    print(f"[-] Saving implicit relationships to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(implicit_relationships, f)

    print("[-] Done.")


if __name__ == '__main__':
    main()
