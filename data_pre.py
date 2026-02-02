import os
import pickle
import csv
import ipaddress
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ==============================================================================
# 配置区域
# ==============================================================================
INPUT_PICKLE_PATH = "../../ALL_Flow_Adjust_DTW/Aright1_traffic_features_total_signed.pkl"
OUTPUT_PICKLE_PATH = "Preprocessed/Traffic_Cleaned_v2.pkl"
DEVICE_TYPE_CSV_PATH = "../../Input/device_type.csv"

# 2. Minimum Length Constraint
MIN_PACKET_COUNT = 5

# 3. Protocol Filtering (DNS List)
KNOWN_PUBLIC_DNS_IPS = {
    '8.8.8.8', '8.8.4.4', '1.1.1.1', '1.0.0.1',
    '208.67.222.222', '208.67.220.220', '9.9.9.9',
    '114.114.114.114', '114.114.115.115', '223.5.5.5', '223.6.6.6'
}

# 5. Anomaly Removal Thresholds
MAX_RST_RATIO = 0.40  # > 40% RST packets
MAX_RETRANS_RATIO = 0.50  # > 50% Retransmission


# ==============================================================================
# 工具函数
# ==============================================================================

def load_device_type_map(csv_path):
    """加载设备类型映射，用于排除 Hub"""
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: Device type CSV not found at {csv_path}")
        return mapping
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # 处理 BOM 头
            if reader.fieldnames and reader.fieldnames[0].startswith('\ufeff'):
                reader.fieldnames[0] = reader.fieldnames[0].replace('\ufeff', '')

            if 'Device_Name' in reader.fieldnames and 'Type' in reader.fieldnames:
                for row in reader:
                    mapping[row['Device_Name'].strip()] = row['Type'].strip()
    except Exception as e:
        print(f"❌ Error loading device map: {e}")
    return mapping


def is_private_ip(ip_str):
    """1. LAN Traffic Exclusion Check"""
    if not ip_str: return False
    try:
        # 移除端口号（如果有）
        clean_ip = ip_str.split(':')[0]
        ip_obj = ipaddress.ip_address(clean_ip)
        return ip_obj.is_private
    except ValueError:
        return False


def is_dns_traffic(remote_ip, protocol):
    """3. Protocol Filtering (DNS) Check"""
    if protocol != 'UDP':
        return False
    clean_ip = remote_ip.split(':')[0]
    if clean_ip in KNOWN_PUBLIC_DNS_IPS:
        return True
    return False


def is_tcp_anomaly(flow):
    """5. Anomaly Removal Check"""
    # 如果不是 TCP，则不算异常（本规则只针对 TCP）
    if flow.get('Protocol') != 'TCP':
        return False

    # 获取总包数，防止除以零
    # 注意：这里假设 Payload_Sequence 长度近似等于包数量，或者数据中有 'Total_Packets' 字段
    seq = flow.get('Payload_Sequence', [])
    total_packets = len(seq)
    if total_packets == 0:
        return True  # 空流视为异常

    # --- 获取关键指标 (需要根据实际数据字段名称修改) ---
    # 假设流数据中包含 'RST_Count' 和 'Retrans_Count'
    # 如果你的数据只有 flags 列表，需要先统计
    rst_count = flow.get('RST_Count', 0)
    retrans_count = flow.get('Retrans_Count', 0)

    # 检查 RST 比例
    if (rst_count / total_packets) > MAX_RST_RATIO:
        return True

    # 检查重传比例
    if (retrans_count / total_packets) > MAX_RETRANS_RATIO:
        return True

    return False


# ==============================================================================
# 主处理逻辑
# ==============================================================================

def main():
    print(f"🚀 Starting Preprocessing...")

    # 1. 加载数据
    if not os.path.exists(INPUT_PICKLE_PATH):
        print(f"❌ Input file not found: {INPUT_PICKLE_PATH}")
        return

    print("⏳ Loading pickle file...")
    with open(INPUT_PICKLE_PATH, 'rb') as f:
        raw_data = pickle.load(f)

    # 兼容处理：如果是字典则拍平
    all_flows = []
    if isinstance(raw_data, dict):
        for v in raw_data.values():
            if isinstance(v, list):
                all_flows.extend(v)
            else:
                all_flows.append(v)
    elif isinstance(raw_data, list):
        all_flows = raw_data

    print(f"📊 Total raw flows: {len(all_flows)}")

    # 加载设备映射
    device_type_map = load_device_type_map(DEVICE_TYPE_CSV_PATH)

    # 统计计数器
    stats = {
        'kept': 0,
        'drop_lan': 0,  # 规则 1
        'drop_len': 0,  # 规则 2
        'drop_dns': 0,  # 规则 3
        'drop_hub': 0,  # 规则 4
        'drop_anomaly': 0  # 规则 5
    }

    cleaned_flows = []

    # 2. 遍历过滤
    for flow in tqdm(all_flows, desc="Filtering Flows"):
        # 提取基础信息
        remote_ip = flow.get('Remote_IP', '')
        protocol = flow.get('Protocol', '')
        device_name = flow.get('Device', 'Unknown')
        seq = flow.get('Payload_Sequence', [])

        # --- Rule 4: Device Filtering (Smart Hubs) ---
        # 优先检查，因为查表很快
        dev_type = device_type_map.get(device_name, 'Unknown')
        if dev_type == 'Hub':
            stats['drop_hub'] += 1
            continue

        # --- Rule 2: Minimum Length Constraint ---
        up_cnt = sum(1 for x in seq if x > 0)
        down_cnt = sum(1 for x in seq if x < 0)
        if up_cnt < MIN_PACKET_COUNT or down_cnt < MIN_PACKET_COUNT:
            stats['drop_len'] += 1
            continue

        # --- Rule 1: LAN Traffic Exclusion ---
        if is_private_ip(remote_ip):
            stats['drop_lan'] += 1
            continue

        # --- Rule 3: Protocol Filtering (DNS) ---
        if is_dns_traffic(remote_ip, protocol):
            stats['drop_dns'] += 1
            continue

        # --- Rule 5: Anomaly Removal (TCP RST/Retrans) ---
        # if is_tcp_anomaly(flow):
        #     stats['drop_anomaly'] += 1
        #     continue

        # ✅ Passed all checks
        cleaned_flows.append(flow)
        stats['kept'] += 1

    # 3. 保存结果
    print("\n" + "=" * 40)
    print("📋 Filter Statistics:")
    print(f"  Total Input   : {len(all_flows)}")
    print(f"  Kept          : {stats['kept']}")
    print("-" * 20)
    print(f"  ❌ Dropped (LAN)     : {stats['drop_lan']}")
    print(f"  ❌ Dropped (Min Len) : {stats['drop_len']}")
    print(f"  ❌ Dropped (DNS)     : {stats['drop_dns']}")
    print(f"  ❌ Dropped (Hubs)    : {stats['drop_hub']}")
    print(f"  ❌ Dropped (Anomaly) : {stats['drop_anomaly']}")
    print("=" * 40)

    os.makedirs(os.path.dirname(OUTPUT_PICKLE_PATH), exist_ok=True)
    with open(OUTPUT_PICKLE_PATH, 'wb') as f:
        pickle.dump(cleaned_flows, f)

    print(f"✅ Cleaned data saved to: {OUTPUT_PICKLE_PATH}")


if __name__ == "__main__":
    main()