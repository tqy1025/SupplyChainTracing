import os
import pickle
import numpy as np
import ipaddress
import gc
import sys
import csv
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import resource
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, fcluster

# ==============================================================================
# 0. Numba 环境检测与配置
# ==============================================================================
try:
    from numba import jit, prange

    HAS_NUMBA = True
    print("✅ Numba detected! High-speed acceleration enabled.")
except ImportError:
    HAS_NUMBA = False
    print("⚠️ Numba not found. Running in slow mode. Install via 'pip install numba'")


    # Dummy decorators for compatibility
    def jit(nopython=True, parallel=False):
        def decorator(func):
            return func

        return decorator


    def prange(n):
        return range(n)

# ==============================================================================
# 1. 参数配置区域
# ==============================================================================

# --- 过滤参数 ---
MIN_PACKET_COUNT = 5  # 最小包数量
# [关键限制] 单设备参与聚类的最大流数量 (防止全连接 O(N^2) 爆内存)
MAX_TOTAL_FLOWS_PER_DEVICE = 100000

# --- DTW 算法核心参数 (1.9.1 版) ---
DTW_TOP_N = 20

# 1.9.1 版参数：固定权重 + MTU 分区
DTW_SMALL_THRESH = 100.0
DTW_SMALL_WEIGHT = 10.0  # 严打小包：控制类包必须精准 (Diff * 20)

DTW_MTU_THRESH = 1400.0  # [1.9.1] MTU 阈值
DTW_MTU_WEIGHT = 1.0  # [1.9.1] MTU 权重：不予优惠，精准区分 (Diff / 1.0)

DTW_BIG_THRESH = 350.0  # 大包阈值：普通载荷
DTW_BIG_WEIGHT = 8.0  # [1.9.1] 大包优惠：容忍载荷波动 (Diff / 5.0)

DTW_CROSS_WEIGHT = 3.0  # 结构性惩罚：小包 vs 非小包

DTW_SIGMA = 35.0  # [1.9.1] Sigma 更新

SIMILARITY_THRESHOLD = 0.90
# 距离阈值 (用于 Complete Linkage)
DISTANCE_THRESHOLD = 1.0 - SIMILARITY_THRESHOLD

# --- 输入/输出路径 ---
INPUT_PICKLE_PATH = "../ALL_Flow_Adjust_DTW/traffic_features_total_signed.pkl"
OUTPUT_PICKLE_PATH = f"../DTW_1_12/1.In_Device_Complete_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_v1.12.pkl"
CHECKPOINT_PATH = f"../DTW_1_12/checkpoint_complete_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_v1.12.pkl"
DEVICE_TYPE_CSV_PATH = "../Input/device_type.csv"

# --- 并行配置 ---
NUM_WORKERS = 32  # 根据 CPU 核数调整
TASK_BATCH_SIZE = 100

# --- 排除列表 ---
KNOWN_PUBLIC_DNS_IPS = {
    '8.8.8.8', '8.8.4.4', '1.1.1.1', '1.0.0.1',
    '208.67.222.222', '208.67.220.220', '9.9.9.9', '149.112.112.112',
    '223.5.5.5', '223.6.6.6', '114.114.114.114', '114.114.115.115',
    '119.29.29.29', '4.2.2.1', '4.2.2.2', '4.2.2.3', '4.2.2.4',
    '64.6.64.6', '64.6.65.6'
}


# ==============================================================================
# 2. 基础工具函数
# ==============================================================================

def limit_memory(maxsize_gb):
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        maxsize_bytes = maxsize_gb * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (maxsize_bytes, hard))
    except Exception:
        pass


def is_private_ip(ip_str):
    if not ip_str: return False
    try:
        clean_ip = ip_str.split(':')[0]
        ip_obj = ipaddress.ip_address(clean_ip)
        return ip_obj.is_private
    except ValueError:
        return False


def is_dns_flow(remote_ip, protocol):
    if protocol != 'UDP': return False
    if remote_ip in KNOWN_PUBLIC_DNS_IPS: return True
    return False


def load_device_type_map(csv_path):
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and reader.fieldnames[0].startswith('\ufeff'):
                reader.fieldnames[0] = reader.fieldnames[0].replace('\ufeff', '')
            if 'Device_Name' in reader.fieldnames and 'Type' in reader.fieldnames:
                for row in reader:
                    mapping[row['Device_Name'].strip()] = row['Type'].strip()
    except Exception:
        pass
    return mapping


def extract_dtw_features(signed_seq):
    if not signed_seq:
        return {'up': np.array([], dtype=np.float64), 'down': np.array([], dtype=np.float64)}
    up = np.array([abs(x) for x in signed_seq if x > 0][:DTW_TOP_N], dtype=np.float64)
    down = np.array([abs(x) for x in signed_seq if x < 0][:DTW_TOP_N], dtype=np.float64)
    return {'up': up, 'down': down}


def convert_feats_to_matrix(feats_list, max_len=DTW_TOP_N):
    n = len(feats_list)
    matrix = np.zeros((n, max_len), dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int32)

    for i, seq in enumerate(feats_list):
        l = len(seq)
        if l > max_len: l = max_len
        if l > 0:
            matrix[i, :l] = seq[:l]
        lengths[i] = l

    return matrix, lengths


# ==============================================================================
# 3. DTW 核心计算 (1.9.1 版 - Numba 优化)
# ==============================================================================

@jit(nopython=True)
def calculate_weighted_dtw_numba(seq_a, seq_b,
                                 small_thresh, small_weight,
                                 mtu_thresh, mtu_weight,  # 新增 MTU 参数
                                 big_thresh, big_weight,
                                 cross_weight):
    """
    1.9.1版 DTW 算法：
    - 固定权重 (无自适应比率)
    - 引入 MTU 区域 (>1400) 区分 VPN/隧道
    - 优先级: 小包 > MTU > 大包 > 跨区 > 其他
    """
    n = len(seq_a)
    m = len(seq_b)
    if n == 0 and m == 0: return 0.0
    if n == 0 or m == 0: return 1e15

    prev_row = np.empty(m + 1, dtype=np.float64)
    curr_row = np.empty(m + 1, dtype=np.float64)
    prev_row[:] = 1e15
    prev_row[0] = 0.0

    for i in range(1, n + 1):
        curr_row[:] = 1e15
        val_a = seq_a[i - 1]

        # 预计算属性
        is_small_a = val_a < small_thresh
        is_big_a = val_a > big_thresh
        is_mtu_a = val_a > mtu_thresh

        for j in range(1, m + 1):
            val_b = seq_b[j - 1]
            base_diff = abs(val_a - val_b)

            is_small_b = val_b < small_thresh
            is_big_b = val_b > big_thresh
            is_mtu_b = val_b > mtu_thresh

            # --- 核心权重逻辑 1.9.1 ---
            # 1. 【双小包】
            if is_small_a and is_small_b:
                cost = base_diff * small_weight
            # 2. 【双 MTU 包】 (优先于普通大包判断)
            elif is_mtu_a and is_mtu_b:
                cost = base_diff / mtu_weight
            # 3. 【双大包】
            elif is_big_a and is_big_b:
                cost = base_diff / big_weight
            # 4. 【结构错位】 (小包 vs 非小包)
            elif is_small_a != is_small_b:
                cost = base_diff * cross_weight
            # 5. 【中包 或 大包vsMTU包】
            else:
                cost = base_diff
            # ---------------------------

            v_diag, v_up, v_left = prev_row[j - 1], prev_row[j], curr_row[j - 1]
            min_val = v_diag
            if v_up < min_val: min_val = v_up
            if v_left < min_val: min_val = v_left
            curr_row[j] = cost + min_val

        for k in range(m + 1):
            prev_row[k] = curr_row[k]

    return prev_row[m]


# ==============================================================================
# 4. 全连接距离矩阵计算 (Numba Parallel + Matrix Input)
# ==============================================================================

@jit(nopython=True, parallel=True)
def compute_condensed_distance_matrix_numba(
        matrix_up, lengths_up,
        matrix_down, lengths_down,
        n,
        small_thresh, small_weight,
        mtu_thresh, mtu_weight,  # New
        big_thresh, big_weight,
        cross_weight, sigma
):
    """
    计算压缩距离矩阵 (1D array)
    """
    dist_len = n * (n - 1) // 2
    dist_array = np.empty(dist_len, dtype=np.float64)

    for i in prange(n):
        len_u_a = lengths_up[i]
        len_d_a = lengths_down[i]
        up_a = matrix_up[i, :len_u_a]
        down_a = matrix_down[i, :len_d_a]
        row_offset = i * n - (i * (i + 1)) // 2

        for j in range(i + 1, n):
            len_u_b = lengths_up[j]
            len_d_b = lengths_down[j]
            up_b = matrix_up[j, :len_u_b]
            down_b = matrix_down[j, :len_d_b]

            # --- DTW Calculation (1.9.1) ---
            dist_up = calculate_weighted_dtw_numba(
                up_a, up_b,
                small_thresh, small_weight,
                mtu_thresh, mtu_weight,
                big_thresh, big_weight,
                cross_weight
            )
            dist_down = calculate_weighted_dtw_numba(
                down_a, down_b,
                small_thresh, small_weight,
                mtu_thresh, mtu_weight,
                big_thresh, big_weight,
                cross_weight
            )

            # 归一化
            mean_len_up = (len_u_a + len_u_b) / 2.0
            mean_len_down = (len_d_a + len_d_b) / 2.0
            norm_up = dist_up / mean_len_up if mean_len_up > 0 else 1e9
            norm_down = dist_down / mean_len_down if mean_len_down > 0 else 1e9
            sim_up = np.exp(-norm_up / sigma) if mean_len_up > 0 else 0.0
            sim_down = np.exp(-norm_down / sigma) if mean_len_down > 0 else 0.0
            final_sim = np.sqrt(sim_up * sim_down)
            d = 1.0 - final_sim
            if d < 0: d = 0.0

            k = row_offset + (j - i - 1)
            dist_array[k] = d

    return dist_array


# ==============================================================================
# 5. 单设备全连接聚类逻辑 (无分治, 确定性)
# ==============================================================================

def process_device_complete_clustering(device_name, flow_list_wrapper):
    flow_list_wrapper.sort(key=lambda x: len(x['raw'].get('Payload_Sequence', [])), reverse=True)
    if len(flow_list_wrapper) > MAX_TOTAL_FLOWS_PER_DEVICE:
        flow_list_wrapper = flow_list_wrapper[:MAX_TOTAL_FLOWS_PER_DEVICE]

    n = len(flow_list_wrapper)
    if n == 0: return []
    if n == 1: return [flow_list_wrapper[0]['raw']]

    feats_up_list = []
    feats_down_list = []
    for item in flow_list_wrapper:
        if 'feats' not in item:
            item['feats'] = extract_dtw_features(item['raw'].get('Payload_Sequence', []))
        feats_up_list.append(item['feats']['up'])
        feats_down_list.append(item['feats']['down'])

    mat_up, len_up = convert_feats_to_matrix(feats_up_list, max_len=DTW_TOP_N)
    mat_down, len_down = convert_feats_to_matrix(feats_down_list, max_len=DTW_TOP_N)

    try:
        # [修改] 传入 1.9.1 版参数
        condensed_matrix = compute_condensed_distance_matrix_numba(
            mat_up, len_up,
            mat_down, len_down,
            n,
            DTW_SMALL_THRESH, DTW_SMALL_WEIGHT,
            DTW_MTU_THRESH, DTW_MTU_WEIGHT,
            DTW_BIG_THRESH, DTW_BIG_WEIGHT,
            DTW_CROSS_WEIGHT, DTW_SIGMA
        )
    except Exception as e:
        print(f"❌ Matrix calc error ({device_name}): {e}")
        return []

    try:
        Z = linkage(condensed_matrix, method='complete')
        labels = fcluster(Z, t=DISTANCE_THRESHOLD, criterion='distance')
    except Exception as e:
        print(f"⚠️ Linkage error ({device_name}): {e}")
        return [f['raw'] for f in flow_list_wrapper]

    cluster_groups = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_groups[label].append(flow_list_wrapper[idx])

    representatives = []
    for members in cluster_groups.values():
        best_rep = max(members, key=lambda x: len(x['raw'].get('Payload_Sequence', [])))
        representatives.append(best_rep['raw'])

    return representatives


# ==============================================================================
# 6. 主程序
# ==============================================================================

def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    return {}


def save_checkpoint(path, data):
    temp = path + ".tmp"
    with open(temp, 'wb') as f: pickle.dump(data, f)
    if os.path.exists(path): os.remove(path)
    os.rename(temp, path)


def main():
    if not os.path.exists(INPUT_PICKLE_PATH):
        print("❌ 输入文件不存在")
        return

    device_type_map = load_device_type_map(DEVICE_TYPE_CSV_PATH)

    print("⏳ 加载数据中...")
    with open(INPUT_PICKLE_PATH, 'rb') as f:
        all_flows = pickle.load(f)
    if isinstance(all_flows, dict):
        temp = []
        for v in all_flows.values():
            if isinstance(v, list):
                temp.extend(v)
            else:
                temp.append(v)
        all_flows = temp

    print("🔍 预处理 (Filter & Group)...")
    device_groups = defaultdict(list)
    stats = {'total': 0, 'drop': 0, 'kept': 0, 'hub_dropped': 0}

    for flow in tqdm(all_flows, desc="Filtering"):
        stats['total'] += 1
        seq = flow.get('Payload_Sequence', [])
        up_cnt = sum(1 for x in seq if x > 0)
        down_cnt = sum(1 for x in seq if x < 0)
        device_name = flow.get('Device', 'Unknown')

        if device_type_map.get(device_name) == 'Hub':
            stats['drop'] += 1
            stats['hub_dropped'] += 1
            continue
        if up_cnt < MIN_PACKET_COUNT or down_cnt < MIN_PACKET_COUNT:
            stats['drop'] += 1
            continue
        if is_private_ip(flow.get('Remote_IP')) or is_dns_flow(flow.get('Remote_IP'), flow.get('Protocol')):
            stats['drop'] += 1
            continue
        device_groups[device_name].append({'raw': flow})
        stats['kept'] += 1

    print("\n" + "=" * 40)
    print(f"设备总数: {len(device_groups)}")
    print(f"保留总流: {stats['kept']}")
    print(f"HUB丢弃 : {stats['hub_dropped']}")
    print("=" * 40 + "\n")

    del all_flows
    gc.collect()

    processed_results = load_checkpoint(CHECKPOINT_PATH)
    all_devs = list(device_groups.keys())
    tasks = [d for d in all_devs if d not in processed_results]

    for d in all_devs:
        if d not in tasks: del device_groups[d]
    gc.collect()

    print(f"📋 任务: {len(tasks)} 个设备待处理 (Mode: Complete Linkage, DTW v1.9.1)")

    if tasks:
        batches = [tasks[i:i + TASK_BATCH_SIZE] for i in range(0, len(tasks), TASK_BATCH_SIZE)]
        for i, batch in enumerate(batches):
            print(f"\n🚀 批次 {i + 1}/{len(batches)} (本批 {len(batch)} 设备)...")
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_dev = {
                    executor.submit(process_device_complete_clustering, dev, device_groups[dev]): dev
                    for dev in batch
                }
                for future in tqdm(as_completed(future_to_dev), total=len(batch)):
                    dev_name = future_to_dev[future]
                    try:
                        res = future.result()
                        processed_results[dev_name] = res
                    except Exception as e:
                        print(f"❌ 错误 ({dev_name}): {e}")
            save_checkpoint(CHECKPOINT_PATH, processed_results)
            for dev in batch: del device_groups[dev]
            gc.collect()

    print("📦 合并并保存结果...")
    final_data = []
    for flows in processed_results.values():
        final_data.extend(flows)
    os.makedirs(os.path.dirname(OUTPUT_PICKLE_PATH), exist_ok=True)
    with open(OUTPUT_PICKLE_PATH, 'wb') as f:
        pickle.dump(final_data, f)
    print(f"✅ 完成! 最终流数量: {len(final_data)}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    limit_memory(200)  # 内存限制 200GB

    main()
