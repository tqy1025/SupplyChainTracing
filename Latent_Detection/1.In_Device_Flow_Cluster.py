import os
import pickle
import numpy as np
import ipaddress
import gc
import resource
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

    def jit(nopython=True, parallel=False):
        def decorator(func):
            return func
        return decorator

    def prange(n):
        return range(n)

# ==============================================================================
# 1. 参数配置区域
# ==============================================================================

# --- 输入/输出路径 ---
INPUT_PICKLE_PATH = "../Flow_Preprocessing/Traffic_Cleaned_v3_Final.pkl"
DTW_SIGMA = 35.0
SIMILARITY_THRESHOLD = 0.90
DISTANCE_THRESHOLD = 1.0 - SIMILARITY_THRESHOLD

OUTPUT_PICKLE_PATH = f"./1.In_Device_Complete_from_v3Final_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_nopad.pkl"
CHECKPOINT_PATH = f"./checkpoint_in_device_from_v3Final_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_nopad.pkl"

# --- 设备规模控制 ---
# 单设备参与聚类的最大流数量（防止全连接 O(N^2) 爆炸）
MAX_FLOWS_PER_DEVICE = 1000000

# --- DTW 序列长度上限（完全无 padding） ---
# up/down 分别最多保留前 MAX_SEQ_LEN 个（保持原顺序）
MAX_SEQ_LEN = 100

# --- DTW 参数 (1.9.1 风格) ---
DTW_SMALL_THRESH = 100.0
DTW_SMALL_WEIGHT = 10.0

DTW_MTU_THRESH = 1400.0
DTW_MTU_WEIGHT = 1.0

DTW_BIG_THRESH = 350.0
DTW_BIG_WEIGHT = 8.0

DTW_CROSS_WEIGHT = 3.0

# --- 并行配置 ---
NUM_WORKERS = 32
TASK_BATCH_SIZE = 50

# --- 可选：轻量校验（不建议再做重过滤） ---
# 如果你确认 v3_final 已彻底清洗，可把 ENABLE_LIGHT_CHECK 设为 False
ENABLE_LIGHT_CHECK = True
MIN_PACKET_COUNT = 5  # 仅用于轻量校验（up/down 最小包数）
DROP_PRIVATE_IP = False  # v2 已做过；这里默认不丢弃
DROP_DNS_FLOW = False    # v2 已做过；这里默认不丢弃

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

def limit_memory(maxsize_gb: int):
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        maxsize_bytes = int(maxsize_gb) * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (maxsize_bytes, hard))
    except Exception:
        pass


def is_private_ip(ip_str):
    if not ip_str:
        return False
    try:
        clean_ip = str(ip_str).split(':')[0]
        ip_obj = ipaddress.ip_address(clean_ip)
        return ip_obj.is_private
    except Exception:
        return False


def is_dns_flow(remote_ip, protocol):
    if protocol != 'UDP':
        return False
    if not remote_ip:
        return False
    clean_ip = str(remote_ip).split(':')[0]
    return clean_ip in KNOWN_PUBLIC_DNS_IPS


def extract_dtw_features(signed_seq, max_seq_len=None):
    """
    signed_seq: Payload_Sequence（带符号）
    max_seq_len:
      - None: 不截断
      - int : up/down 分别最多取 max_seq_len 个（按原顺序）
    """
    if not signed_seq:
        return {'up': np.array([], dtype=np.float64), 'down': np.array([], dtype=np.float64)}

    if max_seq_len is None:
        up = np.array([abs(x) for x in signed_seq if x > 0], dtype=np.float64)
        down = np.array([abs(x) for x in signed_seq if x < 0], dtype=np.float64)
        return {'up': up, 'down': down}

    up_list = []
    down_list = []
    for x in signed_seq:
        if x > 0:
            if len(up_list) < max_seq_len:
                up_list.append(abs(x))
        elif x < 0:
            if len(down_list) < max_seq_len:
                down_list.append(abs(x))
        if len(up_list) >= max_seq_len and len(down_list) >= max_seq_len:
            break

    return {
        'up': np.array(up_list, dtype=np.float64),
        'down': np.array(down_list, dtype=np.float64),
    }


def pack_variable_length_sequences(feats_list):
    """
    将变长序列拼成 1D 大数组 + offsets/lengths，实现 0 padding。
    """
    n = len(feats_list)
    lengths = np.empty(n, dtype=np.int32)
    offsets = np.empty(n, dtype=np.int64)

    total = 0
    for i, arr in enumerate(feats_list):
        l = int(arr.size)
        lengths[i] = l
        offsets[i] = total
        total += l

    flat = np.empty(total, dtype=np.float64)
    pos = 0
    for arr in feats_list:
        l = int(arr.size)
        if l > 0:
            flat[pos:pos + l] = arr
        pos += l

    return flat, offsets, lengths


# ==============================================================================
# 3. DTW 核心计算 (1.9.1 风格 - Numba 优化)
# ==============================================================================

@jit(nopython=True)
def calculate_weighted_dtw_numba(seq_a, seq_b,
                                 small_thresh, small_weight,
                                 mtu_thresh, mtu_weight,
                                 big_thresh, big_weight,
                                 cross_weight):
    n = len(seq_a)
    m = len(seq_b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1e15

    prev_row = np.empty(m + 1, dtype=np.float64)
    curr_row = np.empty(m + 1, dtype=np.float64)
    prev_row[:] = 1e15
    prev_row[0] = 0.0

    for i in range(1, n + 1):
        curr_row[:] = 1e15
        val_a = seq_a[i - 1]

        is_small_a = val_a < small_thresh
        is_big_a = val_a > big_thresh
        is_mtu_a = val_a > mtu_thresh

        for j in range(1, m + 1):
            val_b = seq_b[j - 1]
            base_diff = abs(val_a - val_b)

            is_small_b = val_b < small_thresh
            is_big_b = val_b > big_thresh
            is_mtu_b = val_b > mtu_thresh

            if is_small_a and is_small_b:
                cost = base_diff * small_weight
            elif is_mtu_a and is_mtu_b:
                cost = base_diff / mtu_weight
            elif is_big_a and is_big_b:
                cost = base_diff / big_weight
            elif is_small_a != is_small_b:
                cost = base_diff * cross_weight
            else:
                cost = base_diff

            v_diag = prev_row[j - 1]
            v_up = prev_row[j]
            v_left = curr_row[j - 1]
            min_val = v_diag
            if v_up < min_val:
                min_val = v_up
            if v_left < min_val:
                min_val = v_left

            curr_row[j] = cost + min_val

        for k in range(m + 1):
            prev_row[k] = curr_row[k]

    return prev_row[m]


@jit(nopython=True, parallel=True)
def compute_condensed_distance_matrix_numba_varlen(
        flat_up, offsets_up, lengths_up,
        flat_down, offsets_down, lengths_down,
        n,
        small_thresh, small_weight,
        mtu_thresh, mtu_weight,
        big_thresh, big_weight,
        cross_weight, sigma
):
    """
    输入为 flat + offsets + lengths，完全无 padding，输出 condensed 距离矩阵。
    """
    dist_len = n * (n - 1) // 2
    dist_array = np.empty(dist_len, dtype=np.float64)

    for i in prange(n):
        len_u_a = lengths_up[i]
        len_d_a = lengths_down[i]
        off_u_a = offsets_up[i]
        off_d_a = offsets_down[i]

        up_a = flat_up[off_u_a:off_u_a + len_u_a]
        down_a = flat_down[off_d_a:off_d_a + len_d_a]

        row_offset = i * n - (i * (i + 1)) // 2

        for j in range(i + 1, n):
            len_u_b = lengths_up[j]
            len_d_b = lengths_down[j]
            off_u_b = offsets_up[j]
            off_d_b = offsets_down[j]

            up_b = flat_up[off_u_b:off_u_b + len_u_b]
            down_b = flat_down[off_d_b:off_d_b + len_d_b]

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

            mean_len_up = (len_u_a + len_u_b) / 2.0
            mean_len_down = (len_d_a + len_d_b) / 2.0

            norm_up = dist_up / mean_len_up if mean_len_up > 0 else 1e9
            norm_down = dist_down / mean_len_down if mean_len_down > 0 else 1e9

            sim_up = np.exp(-norm_up / sigma) if mean_len_up > 0 else 0.0
            sim_down = np.exp(-norm_down / sigma) if mean_len_down > 0 else 0.0

            final_sim = np.sqrt(sim_up * sim_down)
            d = 1.0 - final_sim
            if d < 0.0:
                d = 0.0

            k = row_offset + (j - i - 1)
            dist_array[k] = d

    return dist_array


# ==============================================================================
# 4. 单设备聚类逻辑（Complete linkage）
# ==============================================================================

def light_check(flow):
    """
    可选：轻量校验（不做强过滤）
    """
    if not ENABLE_LIGHT_CHECK:
        return True

    seq = flow.get('Payload_Sequence', [])
    up_cnt = 0
    down_cnt = 0
    for x in seq:
        if x > 0:
            up_cnt += 1
        elif x < 0:
            down_cnt += 1
    if up_cnt < MIN_PACKET_COUNT or down_cnt < MIN_PACKET_COUNT:
        return False

    if DROP_PRIVATE_IP and is_private_ip(flow.get('Remote_IP')):
        return False

    if DROP_DNS_FLOW and is_dns_flow(flow.get('Remote_IP'), flow.get('Protocol')):
        return False

    return True


def process_device_complete_clustering(device_name, flows_for_device):
    """
    输入：某个 device 的 flows list（原 flow dict）
    输出：每个簇一个 representative（原 flow dict，保留 Domain_A / Domain_PTR 等字段）
    """
    # 可选轻量校验
    if ENABLE_LIGHT_CHECK:
        flows_for_device = [f for f in flows_for_device if light_check(f)]

    # 按序列长度排序，优先保留信息量更大的流
    flows_for_device.sort(key=lambda x: len(x.get('Payload_Sequence', [])), reverse=True)

    # 设备内流太多则截断
    if len(flows_for_device) > MAX_FLOWS_PER_DEVICE:
        flows_for_device = flows_for_device[:MAX_FLOWS_PER_DEVICE]

    n = len(flows_for_device)
    if n == 0:
        return []
    if n == 1:
        return [flows_for_device[0]]

    # 提取 DTW 特征（up/down 变长，cap 到 MAX_SEQ_LEN）
    feats_up_list = []
    feats_down_list = []
    for flow in flows_for_device:
        feats = extract_dtw_features(flow.get('Payload_Sequence', []), max_seq_len=MAX_SEQ_LEN)
        feats_up_list.append(feats['up'])
        feats_down_list.append(feats['down'])

    # 完全无 padding 打包
    flat_up, offsets_up, lengths_up = pack_variable_length_sequences(feats_up_list)
    flat_down, offsets_down, lengths_down = pack_variable_length_sequences(feats_down_list)

    # condensed 距离矩阵
    try:
        condensed = compute_condensed_distance_matrix_numba_varlen(
            flat_up, offsets_up, lengths_up,
            flat_down, offsets_down, lengths_down,
            n,
            DTW_SMALL_THRESH, DTW_SMALL_WEIGHT,
            DTW_MTU_THRESH, DTW_MTU_WEIGHT,
            DTW_BIG_THRESH, DTW_BIG_WEIGHT,
            DTW_CROSS_WEIGHT, DTW_SIGMA
        )
    except Exception as e:
        print(f"❌ DTW matrix error ({device_name}): {e}")
        # 回退：不聚类，全部返回（保字段）
        return flows_for_device

    # complete linkage 聚类
    try:
        Z = linkage(condensed, method='complete')
        labels = fcluster(Z, t=DISTANCE_THRESHOLD, criterion='distance')
    except Exception as e:
        print(f"⚠️ Linkage error ({device_name}): {e}")
        return flows_for_device

    # 按簇选 representative：取 Payload_Sequence 最长者
    cluster_groups = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_groups[lab].append(flows_for_device[idx])

    representatives = []
    for members in cluster_groups.values():
        best = max(members, key=lambda x: len(x.get('Payload_Sequence', [])))
        representatives.append(best)

    return representatives


# ==============================================================================
# 5. checkpoint（断点续跑）
# ==============================================================================

def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(path, data):
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(data, f)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


# ==============================================================================
# 6. 主程序
# ==============================================================================

def main():
    if not os.path.exists(INPUT_PICKLE_PATH):
        print(f"❌ Input file not found: {INPUT_PICKLE_PATH}")
        return

    print("⏳ Loading v3 final data...")
    raw_data = pd_read_pickle_compatible(INPUT_PICKLE_PATH)
    if raw_data is None:
        return

    # raw_data 统一成 list[dict]
    if isinstance(raw_data, list):
        all_flows = raw_data
    elif isinstance(raw_data, dict):
        # 如果是 dict，拍平
        tmp = []
        for v in raw_data.values():
            if isinstance(v, list):
                tmp.extend(v)
            else:
                tmp.append(v)
        all_flows = tmp
    else:
        # 如果是 DataFrame，转 dict list
        try:
            import pandas as pd
            if hasattr(raw_data, "to_dict"):
                all_flows = raw_data.to_dict(orient="records")
            else:
                print("❌ Unsupported data format from pickle.")
                return
        except Exception:
            print("❌ Unsupported data format from pickle.")
            return

    print(f"📊 Total flows loaded: {len(all_flows)}")

    # 分组：Device -> flows
    device_groups = defaultdict(list)
    missing_device = 0
    for flow in all_flows:
        dev = flow.get('Device', None)
        if dev is None:
            missing_device += 1
            dev = "Unknown"
        device_groups[str(dev)].append(flow)

    print(f"🧩 Devices: {len(device_groups)} (missing Device field: {missing_device})")

    # 释放内存
    del all_flows
    gc.collect()

    processed = load_checkpoint(CHECKPOINT_PATH)
    all_devs = list(device_groups.keys())
    tasks = [d for d in all_devs if d not in processed]

    # 仅保留待处理设备的数据，减少内存
    for d in all_devs:
        if d not in tasks:
            del device_groups[d]
    gc.collect()

    print(f"📋 Pending devices: {len(tasks)} (Complete linkage, DTW nopad, cap={MAX_SEQ_LEN}, max_flows/dev={MAX_FLOWS_PER_DEVICE})")

    if tasks:
        batches = [tasks[i:i + TASK_BATCH_SIZE] for i in range(0, len(tasks), TASK_BATCH_SIZE)]
        for bi, batch in enumerate(batches, 1):
            print(f"\n🚀 Batch {bi}/{len(batches)} (devices={len(batch)})")
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
                future_to_dev = {ex.submit(process_device_complete_clustering, dev, device_groups[dev]): dev for dev in batch}
                for fut in tqdm(as_completed(future_to_dev), total=len(batch), desc="Device Clustering"):
                    dev = future_to_dev[fut]
                    try:
                        reps = fut.result()
                        processed[dev] = reps
                    except Exception as e:
                        print(f"❌ Error ({dev}): {e}")

            # 保存 checkpoint
            save_checkpoint(CHECKPOINT_PATH, processed)

            # 释放本批设备数据
            for dev in batch:
                if dev in device_groups:
                    del device_groups[dev]
            gc.collect()

    # 合并输出：所有设备的 representatives 合在一个 list
    print("📦 Merging representatives and saving...")
    final_reps = []
    for reps in processed.values():
        final_reps.extend(reps)

    # 核心保证：原 flow dict 原样输出（包含 Domain_A / Domain_PTR）
    # 这里给一个小检查输出（不影响数据）
    sample = None
    for x in final_reps:
        if isinstance(x, dict):
            sample = x
            break
    if sample is not None:
        print("🔎 Output sample keys include:", [k for k in ["Domain_A", "Domain_PTR", "Device", "Remote_IP"] if k in sample])

    os.makedirs(os.path.dirname(OUTPUT_PICKLE_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PICKLE_PATH, 'wb') as f:
        pickle.dump(final_reps, f)

    print(f"✅ Done! Output reps: {len(final_reps)}")
    print(f"✅ Saved to: {OUTPUT_PICKLE_PATH}")


def pd_read_pickle_compatible(path):
    """
    兼容：pickle 可能是 list/dict/df
    """
    try:
        # 优先 pandas（若可用）
        import pandas as pd
        return pd.read_pickle(path)
    except Exception:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Failed to load pickle: {e}")
            return None


if __name__ == '__main__':
    multiprocessing.freeze_support()
    limit_memory(200)  # 内存限制 200GB（按需改）
    main()
