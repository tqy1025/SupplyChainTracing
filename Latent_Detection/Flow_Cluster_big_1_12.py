import os
import pickle
import numpy as np
import ipaddress
import pandas as pd
import random
import gc
import sys
import csv
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import resource
from scipy.cluster.hierarchy import linkage, fcluster

# ==============================================================================
# 0. Numba 环境配置
# ==============================================================================
try:
    from numba import jit, prange

    HAS_NUMBA = True
    print("✅ Numba detected! High-speed acceleration enabled.")
except ImportError:
    HAS_NUMBA = False
    print("⚠️ Numba not found. Install via 'pip install numba'")


    def jit(nopython=True, parallel=False):
        def decorator(func): return func

        return decorator


    def prange(n):
        return range(n)

# ==============================================================================
# 1. 配置参数 (1.9.1 版)
# ==============================================================================
DTW_TOP_N = 20

# 1.9.1 版参数：固定权重 + MTU 分区
DTW_SMALL_THRESH = 100.0
DTW_SMALL_WEIGHT = 10.0   # 严打小包：控制类包必须精准 (Diff * 20)

DTW_MTU_THRESH = 1400.0   # [1.9.1] MTU 阈值
DTW_MTU_WEIGHT = 1.0      # [1.9.1] MTU 权重：不予优惠，精准区分 (Diff / 1.0)

DTW_BIG_THRESH = 350.0    # 大包阈值：普通载荷
DTW_BIG_WEIGHT = 8.0      # [1.9.1] 大包优惠：容忍载荷波动 (Diff / 5.0)

DTW_CROSS_WEIGHT = 3.0    # 结构性惩罚：小包 vs 非小包

DTW_SIGMA = 35.0         # [1.9.1] Sigma 更新

SIMILARITY_THRESHOLD = 0.9
# 距离阈值
DISTANCE_THRESHOLD = 1.0 - SIMILARITY_THRESHOLD

# [修改] 输入数据 (Stage 1 v1.9.1 的输出)
INPUT_PATH = f"../DTW_1_12/1.In_Device_Complete_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_v1.12.pkl"

# [修改] 输出路径
OUTPUT_DIR = "../DTW_1_12"
OUTPUT_MATRIX_XLSX = os.path.join(OUTPUT_DIR,
                                  f"2.2.Device_Component_Matrix_Complete_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_v1.12.xlsx")
OUTPUT_DETAILED_CSV = os.path.join(OUTPUT_DIR,
                                   f"2.1.Detailed_Cluster_Flows_Complete_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_v1.12.csv")
MATRIX_CACHE_FILE = os.path.join(OUTPUT_DIR,
                                 f"2.3.Full_Distance_Matrix_Complete_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_v1.12.npy")


# ==============================================================================
# 2. 辅助工具
# ==============================================================================

def get_best_domain_name(flow_dict):
    potential_keys = ['Remote_Hostname', 'SNI', 'Server_Name', 'HTTP_Host', 'Domain', 'dns_query', 'host']
    for key in potential_keys:
        val = flow_dict.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 0:
            if any(c.isalpha() for c in val):
                return val.strip()
    return ""


def extract_features_flat(signed_seq):
    if not signed_seq:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    up = np.array([abs(x) for x in signed_seq if x > 0][:DTW_TOP_N], dtype=np.float64)
    down = np.array([abs(x) for x in signed_seq if x < 0][:DTW_TOP_N], dtype=np.float64)
    return up, down


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
# 3. DTW 核心 (1.9.1 版 - Numba 优化)
# ==============================================================================

@jit(nopython=True)
def calculate_weighted_dtw_numba(seq_a, seq_b,
                                 small_thresh, small_weight,
                                 mtu_thresh, mtu_weight,      # New
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
            # ---------------------------

            min_val = min(prev_row[j - 1], prev_row[j], curr_row[j - 1])
            curr_row[j] = cost + min_val

        for k in range(m + 1):
            prev_row[k] = curr_row[k]

    return prev_row[m]


# ==============================================================================
# 4. 全局距离矩阵计算 (Numba Parallel + Matrix Input)
# ==============================================================================

@jit(nopython=True, parallel=True)
def compute_full_condensed_matrix_numba(
        matrix_up, lengths_up,
        matrix_down, lengths_down,
        n,
        small_thresh, small_weight,
        mtu_thresh, mtu_weight,      # New
        big_thresh, big_weight,
        cross_weight, sigma
):
    """
    计算全局 N*N 的压缩距离矩阵。
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

            # --- DTW (1.9.1) ---
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

            # Norm
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
# 5. 主流程
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print(f"❌ 未找到输入文件: {INPUT_PATH}")
        return

    print(f"📂 加载流数据: {INPUT_PATH}")
    with open(INPUT_PATH, 'rb') as f:
        flow_data = pickle.load(f)

    n_samples = len(flow_data)
    print(f"📊 总样本数: {n_samples}")
    if n_samples < 2:
        print("⚠️ 样本太少，无法聚类。")
        return

    # 1. 准备数据
    print("🔄 转换特征格式 (Matrix Padding)...")
    feats_up_list = []
    feats_down_list = []
    for flow in flow_data:
        seq = flow.get('Payload_Sequence', [])
        u, d = extract_features_flat(seq)
        feats_up_list.append(u)
        feats_down_list.append(d)
    mat_up, len_up = convert_feats_to_matrix(feats_up_list, max_len=DTW_TOP_N)
    mat_down, len_down = convert_feats_to_matrix(feats_down_list, max_len=DTW_TOP_N)

    # 2. 计算/加载距离矩阵
    if os.path.exists(MATRIX_CACHE_FILE):
        print(f"🚀 发现矩阵缓存: {MATRIX_CACHE_FILE}, 直接加载...")
        dist_matrix = np.load(MATRIX_CACHE_FILE)
        expected_len = n_samples * (n_samples - 1) // 2
        if len(dist_matrix) != expected_len:
            print("⚠️ 缓存长度不匹配，重新计算。")
            dist_matrix = None
    else:
        dist_matrix = None

    if dist_matrix is None:
        print(f"🧮 开始计算距离矩阵 (N={n_samples}, DTW v1.9.1, Parallel)...")
        try:
            # [修改] 传入 1.9.1 版参数
            dist_matrix = compute_full_condensed_matrix_numba(
                mat_up, len_up,
                mat_down, len_down,
                n_samples,
                DTW_SMALL_THRESH, DTW_SMALL_WEIGHT,
                DTW_MTU_THRESH, DTW_MTU_WEIGHT,
                DTW_BIG_THRESH, DTW_BIG_WEIGHT,
                DTW_CROSS_WEIGHT, DTW_SIGMA
            )
            print(f"💾 保存矩阵缓存: {MATRIX_CACHE_FILE}")
            np.save(MATRIX_CACHE_FILE, dist_matrix)
        except Exception as e:
            print(f"❌ 计算失败: {e}")
            return

    # 3. 聚类
    print("🌲 层次聚类 (Method: Complete Linkage)...")
    Z = linkage(dist_matrix, method='complete')
    print(f"✂️ 切分 (Distance Threshold={DISTANCE_THRESHOLD:.4f})...")
    cluster_labels = fcluster(Z, t=DISTANCE_THRESHOLD, criterion='distance')

    # 4. 生成报告
    print("📝 生成详细清单...")
    detailed_rows = []
    for idx, label in enumerate(cluster_labels):
        flow = flow_data[idx]
        seq = flow.get('Payload_Sequence', [])
        u, d = extract_features_flat(seq)
        detailed_rows.append({
            'Cluster_ID': f"Comp_{label:04d}",
            'Device': flow.get('Device', 'Unknown'),
            'Remote_IP': flow.get('Remote_IP', ''),
            'Remote_domain': get_best_domain_name(flow),
            'Protocol': flow.get('Protocol', ''),
            'Remote_Port': flow.get('Remote_Port', ''),
            'Sequence_Up': str(u.tolist()),
            'Sequence_Down': str(d.tolist()),
            'Original_Index': idx
        })

    df_details = pd.DataFrame(detailed_rows)
    df_details = df_details.sort_values(by=['Cluster_ID', 'Device'])
    df_details.to_csv(OUTPUT_DETAILED_CSV, index=False, encoding='utf-8-sig')

    print("📝 生成矩阵表...")
    binary_matrix = df_details.pivot_table(
        index='Cluster_ID',
        columns='Device',
        values='Original_Index',
        aggfunc=lambda x: 1,
        fill_value=0
    )
    binary_matrix['Device_Count'] = binary_matrix.sum(axis=1)
    binary_matrix = binary_matrix.sort_values('Device_Count', ascending=False)
    del binary_matrix['Device_Count']

    binary_matrix.to_excel(OUTPUT_MATRIX_XLSX)
    print("✅ 完成！")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except:
        pass
    main()