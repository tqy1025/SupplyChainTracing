import os
import pickle
import numpy as np
import pandas as pd
import gc
import sys
import resource
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

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
# 1. 配置参数
# ==============================================================================
# --- 路径配置 ---
BASE_DIR = "../DTW_1_12"
INPUT_DIR = "../Input"  # 外部输入文件目录

DTW_SIGMA = 35.0
SIMILARITY_THRESHOLD = 0.90
DISTANCE_THRESHOLD = 1.0 - SIMILARITY_THRESHOLD
JACCARD_THRESHOLD = 0.1  # 第二阶段聚类阈值

# 输入文件
INPUT_PKL_PATH = os.path.join(BASE_DIR, f"1.In_Device_Complete_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_v1.12.pkl")
EXTERNAL_DEVICE_LIST = os.path.join(INPUT_DIR, "device_list.csv")
EXTERNAL_DEVICE_TYPE = os.path.join(INPUT_DIR, "device_type.csv")

# 输出文件
OUTPUT_FINAL_CSV = os.path.join(BASE_DIR, f"2.Extended_Network_Flows_vendor_type_raw_v1.12.csv")

# 缓存文件 (用于加速重复运行)
MATRIX_CACHE_FILE = os.path.join(BASE_DIR, f"1.1.Full_Distance_Matrix_Complete_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_v1.12.npy")

# --- DTW 参数 (1.9.1 版) ---
DTW_TOP_N = 20
DTW_SMALL_THRESH = 100.0
DTW_SMALL_WEIGHT = 10.0
DTW_MTU_THRESH = 1400.0
DTW_MTU_WEIGHT = 1.0
DTW_BIG_THRESH = 350.0
DTW_BIG_WEIGHT = 8.0
DTW_CROSS_WEIGHT = 3.0

# ==============================================================================
# 2. 核心算法 (DTW & Feature Extraction)
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

@jit(nopython=True)
def calculate_weighted_dtw_numba(seq_a, seq_b, small_thresh, small_weight, mtu_thresh, mtu_weight, big_thresh, big_weight, cross_weight):
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

            min_val = min(prev_row[j - 1], prev_row[j], curr_row[j - 1])
            curr_row[j] = cost + min_val

        for k in range(m + 1):
            prev_row[k] = curr_row[k]

    return prev_row[m]

@jit(nopython=True, parallel=True)
def compute_full_condensed_matrix_numba(
        matrix_up, lengths_up, matrix_down, lengths_down, n,
        small_thresh, small_weight, mtu_thresh, mtu_weight,
        big_thresh, big_weight, cross_weight, sigma
):
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

            dist_up = calculate_weighted_dtw_numba(up_a, up_b, small_thresh, small_weight, mtu_thresh, mtu_weight, big_thresh, big_weight, cross_weight)
            dist_down = calculate_weighted_dtw_numba(down_a, down_b, small_thresh, small_weight, mtu_thresh, mtu_weight, big_thresh, big_weight, cross_weight)

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
# 3. 步骤函数
# ==============================================================================

def step_1_generate_details(input_path, matrix_cache_path):
    print("\n🔵 [Step 1] Loading Data & Calculating DTW Matrix...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, 'rb') as f:
        flow_data = pickle.load(f)
    n_samples = len(flow_data)
    print(f"   Samples: {n_samples}")

    # 特征提取
    feats_up_list = []
    feats_down_list = []
    for flow in flow_data:
        seq = flow.get('Payload_Sequence', [])
        u, d = extract_features_flat(seq)
        feats_up_list.append(u)
        feats_down_list.append(d)
    mat_up, len_up = convert_feats_to_matrix(feats_up_list)
    mat_down, len_down = convert_feats_to_matrix(feats_down_list)

    # 距离矩阵
    dist_matrix = None
    if os.path.exists(matrix_cache_path):
        print(f"   🚀 Loading cached matrix: {matrix_cache_path}")
        dist_matrix = np.load(matrix_cache_path)
        if len(dist_matrix) != n_samples * (n_samples - 1) // 2:
            print("   ⚠️ Cache mismatch, recomputing...")
            dist_matrix = None

    if dist_matrix is None:
        print("   🧮 Computing DTW matrix (Numba)...")
        dist_matrix = compute_full_condensed_matrix_numba(
            mat_up, len_up, mat_down, len_down, n_samples,
            DTW_SMALL_THRESH, DTW_SMALL_WEIGHT, DTW_MTU_THRESH, DTW_MTU_WEIGHT,
            DTW_BIG_THRESH, DTW_BIG_WEIGHT, DTW_CROSS_WEIGHT, DTW_SIGMA
        )
        os.makedirs(os.path.dirname(matrix_cache_path), exist_ok=True)
        np.save(matrix_cache_path, dist_matrix)

    # 第一阶段聚类
    print("   🌲 Hierarchical Clustering (Complete)...")
    Z = linkage(dist_matrix, method='complete')
    cluster_labels = fcluster(Z, t=DISTANCE_THRESHOLD, criterion='distance')

    # 生成详细 DataFrame
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
    return df_details


def step_2_super_groups(df_details):
    print("\n🔵 [Step 2] Pattern Compression & Super Grouping...")
    # 构建矩阵
    binary_matrix = df_details.pivot_table(
        index='Cluster_ID', columns='Device', aggfunc=len, fill_value=0
    )
    binary_matrix = (binary_matrix > 0).astype(np.int8)

    # 模式压缩
    matrix_values = binary_matrix.values
    index_names = binary_matrix.index.tolist()
    pattern_map = {}
    unique_patterns = []
    print(f"   Original dimensions: {binary_matrix.shape}")

    for idx, row in enumerate(matrix_values):
        comp_id = index_names[idx]
        row_tuple = tuple(row)
        row_hash = hash(row_tuple)
        if row_hash not in pattern_map:
            pattern_map[row_hash] = []
            unique_patterns.append(row)
        pattern_map[row_hash].append(comp_id)

    unique_matrix = np.array(unique_patterns, dtype=np.int8)
    unique_hashes = list(pattern_map.keys())
    print(f"   Compressed dimensions: {unique_matrix.shape}")

    # Jaccard 聚类
    if len(unique_matrix) < 2:
        print("   ⚠️ Not enough patterns to cluster.")
        return pd.DataFrame()

    dists = pdist(unique_matrix, metric='jaccard')
    Z = linkage(dists, method='average')
    labels = fcluster(Z, t=JACCARD_THRESHOLD, criterion='distance')

    # 还原结果
    final_rows = []
    device_columns = binary_matrix.columns
    for i, label in enumerate(labels):
        pattern_hash = unique_hashes[i]
        original_components = pattern_map[pattern_hash]
        row_vec = unique_matrix[i]
        device_indices = np.where(row_vec == 1)[0]
        device_names = [device_columns[idx] for idx in device_indices]
        
        dev_str = ",".join(device_names)
        
        for comp_id in original_components:
            final_rows.append({
                'Super_Group_ID': f"SG_{label:04d}",
                'Component_ID': comp_id,
                'Device_Count': len(device_names),
                'Device_List': dev_str
            })
    
    df_super = pd.DataFrame(final_rows)
    return df_super


def step_3_enrich_attributes(df_super, device_list_path, device_type_path):
    print("\n🔵 [Step 3] Enriching Attributes (Vendor & Type)...")
    if not os.path.exists(device_list_path) or not os.path.exists(device_type_path):
        print(f"   ⚠️ External files missing. Skipping enrichment.")
        # 返回空列以防报错
        df_super['vendor_state'] = 0
        df_super['vendor_name'] = ""
        df_super['device_type_state'] = 0
        df_super['device_type_detail'] = ""
        return df_super, {}, {}

    df_dev_list = pd.read_csv(device_list_path)
    df_dev_type = pd.read_csv(device_type_path)

    device_to_vendor = dict(zip(
        df_dev_list['Device_Name'].astype(str).str.strip(),
        df_dev_list['Vendor'].astype(str).str.strip()
    ))
    device_to_type = dict(zip(
        df_dev_type['Device_Name'].astype(str).str.strip(),
        df_dev_type['Type'].astype(str).str.strip()
    ))

    def analyze(device_list_str, mapping):
        if pd.isna(device_list_str) or str(device_list_str).strip() == "":
            return 0, ""
        devices = [d.strip() for d in str(device_list_str).split(',')]
        attrs = [mapping.get(d) for d in devices if mapping.get(d)]
        if not attrs: return 0, ""
        counts = Counter(attrs)
        if len(counts) == 1:
            return 0, list(counts.keys())[0]
        else:
            sorted_attrs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return 1, ",".join([f"{c}{a}" for a, c in sorted_attrs])

    df_super[['vendor_state', 'vendor_name']] = df_super['Device_List'].apply(
        lambda x: pd.Series(analyze(x, device_to_vendor))
    )
    df_super[['device_type_state', 'device_type_detail']] = df_super['Device_List'].apply(
        lambda x: pd.Series(analyze(x, device_to_type))
    )
    
    return df_super, device_to_vendor, device_to_type


def step_4_final_merge(df_super, df_details, vendor_map, type_map, output_path):
    print("\n🔵 [Step 4] Final Merge & Export...")
    
    # 统一类型
    df_super['Component_ID'] = df_super['Component_ID'].astype(str)
    df_details['Cluster_ID'] = df_details['Cluster_ID'].astype(str)

    # 合并
    df_merged = pd.merge(
        df_super, df_details,
        left_on='Component_ID', right_on='Cluster_ID',
        how='inner'
    )

    # 映射行级属性
    df_merged['Device_Clean'] = df_merged['Device'].astype(str).str.strip()
    df_merged['device_vendor'] = df_merged['Device_Clean'].map(vendor_map).fillna('')
    df_merged['device_type'] = df_merged['Device_Clean'].map(type_map).fillna('')

    # 列筛选
    target_columns = [
        'Super_Group_ID', 'Component_ID', 'Device_Count',
        'vendor_state', 'vendor_name', 
        'device_type_state', 'device_type_detail',
        'Device', 'device_vendor', 'device_type',
        'Remote_IP', 'Remote_domain', 'Protocol', 'Remote_Port',
        'Sequence_Up', 'Sequence_Down', 'Original_Index'
    ]
    available_cols = [c for c in target_columns if c in df_merged.columns]
    df_final = df_merged[available_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Final Result Saved: {output_path}")

# ==============================================================================
# Main
# ==============================================================================

def main():
    multiprocessing_support()
    
    # 检查输入
    if not os.path.exists(INPUT_PKL_PATH):
        print(f"❌ Input file not found: {INPUT_PKL_PATH}")
        return

    # --- Step 1: DTW & Basic Clustering ---
    df_details = step_1_generate_details(INPUT_PKL_PATH, MATRIX_CACHE_FILE)
    gc.collect()

    # --- Step 2: Super Grouping ---
    df_super = step_2_super_groups(df_details)
    if df_super.empty:
        print("❌ Clustering failed or no patterns found.")
        return
    gc.collect()

    # --- Step 3: Enrich Attributes ---
    # 自动生成测试外部文件(如果不存在)，以便代码可运行
    ensure_mock_external_files() 
    
    df_enriched, v_map, t_map = step_3_enrich_attributes(
        df_super, EXTERNAL_DEVICE_LIST, EXTERNAL_DEVICE_TYPE
    )

    # --- Step 4: Final Merge ---
    step_4_final_merge(df_enriched, df_details, v_map, t_map, OUTPUT_FINAL_CSV)

def multiprocessing_support():
    # Windows/Mac 兼容性
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except:
        pass

def ensure_mock_external_files():
    """如果外部输入文件不存在，生成简单的Mock数据，保证流程能跑通 (可选)"""
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
    
    if not os.path.exists(EXTERNAL_DEVICE_LIST):
        print("⚠️ Device list not found, creating dummy file for testing...")
        pd.DataFrame({
            'Device_Name': ['Device_A', 'Device_B'], 'Vendor': ['Vendor_A', 'Vendor_B']
        }).to_csv(EXTERNAL_DEVICE_LIST, index=False)
        
    if not os.path.exists(EXTERNAL_DEVICE_TYPE):
        print("⚠️ Device type not found, creating dummy file for testing...")
        pd.DataFrame({
            'Device_Name': ['Device_A', 'Device_B'], 'Type': ['Type_A', 'Type_B']
        }).to_csv(EXTERNAL_DEVICE_TYPE, index=False)

if __name__ == '__main__':
    main()
