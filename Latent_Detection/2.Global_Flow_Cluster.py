import os
import pickle
import numpy as np
import pandas as pd
import gc
import resource
from collections import defaultdict, Counter
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
        def decorator(func):
            return func
        return decorator

    def prange(n):
        return range(n)

# ==============================================================================
# 1. 配置参数
# ==============================================================================
BASE_DIR = "./"
INPUT_DIR = "../Data"

DTW_SIGMA = 35.0
SIMILARITY_THRESHOLD = 0.90
DISTANCE_THRESHOLD = 1.0 - SIMILARITY_THRESHOLD

# [MOD] 全局聚类：序列 cap（None=不限制；int=up/down 分别最多取前 MAX_SEQ_LEN 个）
MAX_SEQ_LEN = 100

# 输入：来自 v3Final 的“设备内聚类代表流”结果（你前一步生成的）
INPUT_PKL_PATH = os.path.join(
    BASE_DIR,
    f"1.In_Device_Complete_from_v3Final_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_nopad.pkl"
)

# 外部属性表
EXTERNAL_DEVICE_LIST = os.path.join(INPUT_DIR, "device_list.csv")
EXTERNAL_DEVICE_TYPE = os.path.join(INPUT_DIR, "device_type.csv")

# 输出 CSV
OUTPUT_FINAL_CSV = os.path.join(
    BASE_DIR,
    f"2.Global_Component_from_v3Final_vendor_type_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_cap{MAX_SEQ_LEN}_nopad.csv"
)

# condensed 距离矩阵缓存（防止重复计算）
MATRIX_CACHE_FILE = os.path.join(
    BASE_DIR,
    f"1.1.Global_Full_Distance_Matrix_Complete_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_cap{MAX_SEQ_LEN}_nopad_from_v3Final.npy"
)

# DTW 参数（1.9.1 风格）
DTW_SMALL_THRESH = 100.0
DTW_SMALL_WEIGHT = 10.0
DTW_MTU_THRESH = 1400.0
DTW_MTU_WEIGHT = 1.0
DTW_BIG_THRESH = 350.0
DTW_BIG_WEIGHT = 8.0
DTW_CROSS_WEIGHT = 3.0

# ==============================================================================
# 2. 工具函数
# ==============================================================================

def limit_memory(maxsize_gb: int):
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        maxsize_bytes = int(maxsize_gb) * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (maxsize_bytes, hard))
    except Exception:
        pass


def safe_load_pickle(path):
    """
    兼容 list/dict/df
    返回 list[dict]
    """
    try:
        obj = pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            obj = pickle.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        tmp = []
        for v in obj.values():
            if isinstance(v, list):
                tmp.extend(v)
            else:
                tmp.append(v)
        return tmp
    if hasattr(obj, "to_dict"):
        return obj.to_dict(orient="records")

    raise ValueError("Unsupported pickle content format.")


def get_best_domain_name(flow_dict):
    """
    兜底域名展示字段：从多个可能的 key 中取一个可读域名
    注意：这不是 Domain_A / Domain_PTR，只是用于展示/补充
    """
    potential_keys = [
        'Domain_A', 'Domain_PTR',  # 优先使用你 v3Final 的结果（若存在）
        'Remote_Hostname', 'SNI', 'Server_Name', 'HTTP_Host', 'Domain', 'dns_query', 'host'
    ]
    for key in potential_keys:
        val = flow_dict.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 0:
            if any(c.isalpha() for c in val):
                return val.strip()
    return ""


def extract_features_flat(signed_seq, max_seq_len=None):
    """
    完全无 padding 的特征提取：
    - up/down 分别提取 abs 值并保持原顺序
    - 可选 cap：up/down 各自最多取前 max_seq_len 个
    """
    if not signed_seq:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if max_seq_len is None:
        up = np.array([abs(x) for x in signed_seq if x > 0], dtype=np.float64)
        down = np.array([abs(x) for x in signed_seq if x < 0], dtype=np.float64)
        return up, down

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

    return np.array(up_list, dtype=np.float64), np.array(down_list, dtype=np.float64)


def pack_variable_length_sequences(feats_list):
    """
    变长序列打包：flat + offsets + lengths（完全无 padding）
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
# 3. DTW 核心计算（Numba）
# ==============================================================================

@jit(nopython=True)
def calculate_weighted_dtw_numba(
        seq_a, seq_b,
        small_thresh, small_weight,
        mtu_thresh, mtu_weight,
        big_thresh, big_weight,
        cross_weight
):
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

            v1 = prev_row[j - 1]
            v2 = prev_row[j]
            v3 = curr_row[j - 1]
            min_val = v1
            if v2 < min_val:
                min_val = v2
            if v3 < min_val:
                min_val = v3
            curr_row[j] = cost + min_val

        for k in range(m + 1):
            prev_row[k] = curr_row[k]

    return prev_row[m]


@jit(nopython=True, parallel=True)
def compute_full_condensed_matrix_numba_varlen(
        flat_up, offsets_up, lengths_up,
        flat_down, offsets_down, lengths_down,
        n,
        small_thresh, small_weight, mtu_thresh, mtu_weight,
        big_thresh, big_weight, cross_weight, sigma
):
    """
    完全无 padding 的 condensed 距离矩阵计算
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
# 4. Step 1：全局组件聚类（Complete linkage）
# ==============================================================================

def step_1_global_component_clustering(flow_data, matrix_cache_path):
    """
    输入：代表流列表 flow_data (list[dict])
    输出：df_details（每条代表流一行，包含全局 Component_ID 等字段）
    """
    print("\n🔵 [Step 1] Global DTW Matrix + Complete Linkage Clustering...")

    n_samples = len(flow_data)
    print(f"   Samples: {n_samples}")

    # 特征提取（cap）+ 变长打包（无 padding）
    feats_up_list = []
    feats_down_list = []
    for flow in flow_data:
        seq = flow.get('Payload_Sequence', [])
        u, d = extract_features_flat(seq, max_seq_len=MAX_SEQ_LEN)
        feats_up_list.append(u)
        feats_down_list.append(d)

    flat_up, offsets_up, lengths_up = pack_variable_length_sequences(feats_up_list)
    flat_down, offsets_down, lengths_down = pack_variable_length_sequences(feats_down_list)

    # condensed 距离矩阵（cache）
    expected_len = n_samples * (n_samples - 1) // 2
    dist_matrix = None

    if os.path.exists(matrix_cache_path):
        print(f"   🚀 Loading cached matrix: {matrix_cache_path}")
        dist_matrix = np.load(matrix_cache_path)
        if len(dist_matrix) != expected_len:
            print("   ⚠️ Cache mismatch, recomputing...")
            dist_matrix = None

    if dist_matrix is None:
        print("   🧮 Computing DTW condensed matrix (Numba, varlen, no padding)...")
        dist_matrix = compute_full_condensed_matrix_numba_varlen(
            flat_up, offsets_up, lengths_up,
            flat_down, offsets_down, lengths_down,
            n_samples,
            DTW_SMALL_THRESH, DTW_SMALL_WEIGHT, DTW_MTU_THRESH, DTW_MTU_WEIGHT,
            DTW_BIG_THRESH, DTW_BIG_WEIGHT, DTW_CROSS_WEIGHT, DTW_SIGMA
        )
        os.makedirs(os.path.dirname(matrix_cache_path) or ".", exist_ok=True)
        np.save(matrix_cache_path, dist_matrix)

    # complete linkage 聚类
    print("   🌲 Hierarchical Clustering (Complete)...")
    Z = linkage(dist_matrix, method='complete')
    cluster_labels = fcluster(Z, t=DISTANCE_THRESHOLD, criterion='distance')

    # 构建 df_details（保留 Domain_A / Domain_PTR）
    rows = []
    for idx, lab in enumerate(cluster_labels):
        flow = flow_data[idx]
        seq = flow.get('Payload_Sequence', [])
        u, d = extract_features_flat(seq, max_seq_len=MAX_SEQ_LEN)

        rows.append({
            # 组件 id（全局 DTW 聚类）
            "Component_ID": f"Comp_{int(lab):05d}",

            # 代表流的设备信息
            "Device": str(flow.get("Device", "Unknown")),

            # 设备侧 vendor/type（后面 Step 3/4 再补，但先保留原字段不丢）
            # 网络侧
            "Remote_IP": flow.get("Remote_IP", ""),
            "Remote_Port": flow.get("Remote_Port", ""),
            "Protocol": flow.get("Protocol", ""),

            # v3Final 域名字段：强制保留（存在则写出）
            "Domain_A": flow.get("Domain_A", ""),
            "Domain_PTR": flow.get("Domain_PTR", ""),

            # 一个便于查看的 domain 展示字段（兜底）
            "Remote_domain_best": get_best_domain_name(flow),

            # DTW 序列（cap 后用于解释）
            "Sequence_Up": str(u.tolist()),
            "Sequence_Down": str(d.tolist()),

            # 回溯索引
            "Original_Index": idx
        })

    df_details = pd.DataFrame(rows)
    return df_details


# ==============================================================================
# 5. Step 3：补充 vendor/type 组件级属性
# ==============================================================================

def step_3_enrich_component_vendor_type(df_details, device_list_path, device_type_path):
    """
    输出：
      df_components: 组件级表（每个 Component_ID 一行）
      device_to_vendor, device_to_type: 映射字典
    """
    print("\n🔵 [Step 3] Enrich Component Attributes (Vendor & Type)...")

    # 读外部映射
    if not os.path.exists(device_list_path) or not os.path.exists(device_type_path):
        print("   ⚠️ External mapping files missing. Continue with empty mappings.")
        device_to_vendor = {}
        device_to_type = {}
    else:
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

    # 组件级聚合：Component_ID -> Device list
    comp_to_devices = df_details.groupby("Component_ID")["Device"].apply(list).to_dict()

    def analyze_devices(dev_list, mapping):
        """
        state=0: 全一致；state=1: 多样
        detail:
          - state=0: 唯一值
          - state=1: 计数字符串，如 '3Google,1Amazon'
        """
        if not dev_list:
            return 0, ""

        attrs = []
        for d in dev_list:
            v = mapping.get(str(d).strip())
            if v and str(v).strip():
                attrs.append(str(v).strip())

        if not attrs:
            return 0, ""

        counts = Counter(attrs)
        if len(counts) == 1:
            return 0, next(iter(counts.keys()))
        else:
            sorted_attrs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            detail = ",".join([f"{c}{a}" for a, c in sorted_attrs])
            return 1, detail

    comp_rows = []
    for comp_id, devs in comp_to_devices.items():
        devs_unique = sorted(list(set([str(x) for x in devs])))

        vendor_state, vendor_detail = analyze_devices(devs_unique, device_to_vendor)
        type_state, type_detail = analyze_devices(devs_unique, device_to_type)

        comp_rows.append({
            "Component_ID": comp_id,
            "Device_Count": len(devs_unique),
            "Device_List": ",".join(devs_unique),

            "vendor_state": vendor_state,
            "vendor_detail": vendor_detail,

            "device_type_state": type_state,
            "device_type_detail": type_detail,
        })

    df_components = pd.DataFrame(comp_rows)
    return df_components, device_to_vendor, device_to_type


# ==============================================================================
# 6. Step 4：组件级信息回灌到流级，并导出 CSV
# ==============================================================================

def step_4_merge_and_export(df_components, df_details, device_to_vendor, device_to_type, output_path):
    print("\n🔵 [Step 4] Merge Component-level Info Back to Flow-level & Export CSV...")

    # 回灌：组件级 -> 行级
    df_merged = pd.merge(df_details, df_components, on="Component_ID", how="left")

    # 行级：设备 vendor/type
    df_merged["Device_Clean"] = df_merged["Device"].astype(str).str.strip()
    df_merged["device_vendor"] = df_merged["Device_Clean"].map(device_to_vendor).fillna("")
    df_merged["device_type"] = df_merged["Device_Clean"].map(device_to_type).fillna("")

    # 选择输出列（按你需求最核心的）
    target_columns = [
        # 组件级
        "Component_ID", "Device_Count", "Device_List",
        "vendor_state", "vendor_detail",
        "device_type_state", "device_type_detail",

        # 行级（代表流）
        "Device", "device_vendor", "device_type",
        "Remote_IP", "Remote_Port", "Protocol",
        "Domain_A", "Domain_PTR", "Remote_domain_best",
        "Sequence_Up", "Sequence_Down",
        "Original_Index",
    ]
    available = [c for c in target_columns if c in df_merged.columns]
    df_out = df_merged[available]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV saved: {output_path}")
    print(f"   Rows: {len(df_out)}, Components: {df_out['Component_ID'].nunique()}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    # 可选：限制内存
    try:
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except Exception:
        pass

    if not os.path.exists(INPUT_PKL_PATH):
        print(f"❌ Input file not found: {INPUT_PKL_PATH}")
        return

    print(f"⏳ Loading input: {INPUT_PKL_PATH}")
    flow_data = safe_load_pickle(INPUT_PKL_PATH)
    print(f"📊 Loaded flows: {len(flow_data)}")

    # Step 1：全局组件聚类
    df_details = step_1_global_component_clustering(flow_data, MATRIX_CACHE_FILE)
    gc.collect()

    # Step 3：补充 vendor/type 组件级属性
    df_components, device_to_vendor, device_to_type = step_3_enrich_component_vendor_type(
        df_details, EXTERNAL_DEVICE_LIST, EXTERNAL_DEVICE_TYPE
    )
    gc.collect()

    # Step 4：回灌并导出
    step_4_merge_and_export(df_components, df_details, device_to_vendor, device_to_type, OUTPUT_FINAL_CSV)


if __name__ == "__main__":
    main()
