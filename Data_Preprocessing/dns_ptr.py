# import pandas as pd
# import socket
# import concurrent.futures
# from tqdm import tqdm
#
# # ================= 配置部分 =================
# # 输入文件：应该是上一轮 Tier-1 代码生成的包含 Domain_A 的文件
# INPUT_FILE = 'Traffic_Cleaned_v3_DomainA.pkl'
# OUTPUT_FILE = 'Traffic_Cleaned_v3_Final.pkl'
#
# # rDNS 配置
# MAX_WORKERS = 100  # 并发线程数 (建议 50-200，取决于网络环境)
# TIMEOUT = 5  # 单次查询超时时间 (秒)
#
#
# # ================= 辅助函数 =================
#
# def lookup_ptr(ip):
#     """
#     执行单次反向 DNS (PTR) 查询。
#     返回: 域名字符串 (如 'ec2-54-x.compute.amazonaws.com') 或 None
#     """
#     try:
#         # socket.gethostbyaddr 返回 (hostname, aliaslist, ipaddrlist)
#         name, _, _ = socket.gethostbyaddr(ip)
#         return name
#     except (socket.herror, socket.gaierror, socket.timeout):
#         # socket.herror: Unknown host
#         # socket.gaierror: Address-related error
#         return None
#     except Exception:
#         return None
#
#
# # ================= 主流程 =================
#
# def main():
#     # 1. 加载数据 (来自 Tier-1 的输出)
#     print(f"[-] Loading Tier-1 data from {INPUT_FILE}...")
#     try:
#         raw_data = pd.read_pickle(INPUT_FILE)
#
#         # 确保转换为 DataFrame
#         if isinstance(raw_data, list):
#             df = pd.DataFrame(raw_data)
#         elif isinstance(raw_data, pd.DataFrame):
#             df = raw_data
#         else:
#             print("[!] Unknown data format.")
#             return
#
#         print(f"    > Data Shape: {df.shape}")
#
#         # 检查是否包含必要的列
#         if 'Remote_IP' not in df.columns:
#             print("[!] Error: 'Remote_IP' column missing.")
#             return
#
#     except FileNotFoundError:
#         print(f"[!] Error: File {INPUT_FILE} not found. Please run Tier-1 code first.")
#         return
#
#     # 2. 准备 IP 列表
#     # 我们只查询唯一的 IP，避免重复劳动
#     unique_ips = df['Remote_IP'].unique()
#     total_ips = len(unique_ips)
#     print(f"[-] Extracting Infrastructure Layer (Tier-2)...")
#     print(f"    > Found {total_ips} unique IPs to resolve.")
#
#     # 3. 执行并发 rDNS 查询
#     print(f"    > Starting active rDNS lookups (Workers={MAX_WORKERS})...")
#
#     # 设置全局 Socket 超时
#     socket.setdefaulttimeout(TIMEOUT)
#
#     ip_ptr_map = {}  # 存储结果: IP -> PTR Domain
#
#     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         # 提交所有任务
#         future_to_ip = {executor.submit(lookup_ptr, ip): ip for ip in unique_ips}
#
#         # 使用 tqdm 显示进度
#         for future in tqdm(concurrent.futures.as_completed(future_to_ip),
#                            total=total_ips,
#                            desc="Resolving PTR"):
#             ip = future_to_ip[future]
#             try:
#                 result = future.result()
#                 ip_ptr_map[ip] = result
#             except Exception:
#                 ip_ptr_map[ip] = None
#
#     # 4. 结果映射回 DataFrame
#     print("[-] Mapping infrastructure labels to flows...")
#     # 使用 map 函数快速填充新列 'Domain_PTR'
#     df['Domain_PTR'] = df['Remote_IP'].map(ip_ptr_map)
#
#     # 5. 统计与保存
#     resolved_count = df['Domain_PTR'].notna().sum()
#     coverage = (resolved_count / len(df)) * 100
#
#     print("-" * 40)
#     print(f"Tier-2 Extraction Complete.")
#     print(f"Total Flows: {len(df)}")
#     print(f"Flows with Infrastructure Label (PTR): {resolved_count} ({coverage:.2f}%)")
#     print("-" * 40)
#
#     print(f"[-] Saving final dataset to {OUTPUT_FILE}...")
#     df.to_pickle(OUTPUT_FILE)
#
#     # 6. 数据预览
#     print("\n[Final Data Preview]")
#     # 选取几行展示双层映射效果
#     # 优先展示 Domain_A 和 Domain_PTR 都有值的行
#     dual_layer_mask = df['Domain_A'].notna() & df['Domain_PTR'].notna()
#
#     cols = ['Device', 'Remote_IP', 'Domain_A', 'Domain_PTR']
#     # 如果截图中的列名是 'Device_ID' 请自行调整，这里沿用截图逻辑 'Device'
#     if 'Device' not in df.columns and 'Device_ID' in df.columns:
#         cols[0] = 'Device_ID'
#
#     if dual_layer_mask.any():
#         print("--- Rows with Dual-Layer Labels (Business + Infrastructure) ---")
#         print(df.loc[dual_layer_mask, cols].head(5).to_string())
#     else:
#         print(df[cols].head(5).to_string())
#
#     print("\n[-] Done.")
#
#
# if __name__ == '__main__':
#     main()

import pandas as pd
import socket
import concurrent.futures
from tqdm import tqdm

# ================= 配置部分 =================
# 输入文件：上一轮 Tier-1 代码生成的包含 Domain_A 的文件
INPUT_FILE = 'Traffic_Cleaned_v3_DomainA.pkl'
EXTERNAL_DNS_CSV = '../../rDNS/all_device_rdns.csv'  # 外部数据源
OUTPUT_FILE = 'Traffic_Cleaned_v3_Final.pkl'

# Active rDNS 配置
MAX_WORKERS = 100  # 并发线程数
TIMEOUT = 3  # 单次查询超时(秒)


# ================= 辅助函数 =================

def lookup_ptr(ip):
    """
    执行单次主动 rDNS 查询
    """
    try:
        name, _, _ = socket.gethostbyaddr(ip)
        return name
    except (socket.herror, socket.gaierror, socket.timeout):
        return None
    except Exception:
        return None


def clean_rapiddns_record(dns_str):
    """
    清洗 RapidDNS 记录:
    如果存在多值 'ec2-a.com|ec2-b.com'，取第一个非空值。
    """
    if not isinstance(dns_str, str):
        return None

    parts = dns_str.split('|')
    for p in parts:
        clean_p = p.strip()
        if clean_p:
            return clean_p
    return None


# ================= 主流程 =================

def main():
    # ---------------------------------------------------------
    # 1. 加载数据 (来自 Tier-1 的输出)
    # ---------------------------------------------------------
    print(f"[-] Loading Tier-1 data from {INPUT_FILE}...")
    try:
        raw_data = pd.read_pickle(INPUT_FILE)
        if isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data
        else:
            print("[!] Unknown data format.")
            return

        # 确保 Domain_PTR 列存在，初始化为 None
        df['Domain_PTR'] = None
        print(f"    > Data Shape: {df.shape}")

    except FileNotFoundError:
        print(f"[!] Error: File {INPUT_FILE} not found.")
        return

    unique_ips = df['Remote_IP'].unique()
    print(f"[-] Processing Tier-2 for {len(unique_ips)} unique IPs...")

    # ---------------------------------------------------------
    # 2. 执行 Active rDNS 查询 (Primary Source)
    # ---------------------------------------------------------
    print(f"[-] Starting Active rDNS queries (Threads={MAX_WORKERS})...")

    socket.setdefaulttimeout(TIMEOUT)
    active_ptr_map = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ip = {executor.submit(lookup_ptr, ip): ip for ip in unique_ips}

        for future in tqdm(concurrent.futures.as_completed(future_to_ip),
                           total=len(unique_ips),
                           desc="Active Query"):
            ip = future_to_ip[future]
            active_ptr_map[ip] = future.result()

    # 将主动查询结果填入 DataFrame
    # 注意：这里只填充非 None 的结果
    df['Domain_PTR'] = df['Remote_IP'].map(active_ptr_map)

    count_active = df['Domain_PTR'].notna().sum()
    print(f"    > Active rDNS found: {count_active} records.")

    # ---------------------------------------------------------
    # 3. 加载 External RapidDNS 数据 (Fallback Source)
    # ---------------------------------------------------------
    print(f"[-] Loading External RapidDNS data from {EXTERNAL_DNS_CSV}...")
    try:
        ext_df = pd.read_csv(EXTERNAL_DNS_CSV)

        # 3.1 筛选 query_method == 'RapidDNS'
        rapid_df = ext_df[ext_df['query_method'] == 'RapidDNS'].copy()

        if not rapid_df.empty:
            # 3.2 清洗 DNS 列 (处理 | 分割)
            tqdm.pandas(desc="Cleaning RapidDNS")
            rapid_df['Cleaned_PTR'] = rapid_df['DNS'].progress_apply(clean_rapiddns_record)

            # 3.3 去重并建立字典 IP -> Cleaned_PTR
            # dropna: 去掉清洗后为空的
            # drop_duplicates: 确保一个 IP 只有一个 PTR (保留第一个出现的)
            rapid_df = rapid_df.dropna(subset=['Cleaned_PTR'])
            rapid_df = rapid_df.drop_duplicates(subset=['IP'])

            rapid_map = dict(zip(rapid_df['IP'], rapid_df['Cleaned_PTR']))
            print(f"    > Loaded {len(rapid_map)} valid RapidDNS records.")

            # 3.4 填补 Active rDNS 失败的空缺
            missing_mask = df['Domain_PTR'].isna()
            missing_indices = df[missing_mask].index

            # 查找
            # 创建临时 list 进行 map
            current_missing_ips = df.loc[missing_mask, 'Remote_IP']
            filled_values = current_missing_ips.map(rapid_map)

            # 仅更新非空值
            # 找出 map 后非 NaN 的索引
            fillable_mask = filled_values.notna()
            if fillable_mask.any():
                # 注意：fillable_mask 是相对于 missing_mask 的子集
                # 我们需要通过索引操作
                indices_to_update = filled_values[fillable_mask].index
                df.loc[indices_to_update, 'Domain_PTR'] = filled_values[fillable_mask]

                print(f"    > Successfully filled {len(indices_to_update)} missing PTRs using RapidDNS.")
            else:
                print("    > No additional PTRs found in RapidDNS for missing IPs.")
        else:
            print("    > No 'RapidDNS' records found in CSV.")

    except Exception as e:
        print(f"[!] Warning: Could not process external CSV: {e}")

    # ---------------------------------------------------------
    # 4. 统计与保存
    # ---------------------------------------------------------
    final_count = df['Domain_PTR'].notna().sum()
    coverage = (final_count / len(df)) * 100

    print("-" * 40)
    print(f"Tier-2 Extraction Summary:")
    print(f"  Total Flows:        {len(df)}")
    print(f"  Active rDNS Found:  {count_active}")
    print(f"  Total with PTR:     {final_count} ({coverage:.2f}%)")
    print("-" * 40)

    print(f"[-] Saving to {OUTPUT_FILE}...")
    df.to_pickle(OUTPUT_FILE)

    # 5. 预览
    print("\n[Preview: Rows with External Fill]")
    # 尝试展示被 RapidDNS 填充的行 (PTR不为空 但 ActiveMap中为空或为None)
    # 为了演示简单，直接展示 Domain_PTR 非空的样本
    sample_cols = ['Remote_IP', 'Domain_A', 'Domain_PTR']
    if 'Device' in df.columns: sample_cols.insert(0, 'Device')

    if final_count > 0:
        print(df[df['Domain_PTR'].notna()][sample_cols].sample(5).to_string())

    print("\n[-] Done.")


if __name__ == '__main__':
    main()