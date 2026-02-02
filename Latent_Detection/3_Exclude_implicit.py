import pandas as pd
import pickle
import os


def filter_implicit_and_multivendor(csv_file, pkl_file, output_file):
    print(f"[-] 正在读取流量数据: {csv_file}")
    if not os.path.exists(csv_file) or not os.path.exists(pkl_file):
        print("[!] 错误: 输入文件不存在。")
        return

    # 1. 加载数据
    df = pd.read_csv(csv_file)
    original_count = df['Component_ID'].nunique()
    print(f"    > 原始组件数量: {original_count}")

    print(f"[-] 正在加载隐性关系库: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        implicit_data = pickle.load(f)

    # 2. 提取合法的隐性目标集合 (Whitelist)
    # Implicit_Relationships.pkl 结构: {'IP': {target: {devs}}, 'Service': {target: {devs}}, ...}
    valid_ips = set(implicit_data.get('IP', {}).keys())
    valid_domains = set(implicit_data.get('Service', {}).keys())
    # Infrastructure 层通常也对应域名或组织名，这里主要匹配 Remote_domain
    valid_domains.update(implicit_data.get('Infrastructure', {}).keys())

    print(f"    > 加载了 {len(valid_ips)} 个隐性 IP 目标")
    print(f"    > 加载了 {len(valid_domains)} 个隐性域名目标")

    # 3. 过滤条件 A: 目标匹配 (Implicit Check)
    # 保留那些 Remote_IP 在 valid_ips 中，或者 Remote_domain 在 valid_domains 中的行
    # 注意：Remote_domain 可能为空，需要处理

    # 转换为字符串处理，防止 NaN 报错
    df['Remote_domain'] = df['Remote_domain'].fillna('').astype(str)
    df['Remote_IP'] = df['Remote_IP'].fillna('').astype(str)

    # 标记哪些行匹配到了隐性目标
    mask_ip = df['Remote_IP'].isin(valid_ips)
    mask_domain = df['Remote_domain'].isin(valid_domains)

    # 获取匹配隐性目标的 Component_ID 列表
    # 逻辑：只要组件中有一条流量匹配到了隐性目标，该组件在这一步就被认为是候选
    # (更严格的逻辑是只保留匹配的那几行，通常 Component_ID 定义了同一种行为，所以按 ID 筛选)
    implicit_component_ids = df[mask_ip | mask_domain]['Component_ID'].unique()

    print(f"    > 满足隐性目标关系的组件数: {len(implicit_component_ids)}")

    # 4. 过滤条件 B: 跨厂商检查 (Multi-Vendor Check)
    # 我们只关注通过了条件 A 的组件
    df_candidates = df[df['Component_ID'].isin(implicit_component_ids)].copy()

    # 统计每个组件包含的唯一厂商数量
    # 排除 'unknown', 'nan', 空值对统计的干扰 (可视情况调整)
    df_candidates['device_vendor'] = df_candidates['device_vendor'].fillna('unknown').astype(str)

    # 分组计算厂商数量
    vendor_counts = df_candidates.groupby('Component_ID')['device_vendor'].nunique()

    # 筛选厂商数量 > 1 的组件 ID
    multi_vendor_ids = vendor_counts[vendor_counts > 1].index

    print(f"    > 进一步满足跨厂商 (>1 Vendor) 的组件数: {len(multi_vendor_ids)}")

    # 5. 生成最终结果
    # 只保留同时满足条件的组件的所有行
    df_final = df[df['Component_ID'].isin(multi_vendor_ids)]

    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)

    print("-" * 50)
    print(f"筛选完成！")
    print(f"输入组件数: {original_count}")
    print(f"输出组件数: {df_final['Component_ID'].nunique()}")
    print(f"丢弃组件数: {original_count - df_final['Component_ID'].nunique()}")
    print(f"结果已保存至: {output_file}")
    print("-" * 50)

    # 预览
    if not df_final.empty:
        print("预览前 3 行 (验证厂商混合情况):")
        sample_id = df_final['Component_ID'].iloc[0]
        print(f"Component: {sample_id}")
        print(df_final[df_final['Component_ID'] == sample_id][['Device', 'device_vendor', 'Remote_domain']].head(5))


# --- 执行配置 ---
if __name__ == "__main__":
    # base_dir = '../DTW_1_12'

    # 输入 1: 之前生成的带有厂商信息的宽表
    input_csv = f'../../DTW_1_12/2.Extended_Network_Flows_vendor_type_raw_v1.12.csv'

    # 输入 2: 之前生成的隐性关系字典
    input_pkl = '../2_Implicit_Identification/Implicit_Relationships.pkl'  # 假设在当前目录，如果在 base_dir 请修改

    # 输出: 过滤后的中间文件，供下一步使用
    output_csv = f'./3.Filtered_Implicit_MultiVendor_Flows_v1.12.csv'


    filter_implicit_and_multivendor(input_csv, input_pkl, output_csv)
