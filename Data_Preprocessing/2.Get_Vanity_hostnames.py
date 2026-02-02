import pandas as pd
import tldextract
import os
import json
import re
from tqdm import tqdm
from openai import OpenAI

# ================= 配置部分 =================
INPUT_PICKLE = 'Preprocessed/Traffic_Cleaned_v2.pkl'
DEVICE_LIST_CSV = '../../Input/device_list.csv'
EXTERNAL_DNS_CSV = '../../rDNS/all_device_rdns.csv'
OUTPUT_FILE = 'Traffic_Cleaned_v3_DomainA.pkl'


# LLM 配置
PROXY_API_KEY = os.getenv("PROXY_API_KEY") or "xxxxxxxxxxxxxxxxx"  # 请替换为你的 Key
BASE_URL = "xxxxxxxxxxxxxxxxxxxxxxxxxx"  # 或你的 Proxy 地址

client = OpenAI(
    api_key=PROXY_API_KEY,
    base_url=BASE_URL
)


# ================= 辅助函数 =================

def extract_vendor_from_single_domain(domain):
    """(内部函数) 提取单个域名的厂商"""
    if not isinstance(domain, str) or not domain:
        return None
    try:
        extracted = tldextract.extract(domain)
        if extracted.domain:
            return extracted.domain.lower()
    except:
        pass
    parts = domain.split('.')
    if len(parts) >= 2:
        return parts[-2].lower()
    return str(domain).lower()


def get_domain_vendor(domain_str):
    """
    提取域名对应的主厂商 (用于知识库构建)。
    适配处理多域名格式 'dom1.com|dom2.com'。
    由于后续清洗步骤已保证一致性，直接取第一部分解析。
    """
    if domain_str is None or not isinstance(domain_str, str):
        return None
    if '|' in domain_str:
        parts = [p.strip() for p in domain_str.split('|') if p.strip()]
        if parts:
            return extract_vendor_from_single_domain(parts[0])
        return None
    return extract_vendor_from_single_domain(domain_str)


def get_vendor_set_from_multidomain(dns_str):
    """
    从 'a.com|b.com' 中提取唯一的厂商集合 {'vendor_a', 'vendor_b'}
    """
    if not isinstance(dns_str, str) or '|' not in dns_str:
        return None

    parts = [p.strip() for p in dns_str.split('|') if p.strip()]
    vendors = set()
    for p in parts:
        v = extract_vendor_from_single_domain(p)
        if v:
            vendors.add(v)

    if not vendors:
        return None

    return tuple(sorted(list(vendors)))  # 转为 tuple 以便作为字典 key


# ================= LLM 判别函数 =================

def batch_check_vendor_consistency(vendor_groups):
    """
    [NEW] 批量检查一组厂商是否属于同一实体。
    Args:
        vendor_groups: List of tuples, e.g. [('mi', 'xiaomi'), ('google', 'aws')]
    Returns:
        dict: {('mi', 'xiaomi'): True, ('google', 'aws'): False}
    """
    if not vendor_groups:
        return {}

    # 构造输入
    input_data = [{"id": i, "vendors": list(grp)} for i, grp in enumerate(vendor_groups)]

    prompt = f"""
    You are a corporate structure expert. You will receive lists of vendor names extracted from domain names.
    For each list, determine if ALL vendors in the list belong to the SAME parent company, ecosystem, or brand family.

    Rules:
    1. Return 'is_consistent': true ONLY if they are the same entity.
    2. Examples of TRUE: ["xiaomi", "mi"], ["google", "youtube", "doubleclick"], ["amazon", "aws"], ["aliyun", "alibaba"].
    3. Examples of FALSE: ["google", "aws"], ["tuya", "xiaomi"].

    Input: {json.dumps(input_data)}
    Output JSON: {{"results": [{{"id": 0, "is_consistent": true}}, ...]}}
    """

    try:
        response = client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```json|```$", "", content, flags=re.MULTILINE).strip()
        result_json = json.loads(content)

        result_map = {}
        for item in result_json.get("results", []):
            idx = item.get("id")
            if idx is not None and idx < len(vendor_groups):
                result_map[vendor_groups[idx]] = item.get("is_consistent", False)
        return result_map
    except Exception as e:
        print(f"[!] LLM Consistency Check Error: {e}")
        return {}


def batch_evaluate_vendor_matches(vendor_pairs):
    """(原有) Rule 2 使用的 LLM 函数"""
    if not vendor_pairs:
        return {}

    input_data = [{"id": i, "dev": v[0], "dom": v[1]} for i, v in enumerate(vendor_pairs)]

    prompt = f"""
    You are a supply chain expert. Analyze pairs of 'dev' (Device Vendor) and 'dom' (Domain Vendor).
    Determine if 'dom' belongs to the same parent company or ecosystem as 'dev'.
    Rules: Return 'is_match': true ONLY for same entity.
    Input: {json.dumps(input_data)}
    Output JSON: {{"results": [{{"id": 0, "is_match": true}}, ...]}}
    """

    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash-exp",
            messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = re.sub(r"^```json|```$", "", response.choices[0].message.content.strip(), flags=re.MULTILINE).strip()
        result_json = json.loads(content)
        match_map = {}
        for item in result_json.get("results", []):
            match_map[vendor_pairs[item["id"]]] = item.get("is_match", False)
        return match_map
    except Exception:
        return {}


# ================= 主流程 =================

def main():
    # 1. 加载流量数据
    print(f"[-] Loading traffic data from {INPUT_PICKLE}...")
    try:
        raw_data = pd.read_pickle(INPUT_PICKLE)
        df = pd.DataFrame(raw_data) if isinstance(raw_data, list) else raw_data
    except Exception as e:
        print(f"[!] File error: {e}")
        return

    # 2. 合并 Vendor 信息
    print(f"[-] Loading vendors from {DEVICE_LIST_CSV}...")
    try:
        device_df = pd.read_csv(DEVICE_LIST_CSV)
        vendor_map = dict(zip(device_df['Device_Name'], device_df['Vendor']))
        df['Vendor'] = df['Device'].map(vendor_map).fillna('unknown').astype(str).str.lower()
    except Exception as e:
        print(f"[!] CSV Error: {e}")
        return

    df['Domain_A'] = df['Domain'].replace({'None': None, 'nan': None})

    # ---------------------------------------------------------
    # 3. [UPDATED] 加载并使用 LLM 清洗 External DNS CSV
    # ---------------------------------------------------------
    print(f"[-] Integrating external DNS data from {EXTERNAL_DNS_CSV}...")
    try:
        ext_df = pd.read_csv(EXTERNAL_DNS_CSV)
        valid_ext = ext_df[ext_df['query_method'] == 'dns_query'].copy()

        # 3.1 识别包含多域名的行
        # 生成 Vendor Set tuple: ('mi', 'xiaomi')
        valid_ext['Vendor_Set'] = valid_ext['DNS'].apply(lambda x: get_vendor_set_from_multidomain(str(x)))

        # 3.2 收集需要 LLM 判断的唯一组合
        # 筛选条件: Vendor_Set 不为空，且包含多个不同的厂商 (len > 1)
        multi_vendor_rows = valid_ext[valid_ext['Vendor_Set'].notna()]
        candidates = set()
        for v_set in multi_vendor_rows['Vendor_Set']:
            if len(v_set) > 1:
                candidates.add(v_set)

        candidate_list = list(candidates)
        print(f"    > Found {len(candidate_list)} unique multi-vendor groups requiring LLM consistency check.")

        # 3.3 批量 LLM 判别一致性
        consistency_map = {}  # {('mi', 'xiaomi'): True}
        CHUNK_SIZE = 50
        if candidate_list:
            for i in tqdm(range(0, len(candidate_list), CHUNK_SIZE), desc="LLM Consistency Check"):
                chunk = candidate_list[i:i + CHUNK_SIZE]
                res = batch_check_vendor_consistency(chunk)
                consistency_map.update(res)

        # 3.4 定义最终应用函数
        def clean_dns_with_llm(row):
            dns_str = str(row['DNS'])
            v_set = row['Vendor_Set']

            # 单域名或无域名，直接返回原值(strip后)
            if v_set is None:
                # 检查是否为空
                if not dns_str or dns_str.lower() == 'nan': return None
                return dns_str.strip()

            # 多域名情况
            if len(v_set) == 1:
                # 只有一种厂商 (例如 'a.mi.com|b.mi.com') -> 自动保留
                return dns_str.strip()

            # 多种厂商 -> 查表
            is_consistent = consistency_map.get(v_set, False)
            if is_consistent:
                # LLM 说是同一家 -> 保留
                return dns_str.strip()
            else:
                # 冲突 -> 丢弃
                return None

        # 3.5 应用清洗
        tqdm.pandas(desc="Applying LLM Filter")
        valid_ext['Cleaned_DNS'] = valid_ext.progress_apply(clean_dns_with_llm, axis=1)

        # 过滤无效行
        valid_ext = valid_ext[valid_ext['Cleaned_DNS'].notna()]

        # 3.6 填补 Domain_A
        valid_ext = valid_ext[['Folder', 'IP', 'Cleaned_DNS']].drop_duplicates(subset=['Folder', 'IP'])
        ext_map = dict(zip(zip(valid_ext['Folder'], valid_ext['IP']), valid_ext['Cleaned_DNS']))

        print(f"    > Found {len(ext_map)} valid external records after LLM filtering.")

        missing_mask = df['Domain_A'].isna()
        temp_keys = list(zip(df.loc[missing_mask, 'Device'], df.loc[missing_mask, 'Remote_IP']))
        new_values = [ext_map.get(k) for k in temp_keys]
        updates = {idx: val for idx, val in zip(df[missing_mask].index, new_values) if val is not None}

        if updates:
            df.loc[updates.keys(), 'Domain_A'] = list(updates.values())
            print(f"    > Filled {len(updates)} missing domains.")

    except Exception as e:
        print(f"[!] Warning processing external CSV: {e}")

    # ---------------------------------------------------------
    # 4. 构建 IP 知识库
    # ---------------------------------------------------------
    print("[-] Building IP Knowledge Base...")
    valid_dns_df = df[df['Domain_A'].notna()].copy()

    # 提取厂商 (get_domain_vendor 已适配多域名，直接取第一个)
    valid_dns_df['Domain_Vendor'] = valid_dns_df['Domain_A'].apply(get_domain_vendor)

    ip_knowledge = {}
    unique_records = valid_dns_df[['Remote_IP', 'Domain_A', 'Vendor', 'Domain_Vendor']].drop_duplicates()

    for _, row in unique_records.iterrows():
        ip = row['Remote_IP']
        if ip not in ip_knowledge: ip_knowledge[ip] = []
        ip_knowledge[ip].append({
            'domain': row['Domain_A'],
            's_vendor': row['Vendor'],
            'd_vendor': row['Domain_Vendor']
        })

    # ---------------------------------------------------------
    # 5. 准备 Rule 2 LLM
    # ---------------------------------------------------------
    print("[-] Preparing LLM for Rule 2...")
    mask_missing = df['Domain_A'].isna()
    target_indices = df[mask_missing].index

    candidate_pairs = set()
    for idx in target_indices:
        ip = df.at[idx, 'Remote_IP']
        my_vendor = df.at[idx, 'Vendor']
        if ip in ip_knowledge:
            for rec in ip_knowledge[ip]:
                d_vendor = str(rec['d_vendor']).lower()
                if d_vendor and d_vendor not in ['none', 'nan'] and my_vendor != d_vendor:
                    candidate_pairs.add((my_vendor, d_vendor))

    candidate_list = list(candidate_pairs)
    print(f"    > Found {len(candidate_list)} pairs for Rule 2 check.")

    llm_matches = {}
    if candidate_list:
        for i in tqdm(range(0, len(candidate_list), CHUNK_SIZE), desc="Rule 2 LLM"):
            chunk = candidate_list[i:i + CHUNK_SIZE]
            llm_matches.update(batch_evaluate_vendor_matches(chunk))

    # ---------------------------------------------------------
    # 6. 执行推断
    # ---------------------------------------------------------
    print("[-] Inferring Domain_A...")
    inferred_domains = {}
    c1, c2 = 0, 0

    for idx in tqdm(target_indices, desc="Processing"):
        ip = df.at[idx, 'Remote_IP']
        my_vendor = df.at[idx, 'Vendor']

        if ip not in ip_knowledge: continue

        candidates = ip_knowledge[ip]
        found_domain = None

        # Rule 1
        for rec in candidates:
            if rec['s_vendor'] == my_vendor:
                found_domain = rec['domain']
                c1 += 1
                break

        # Rule 2
        if not found_domain:
            for rec in candidates:
                d_vendor = str(rec['d_vendor']).lower()
                is_match = False
                if d_vendor == my_vendor:
                    is_match = True
                elif (my_vendor, d_vendor) in llm_matches:
                    is_match = llm_matches[(my_vendor, d_vendor)]

                if is_match:
                    found_domain = rec['domain']
                    c2 += 1
                    break

        if found_domain: inferred_domains[idx] = found_domain

    if inferred_domains:
        df.loc[inferred_domains.keys(), 'Domain_A'] = list(inferred_domains.values())

    print(f"Total Inferred: {len(inferred_domains)} (R1: {c1}, R2: {c2})")

    print(f"[-] Saving to {OUTPUT_FILE}...")
    df.to_pickle(OUTPUT_FILE)
    print("[-] Done.")


if __name__ == "__main__":

    main()

