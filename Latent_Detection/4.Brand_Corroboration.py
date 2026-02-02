import pandas as pd
import os
import json
import itertools
from openai import OpenAI
from tqdm import tqdm
import time

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# --- Input/Output Files ---
INPUT_CSV = '3.2.Clusters_Inconsistent_NoImplicit.csv'
OUTPUT_VALIDATED_CSV = '4.1.Clusters_Validated_by_LLM.csv'
OUTPUT_REMAINING_CSV = '4.2.Clusters_Final_Latent.csv'

# --- LLM API Configuration ---
# Uses environment variable PROXY_API_KEY, with a fallback for local testing
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "sk-8VI89370effc22b5f385033ad7d3930a4c227b7ac06kV9ug")
BASE_URL = "https://api.gptsapi.net/v1"
LLM_MODEL = "gemini-3-pro-preview"  # Using a standard Gemini model name

# --- Caching ---
# File to store LLM responses to avoid re-querying the same vendor pairs
CACHE_FILE = 'llm_vendor_cache.json'

# ==============================================================================
# 2. LLM INTERACTION & CACHING
# ==============================================================================

# Initialize the OpenAI client to point to the proxy
if not PROXY_API_KEY or "your_fallback_key_here" in PROXY_API_KEY:
    print("[!] WARNING: PROXY_API_KEY is not set or is using a placeholder. LLM calls will fail.")
    # You can raise an error here if you want to enforce the key is set
    # raise ValueError("PROXY_API_KEY is not set.")

client = OpenAI(
    api_key=PROXY_API_KEY,
    base_url=BASE_URL
)


def load_cache():
    """Loads the LLM response cache from a file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Saves the LLM response cache to a file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def ask_llm_about_vendors(vendor1, vendor2, cache):
    """
    Queries the LLM about the relationship between two vendors, using a cache.
    Returns a dictionary: {'is_related': bool, 'reason': str}
    """
    # Create a canonical key for the cache (order-independent)
    cache_key = tuple(sorted((vendor1.lower(), vendor2.lower())))

    if str(cache_key) in cache:
        return cache[str(cache_key)]

    prompt = f"""
    As a senior IoT supply chain analyst, determine if there is a known business relationship between "{vendor1}" and "{vendor2}".
    Focus ONLY on relationships relevant to IoT device manufacturing and software platforms, such as:
    1.  One is an OEM/ODM for the other.
    2.  They are part of the same technology ecosystem (e.g., Xiaomi Ecosystem, Tuya Smart, Amazon Alexa, Google Home).
    3.  One is a parent company or subsidiary of the other.
    4.  They have a formal strategic partnership for device integration.

    Do NOT consider supplier relationships for generic components (e.g., Samsung providing screens to Apple).

    Respond with a single, valid JSON object containing two keys:
    - "is_related": boolean (true or false)
    - "reason": a brief, one-sentence explanation.

    Example for "Xiaomi" and "Roborock":
    {{
      "is_related": true,
      "reason": "Roborock is a key member of the Xiaomi ecological chain, manufacturing smart vacuum cleaners for the Mi Home platform."
    }}

    Example for "Apple" and "Google":
    {{
      "is_related": false,
      "reason": "They are major competitors in the smart home market, despite some cross-platform app support."
    }}
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds only in valid JSON format."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0
        )

        response_text = response.choices[0].message.content.strip()

        # Robust JSON parsing
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        result = json.loads(response_text)

        # Validate result format before caching
        if 'is_related' in result and 'reason' in result:
            cache[str(cache_key)] = result
            return result
        else:
            raise ValueError("LLM response missing required keys.")

    except Exception as e:
        print(f"[!] LLM Error for ({vendor1}, {vendor2}): {e}")
        # Return a default 'false' response on error
        return {"is_related": False, "reason": f"API_ERROR: {str(e)}"}


# ==============================================================================
# 3. MAIN LOGIC
# ==============================================================================

def main():
    print("[-] Starting Vendor Ecosystem Corroboration...")

    # 1. Load data
    if not os.path.exists(INPUT_CSV):
        print(f"[!] Error: Input file '{INPUT_CSV}' not found.")
        return
    df_flows = pd.read_csv(INPUT_CSV)

    # 2. Load LLM cache
    llm_cache = load_cache()

    # 3. Group clusters and prepare for processing
    # Assuming 'vendor_name' is the correct column. If not, change it.
    if 'vendor_name' not in df_flows.columns:
        print("[!] Error: 'vendor_name' column not found in CSV.")
        return

    clusters = df_flows.groupby('Component_ID')['device_vendor'].unique().apply(list).to_dict()
    print(f"[-] Found {len(clusters)} clusters to analyze.")

    validation_results = []

    try:
        # 4. Iterate through each cluster and query LLM
        for comp_id, vendors in tqdm(clusters.items(), desc="Analyzing Clusters"):

            unique_vendors = sorted(list(set(vendors)))

            is_validated = False
            validation_evidence = []  # To store reasons

            # Only check if there are 2 or more unique vendors
            if len(unique_vendors) >= 2:
                # Create all unique pairs of vendors to check
                vendor_pairs = list(itertools.combinations(unique_vendors, 2))

                for v1, v2 in vendor_pairs:
                    # Introduce a small delay to respect API rate limits
                    time.sleep(0.5)

                    llm_result = ask_llm_about_vendors(v1, v2, llm_cache)

                    if llm_result.get('is_related'):
                        is_validated = True
                        evidence = f"Relationship found between '{v1}' and '{v2}': {llm_result.get('reason')}"
                        validation_evidence.append(evidence)
                        # Once one valid relationship is found, we can stop checking this cluster
                        break

            validation_results.append({
                'Component_ID': comp_id,
                'LLM_Validated': is_validated,
                'LLM_Validation_Evidence': "; ".join(validation_evidence) if validation_evidence else None
            })

    finally:
        # 5. Save cache regardless of success or failure
        print("\n[-] Saving LLM cache...")
        save_cache(llm_cache)

    # 6. Merge results and split data
    df_validation = pd.DataFrame(validation_results)
    df_final = df_flows.merge(df_validation, on='Component_ID', how='left')

    df_validated = df_final[df_final['LLM_Validated'] == True].copy()
    df_remaining = df_final[df_final['LLM_Validated'] == False].copy()

    # Clean up columns for the final latent set
    df_remaining.drop(columns=['LLM_Validated', 'LLM_Validation_Evidence'], inplace=True)

    # 7. Save output files
    print(f"[-] Saving {len(df_validated)} validated rows to {OUTPUT_VALIDATED_CSV}...")
    df_validated.to_csv(OUTPUT_VALIDATED_CSV, index=False)

    print(f"[-] Saving {len(df_remaining)} remaining latent rows to {OUTPUT_REMAINING_CSV}...")
    df_remaining.to_csv(OUTPUT_REMAINING_CSV, index=False)

    # 8. Print summary
    print("\n" + "=" * 40)
    print("LLM VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Total Clusters Analyzed: {len(clusters)}")
    print(f"Clusters Validated by LLM: {len(df_validated['Component_ID'].unique())}")
    print(f"Remaining Latent Clusters: {len(df_remaining['Component_ID'].unique())}")
    print("=" * 40)


if __name__ == '__main__':
    main()

# import pandas as pd
# import os
# import json
# import itertools
# from openai import OpenAI, RateLimitError, APIStatusError
# from tqdm import tqdm
# import time
#
# # ==============================================================================
# # 1. CONFIGURATION (与之前脚本保持一致)
# # ==============================================================================
#
# # --- Input/Output Files ---
# INPUT_CSV = '4.4.Clusters_Inconsistent_NoImplicit.csv'
# OUTPUT_VALIDATED_CSV = '5.1.Clusters_Validated_by_LLM.csv'
# OUTPUT_REMAINING_CSV = '5.1.Clusters_Final_Latent.csv'
#
# # --- LLM API Configuration ---
# PROXY_API_KEY = os.getenv("PROXY_API_KEY", "sk-8VI89370effc22b5f385033ad7d3930a4c227b7ac06kV9ug")
# BASE_URL = "https://api.gptsapi.net/v1"
# LLM_MODEL = "gemini-2.5-pro"
#
# # --- Caching & Resumption ---
# CACHE_FILE = 'llm_vendor_cache.json'
#
# # --- [NEW] Resiliency Configuration ---
# API_RETRY_ATTEMPTS = 3  # 每次 API 调用失败后的重试次数
# API_RETRY_DELAY_SECONDS = 60  # 遇到速率/额度问题时的等待时间（秒）
# API_NORMAL_DELAY = 1  # 正常调用之间的延迟，防止触发速率限制
#
# # ==============================================================================
# # 2. LLM INTERACTION (带缓存和重试功能)
# # ==============================================================================
#
# # Initialize client
# if not PROXY_API_KEY or "your_fallback_key_here" in PROXY_API_KEY:
#     print("[!] WARNING: PROXY_API_KEY is not set. LLM calls will fail.")
# client = OpenAI(api_key=PROXY_API_KEY, base_url=BASE_URL)
#
#
# def load_cache():
#     if os.path.exists(CACHE_FILE):
#         with open(CACHE_FILE, 'r') as f:
#             return json.load(f)
#     return {}
#
#
# def save_cache(cache):
#     with open(CACHE_FILE, 'w') as f:
#         json.dump(cache, f, indent=2)
#
#
# def ask_llm_about_vendors_resumable(vendor1, vendor2, cache):
#     """
#     Queries the LLM, now with caching and robust retry logic.
#     """
#     cache_key = str(tuple(sorted((vendor1.lower(), vendor2.lower()))))
#
#     if cache_key in cache:
#         # 如果缓存命中，直接返回，不打印，让进度条快速前进
#         return cache[cache_key]
#
#     # --- 如果缓存未命中，执行 API 调用 ---
#     print(f"\n[API CALL] Querying for ({vendor1}, {vendor2})...")
#
#     prompt = f"""
#     As a senior IoT supply chain analyst, determine if there is a known business relationship between "{vendor1}" and "{vendor2}".
#     Focus ONLY on relationships relevant to IoT device manufacturing and software platforms, such as:
#     1.  One is an OEM/ODM for the other.
#     2.  They are part of the same technology ecosystem (e.g., Xiaomi Ecosystem, Tuya Smart, Amazon Alexa, Google Home).
#     3.  One is a parent company or subsidiary of the other.
#     4.  They have a formal strategic partnership for device integration.
#     Do NOT consider supplier relationships for generic components.
#     Respond with a single, valid JSON object with keys "is_related" (boolean) and "reason" (string).
#     """
#
#     for attempt in range(API_RETRY_ATTEMPTS):
#         try:
#             # 正常延迟
#             time.sleep(API_NORMAL_DELAY)
#
#             response = client.chat.completions.create(
#                 model=LLM_MODEL,
#                 messages=[
#                     {"role": "system",
#                      "content": "You are a helpful assistant that responds only in valid JSON format."},
#                     {"role": "user", "content": prompt},
#                 ],
#                 temperature=0.0
#             )
#             response_text = response.choices[0].message.content.strip()
#             if response_text.startswith("```json"): response_text = response_text[7:]
#             if response_text.endswith("```"): response_text = response_text[:-3]
#
#             result = json.loads(response_text)
#
#             if 'is_related' in result and 'reason' in result:
#                 cache[cache_key] = result  # 成功后存入缓存
#                 return result
#             else:
#                 raise ValueError("LLM response missing required keys.")
#
#         except (RateLimitError, APIStatusError) as e:
#             # 捕获 429 (速率) 和 403 (额度) 错误
#             print(
#                 f"  -> API Error ({e.status_code}) for ({vendor1}, {vendor2}). Retrying in {API_RETRY_DELAY_SECONDS}s... (Attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")
#             time.sleep(API_RETRY_DELAY_SECONDS)
#             continue  # 继续下一次重试
#
#         except Exception as e:
#             print(f"  -> General Error for ({vendor1}, {vendor2}): {e}. Aborting this pair.")
#             # 记录错误但不重试
#             error_result = {"is_related": False, "reason": f"GENERAL_ERROR: {str(e)}"}
#             cache[cache_key] = error_result
#             return error_result
#
#     # 如果所有重试都失败了
#     print(f"  -> All retries failed for ({vendor1}, {vendor2}). Marking as unrelated.")
#     fail_result = {"is_related": False, "reason": "All API retries failed."}
#     cache[cache_key] = fail_result
#     return fail_result
#
#
# # ==============================================================================
# # 3. MAIN LOGIC (几乎不变，但调用了新的函数)
# # ==============================================================================
# def main():
#     print("[-] Starting Vendor Ecosystem Corroboration (Resumption Mode)...")
#
#     if not os.path.exists(INPUT_CSV):
#         print(f"[!] Error: Input file '{INPUT_CSV}' not found.")
#         return
#     df_flows = pd.read_csv(INPUT_CSV)
#
#     llm_cache = load_cache()
#     print(f"    > Loaded {len(llm_cache)} vendor pairs from cache.")
#
#     if 'vendor_name' not in df_flows.columns:
#         print("[!] Error: 'vendor_name' column not found.")
#         return
#
#     clusters = df_flows.groupby('Component_ID')['vendor_name'].unique().apply(list).to_dict()
#     print(f"[-] Analyzing {len(clusters)} total clusters.")
#
#     validation_results = []
#
#     try:
#         with tqdm(clusters.items(), desc="Analyzing Clusters") as pbar:
#             for comp_id, vendors in pbar:
#                 unique_vendors = sorted(list(set(vendors)))
#                 is_validated = False
#                 validation_evidence = []
#
#                 if len(unique_vendors) >= 2:
#                     vendor_pairs = list(itertools.combinations(unique_vendors, 2))
#
#                     for v1, v2 in vendor_pairs:
#                         # 使用新的、可恢复的函数
#                         llm_result = ask_llm_about_vendors_resumable(v1, v2, llm_cache)
#
#                         if llm_result.get('is_related'):
#                             is_validated = True
#                             evidence = f"Relationship between '{v1}' and '{v2}': {llm_result.get('reason')}"
#                             validation_evidence.append(evidence)
#                             break  # 找到一个关系就足以验证该 cluster
#
#                 validation_results.append({
#                     'Component_ID': comp_id,
#                     'LLM_Validated': is_validated,
#                     'LLM_Validation_Evidence': "; ".join(validation_evidence) if validation_evidence else None
#                 })
#     finally:
#         print("\n[-] Saving LLM cache to ensure progress is kept...")
#         save_cache(llm_cache)
#
#     # --- 后续处理与之前完全相同 ---
#     df_validation = pd.DataFrame(validation_results)
#     df_final = df_flows.merge(df_validation, on='Component_ID', how='left')
#
#     df_validated = df_final[df_final['LLM_Validated'] == True].copy()
#     df_remaining = df_final[df_final['LLM_Validated'] == False].copy()
#
#     df_remaining.drop(columns=['LLM_Validated', 'LLM_Validation_Evidence'], inplace=True)
#
#     print(f"[-] Saving {len(df_validated)} validated rows to {OUTPUT_VALIDATED_CSV}...")
#     df_validated.to_csv(OUTPUT_VALIDATED_CSV, index=False)
#
#     print(f"[-] Saving {len(df_remaining)} remaining latent rows to {OUTPUT_REMAINING_CSV}...")
#     df_remaining.to_csv(OUTPUT_REMAINING_CSV, index=False)
#
#     print("\n" + "=" * 40)
#     print("LLM VALIDATION SUMMARY")
#     print("=" * 40)
#     print(f"Total Clusters Analyzed: {len(clusters)}")
#     print(f"Clusters Validated by LLM: {len(df_validated['Component_ID'].unique())}")
#     print(f"Remaining Latent Clusters: {len(df_remaining['Component_ID'].unique())}")
#     print("=" * 40)
#
#
# if __name__ == '__main__':
#     main()

