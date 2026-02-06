#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd

# ==============================================================================
# 0) Config
# ==============================================================================
DTW_SIGMA = 35.0
SIMILARITY_THRESHOLD = 0.90
MAX_SEQ_LEN = 100

INPUT_CSV = f"./2.Global_Component_from_v3Final_vendor_type_{DTW_SIGMA}_{SIMILARITY_THRESHOLD:.2f}_cap{MAX_SEQ_LEN}_nopad.csv"

OUTPUT_COMPONENT_SUMMARY = f"./3.0.Component_Summary_Strict_cap{MAX_SEQ_LEN}_nopad.csv"
OUTPUT_CONSISTENT = f"./3.1.Flows_Consistent_Strict_cap{MAX_SEQ_LEN}_nopad.csv"
OUTPUT_INCONSISTENT = f"./3.2.Flows_Inconsistent_Strict_cap{MAX_SEQ_LEN}_nopad.csv"

# generic endpoint filtering switch
ENABLE_GENERIC_FILTER = True

# Your real header bindings
COL_COMPONENT = "Component_ID"
COL_DEVICE_COUNT = "Device_Count"
COL_DEVICE = "Device"
COL_VENDOR = "device_vendor"
COL_REMOTE_IP = "Remote_IP"
COL_SERVICE = "Domain_A"      # Service layer
COL_INFRA = "Domain_PTR"      # Infrastructure layer

# ==============================================================================
# 1) Generic filtering rules (evidence-only)
# ==============================================================================

HARD_GENERIC_PATTERNS = [
    # NTP / time
    r"^pool\.ntp\.org$",
    r".*\.pool\.ntp\.org$",
    r"^time\d*\.google\.com$",
    r"^time(-[a-z0-9-]+)?\.g\.aaplimg\.com$",
    r"^time\.microsoft\.akadns\.net$",
    r"^time-a-b\.nist\.gov$",
    r"^ntp\d*\.glb\.nist\.gov$",
    r"^utcnist2\.colorado\.edu$",
    r"^ntp-.*\.amazon\.com$",

    # Captive / connectivity check
    r"^connectivitycheck\.gstatic\.com$",
    r"^clients3\.google\.com$",
    r"^clients\.l\.google\.com$",
    r"^captive\.apple\.com$",
    r"^msftconnecttest\.com$",
    r"^www\.msftconnecttest\.com$",
    r"^msftncsi\.com$",
    r"^ipv4\.connman\.net$",

    # Google talk / messaging frontends
    r"^mtalk\.google\.com$",
    r"^mobile-gtalk\.l\.google\.com$",

    # OCSP/CRL (optional; comment out if you want to keep)
    r"^ocsp\..*$",
    r"^crl\..*$",
]

INFRA_ROOT_SUFFIXES = [
    # AWS/Amazon infra
    "amazonaws.com",
    "amazonaws.com.cn",
    "cloudfront.net",

    # Akamai
    "akamai.net",
    "akamaihd.net",
    "akamaiedge.net",
    "akamaitechnologies.com",
    "edgesuite.net",
    "edgekey.net",

    # Azure/Microsoft
    "cloudapp.net",
    "windows.net",
    "azureedge.net",
    "azurefd.net",
    "trafficmanager.net",
    "azurewebsites.net",

    # Google infra-ish
    "googleusercontent.com",
    "1e100.net",

    # Huawei reverse-ish
    "hwclouds-dns.com",

    # LLNW
    "llnwd.net",

    # Fastly
    "fastly.net",
]

# patterns that are *specific enough* (so not treated as generic even if under infra roots)
ALLOW_SPECIFIC_PATTERNS = [
    # --- AWS ---
    r"^[a-z0-9][a-z0-9\.\-]{1,61}[a-z0-9]\.s3([.-][a-z0-9\-]+)?\.amazonaws\.com(\.cn)?$",
    r"^[a-z0-9]{8,}\.execute-api\.[a-z0-9\-]+\.amazonaws\.com$",
    r"^.+\.[a-z0-9\-]+\.elb\.amazonaws\.com$",
    r"^.+\.elb\.amazonaws\.com$",
    r"^.+\.aws-prd\.net$",

    # --- Aliyun ---
    r"^[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]\.oss-[a-z0-9\-]+\.aliyuncs\.com$",
    r"^[a-z0-9\-\.]+\.[a-z0-9\-]+\.log\.aliyuncs\.com$",

    # --- Azure ---
    r"^[a-z0-9]{3,24}\.(blob|file|queue|table)\.core\.windows\.net$",
    r"^[a-z0-9\-]{8,}\.cloudapp\.net$",

    # --- Netflix business naming ---
    r"^.+\.prodaa\.netflix\.com$",
    r"^.+\.nflxso\.net$",

    # --- Akamai structured edge hostnames ---
    r"^e\d+\.[a-z0-9\-]+\.(akamaiedge\.net)$",
]

_RE_HARD = [re.compile(p, re.IGNORECASE) for p in HARD_GENERIC_PATTERNS]
_RE_ALLOW = [re.compile(p, re.IGNORECASE) for p in ALLOW_SPECIFIC_PATTERNS]


def normalize_host(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s in ("none", "nan", "null", ""):
        return ""
    if s.endswith("."):
        s = s[:-1]
    return s


def pick_first_nonempty_pipe_value(x) -> str:
    s = normalize_host(x)
    if not s:
        return ""
    if "|" not in s:
        return s
    parts = [p.strip() for p in s.split("|") if p.strip()]
    return parts[0] if parts else ""


def is_allow_specific(host: str) -> bool:
    for r in _RE_ALLOW:
        if r.match(host):
            return True
    return False


def is_hard_generic(host: str) -> bool:
    for r in _RE_HARD:
        if r.match(host):
            return True
    return False


def is_infra_root(host: str) -> bool:
    for suf in INFRA_ROOT_SUFFIXES:
        if host.endswith(suf):
            return True
    return False


def is_generic_public_service(host: str) -> bool:
    h = normalize_host(host)
    if not h:
        return False
    # allow-specific overrides generic judgement
    if is_allow_specific(h):
        return False
    if is_hard_generic(h):
        return True
    if is_infra_root(h):
        return True
    return False


# ==============================================================================
# 2) Strict consistency logic
# ==============================================================================

def strict_consistency_check_component(group_df: pd.DataFrame,
                                      ip_col: str,
                                      svc_col_evi: str,
                                      infra_col_evi: str):
    """
    Strict:
      - IP: all non-empty and exactly one unique
      - Service: all non-empty and exactly one unique
      - Infra: all non-empty and exactly one unique
    """
    res = {
        "Is_Strict_Implicit": False,
        "Consistent_Layers": [],
        "Cluster_Val_IP": None,
        "Cluster_Val_Service": None,
        "Cluster_Val_Infra": None,
    }

    # IP
    ips = group_df[ip_col].astype(str).str.strip().replace({"nan": "", "None": ""})
    if (ips != "").all():
        uniq = ips.unique().tolist()
        if len(uniq) == 1:
            res["Cluster_Val_IP"] = uniq[0]
            res["Consistent_Layers"].append("IP")
            res["Is_Strict_Implicit"] = True

    # Service
    svcs = group_df[svc_col_evi].astype(str).str.strip().replace({"nan": "", "None": ""})
    if (svcs != "").all():
        uniq = svcs.unique().tolist()
        if len(uniq) == 1:
            res["Cluster_Val_Service"] = uniq[0]
            res["Consistent_Layers"].append("Service")
            res["Is_Strict_Implicit"] = True

    # Infra
    infs = group_df[infra_col_evi].astype(str).str.strip().replace({"nan": "", "None": ""})
    if (infs != "").all():
        uniq = infs.unique().tolist()
        if len(uniq) == 1:
            res["Cluster_Val_Infra"] = uniq[0]
            res["Consistent_Layers"].append("Infrastructure")
            res["Is_Strict_Implicit"] = True

    res["Consistent_Layers"] = "|".join(res["Consistent_Layers"])
    return res


# ==============================================================================
# 3) Main pipeline
# ==============================================================================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    try:
        df = pd.read_csv(INPUT_CSV, dtype={COL_REMOTE_IP: str})
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, encoding="gbk", dtype={COL_REMOTE_IP: str})

    # sanity check required columns
    required_cols = [COL_COMPONENT, COL_DEVICE_COUNT, COL_DEVICE, COL_VENDOR, COL_REMOTE_IP, COL_SERVICE, COL_INFRA]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # normalize
    df[COL_COMPONENT] = df[COL_COMPONENT].astype(str).str.strip()
    df[COL_DEVICE] = df[COL_DEVICE].astype(str).str.strip()
    df[COL_VENDOR] = df[COL_VENDOR].astype(str).str.strip().str.lower()
    df[COL_REMOTE_IP] = df[COL_REMOTE_IP].astype(str).str.strip()
    df[COL_DEVICE_COUNT] = pd.to_numeric(df[COL_DEVICE_COUNT], errors="coerce").fillna(0).astype(int)

    # --------------------------------------------------------------------------
    # Step 2: remove Device_Count == 1 components
    # --------------------------------------------------------------------------
    before = len(df)
    df = df[df[COL_DEVICE_COUNT] > 1].copy()
    print(f"[Filter] Remove Device_Count==1: rows {before} -> {len(df)}")

    # --------------------------------------------------------------------------
    # Step 3: remove same-vendor components (cross-vendor only)
    #   rule: within component, vendors all identical AND non-empty
    # --------------------------------------------------------------------------
    def is_single_vendor(group: pd.DataFrame) -> bool:
        vs = [v for v in group[COL_VENDOR].tolist() if v and v not in ("nan", "none", "")]
        if not vs:
            return False  # cannot conclude same-vendor if missing
        return len(set(vs)) == 1

    before = len(df)
    single_vendor_flags = df.groupby(COL_COMPONENT, sort=False).apply(is_single_vendor)
    drop_components = set(single_vendor_flags[single_vendor_flags].index.tolist())
    df = df[~df[COL_COMPONENT].isin(drop_components)].copy()
    print(f"[Filter] Remove same-vendor comps: rows {before} -> {len(df)} (dropped comps={len(drop_components)})")

    # --------------------------------------------------------------------------
    # Step 4: generic filtering evidence-only (keep originals in output)
    #   Evidence columns used for strict checks
    # --------------------------------------------------------------------------
    df["Service_Orig"] = df[COL_SERVICE]
    df["Infra_Orig"] = df[COL_INFRA]

    df["Service_Norm"] = df["Service_Orig"].apply(pick_first_nonempty_pipe_value)
    df["Infra_Norm"] = df["Infra_Orig"].apply(pick_first_nonempty_pipe_value)

    if ENABLE_GENERIC_FILTER:
        def to_evidence(host):
            h = normalize_host(host)
            if not h:
                return ""
            return "" if is_generic_public_service(h) else h

        df["Service_Evidence"] = df["Service_Norm"].apply(to_evidence)
        df["Infra_Evidence"] = df["Infra_Norm"].apply(to_evidence)
        print("[Generic] ENABLED: evidence blanks generic endpoints; originals preserved.")
    else:
        df["Service_Evidence"] = df["Service_Norm"]
        df["Infra_Evidence"] = df["Infra_Norm"]
        print("[Generic] DISABLED: evidence equals normalized originals.")

    # --------------------------------------------------------------------------
    # Step 5: strict consistency at component level
    # --------------------------------------------------------------------------
    print(f"[Strict] Checking {df[COL_COMPONENT].nunique()} components...")

    comp_rows = []
    for comp_id, g in df.groupby(COL_COMPONENT, sort=False):
        r = strict_consistency_check_component(
            g,
            ip_col=COL_REMOTE_IP,
            svc_col_evi="Service_Evidence",
            infra_col_evi="Infra_Evidence"
        )
        r[COL_COMPONENT] = comp_id
        r["Num_Flows"] = len(g)
        r["Num_Devices"] = g[COL_DEVICE].nunique()
        r["Num_Vendors"] = len(set([v for v in g[COL_VENDOR].tolist() if v and v not in ("nan", "none", "")]))
        comp_rows.append(r)

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(OUTPUT_COMPONENT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"[Out] Component summary saved: {OUTPUT_COMPONENT_SUMMARY}")

    # --------------------------------------------------------------------------
    # Step 6: backfill to flow-level and split outputs
    # --------------------------------------------------------------------------
    df_final = df.merge(df_comp, on=COL_COMPONENT, how="left")

    df_consistent = df_final[df_final["Is_Strict_Implicit"] == True].copy()
    df_inconsistent = df_final[df_final["Is_Strict_Implicit"] == False].copy()

    df_consistent.to_csv(OUTPUT_CONSISTENT, index=False, encoding="utf-8-sig")
    df_inconsistent.to_csv(OUTPUT_INCONSISTENT, index=False, encoding="utf-8-sig")

    # Stats
    total_components = len(df_comp)
    strict_components = int(df_comp["Is_Strict_Implicit"].sum())
    print("\n" + "=" * 60)
    print("STRICT CONSISTENCY STATISTICS (after filtering)")
    print("=" * 60)
    print(f"Components total: {total_components}")
    pct = (strict_components / total_components * 100.0) if total_components else 0.0
    print(f"Strict implicit:  {strict_components} ({pct:.2f}%)")
    print(f"Inconsistent:     {total_components - strict_components}")
    print("-" * 60)
    print(f"Flows consistent:   {len(df_consistent)}")
    print(f"Flows inconsistent: {len(df_inconsistent)}")
    print("=" * 60)
    print(f"[Out] Consistent flows:   {OUTPUT_CONSISTENT}")
    print(f"[Out] Inconsistent flows: {OUTPUT_INCONSISTENT}")


if __name__ == "__main__":
    main()
