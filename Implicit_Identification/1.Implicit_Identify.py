#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pickle
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd


# ==============================================================================
# 0) Config
# ==============================================================================
INPUT_FILE = '../Traffic_Cleaned_v3_Final.pkl'  # Tier-2 output (flow-level)
OUT_DIR = './Implicit_Relationships'
os.makedirs(OUT_DIR, exist_ok=True)

MIN_DEVICES = 2  # at least 2 devices in a set

# Optional generic public service filtering
ENABLE_GENERIC_FILTER = True
# Choose which layers to apply generic filtering on:
# {"Service","Infrastructure"} is typical; you may also include "IP" if you want.
GENERIC_FILTER_LAYERS = {"Service", "Infrastructure"}

# How to treat unknown vendor
COUNT_UNKNOWN_VENDOR_AS_VENDOR = True   # if False, 'unknown' won't contribute to vendor-set size

# Output files
OUTPUT_TARGETLEVEL_CSV = os.path.join(OUT_DIR, "Targets_Implicit_DedupByFlow.csv")
OUTPUT_PKL = os.path.join(OUT_DIR, "Intersections_Implicit.pkl")
OUTPUT_STATS_CSV = os.path.join(OUT_DIR, "Intersections_Stats_DedupByFlow.csv")


# Column bindings (expect these columns exist; will fill if missing)
COL_DEVICE = 'Device'
COL_VENDOR = 'Vendor'
COL_REMOTE_IP = 'Remote_IP'
COL_SERVICE = 'Domain_A'
COL_INFRA = 'Domain_PTR'


# ==============================================================================
# 1) Generic endpoint rules (from your provided list)
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


# ==============================================================================
# 2) Helpers
# ==============================================================================
def norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("none", "nan", "null", ""):
        return ""
    if s.endswith("."):
        s = s[:-1]
    return s


def pick_first_nonempty_pipe_value(x) -> str:
    """Domain_A/Domain_PTR may be 'a|b', pick first non-empty 'a'."""
    s = norm_str(x).lower()
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
    """
    Generic judgement:
      - allow-specific overrides
      - hard generic patterns
      - infra root suffixes
    """
    h = norm_str(host).lower()
    if not h:
        return False
    if is_allow_specific(h):
        return False
    if is_hard_generic(h):
        return True
    if is_infra_root(h):
        return True
    return False


def apply_generic_filter(layer_name: str, target_norm: str) -> str:
    """
    If enabled & layer selected: blank out generic endpoints.
    Otherwise: keep target.
    """
    if not ENABLE_GENERIC_FILTER:
        return target_norm
    if layer_name not in GENERIC_FILTER_LAYERS:
        return target_norm
    if not target_norm:
        return ""
    return "" if is_generic_public_service(target_norm) else target_norm


def hash_flowuid_list(flow_ids) -> str:
    """
    Stable hash for Flow_UID set (order-independent):
      - sort ints
      - join with comma
      - sha1
    """
    if not flow_ids:
        return ""
    ids = sorted(int(x) for x in flow_ids)
    s = ",".join(map(str, ids))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def extract_vendors(device_set, vendor_map):
    vendors = set()
    for d in device_set:
        v = vendor_map.get(d, 'unknown')
        v = norm_str(v).lower() if v is not None else 'unknown'
        if not COUNT_UNKNOWN_VENDOR_AS_VENDOR and v == 'unknown':
            continue
        vendors.add(v)
    return vendors


# ==============================================================================
# 3) Core: generate candidates per layer (cross-vendor + optional generic filter)
# ==============================================================================
def generate_layer_candidates_cross_vendor(df: pd.DataFrame,
                                          layer_name: str,
                                          target_col: str,
                                          vendor_map: dict,
                                          min_devices: int = 2):
    """
    Build cross-vendor candidate sets for a given layer.

    Returns:
      targets_all: {target: set(devices)}           (after min_devices + generic-row filter)
      targets_implicit: {target: set(devices)}      (cross-vendor)
      rows_out: list[dict]                          (target-level rows w/ flow-set hash)
      stats: dict
    """
    print(f"[-] Generating candidates for {layer_name} (col={target_col}) ...")

    # normalize target (pipe -> first value)
    tgt_norm = df[target_col].apply(pick_first_nonempty_pipe_value)

    # evidence after optional generic filter
    tgt_evi = tgt_norm.apply(lambda x: apply_generic_filter(layer_name, x))

    # pre-group filtering:
    valid_mask = tgt_evi.notna() & (tgt_evi.astype(str).str.strip() != "")
    dropped_generic_rows = int((~valid_mask).sum())

    df_valid = df.loc[valid_mask, [COL_DEVICE, "Flow_UID"]].copy()
    df_valid["_Target"] = tgt_evi[valid_mask].astype(str)

    if df_valid.empty:
        return {}, {}, [], {
            "layer": layer_name,
            "targets_all": 0,
            "targets_implicit": 0,
            "dropped_explicit_targets": 0,
            "ignored_small_targets": 0,
            "dropped_generic_rows(pre-group)": dropped_generic_rows,
        }

    # group by target
    g = df_valid.groupby("_Target", sort=False)

    # target -> unique devices
    devs_by_target = g[COL_DEVICE].unique().to_dict()

    # target -> flow uid list (unique, sorted)
    flowuids_by_target = g["Flow_UID"].apply(
        lambda x: sorted(pd.to_numeric(x, errors="coerce").dropna().astype(int).unique().tolist())
    ).to_dict()

    # filter by MIN_DEVICES
    targets_all = {}
    ignored_small_targets = 0
    for t, devs in devs_by_target.items():
        if len(devs) >= min_devices:
            targets_all[t] = set(map(str, devs))
        else:
            ignored_small_targets += 1

    # cross-vendor filter
    targets_implicit = {}
    dropped_explicit_targets = 0

    rows_out = []
    for t, devset in targets_all.items():
        vendors = extract_vendors(devset, vendor_map)

        if len(vendors) <= 1:
            dropped_explicit_targets += 1
            continue

        targets_implicit[t] = devset

        flow_list = flowuids_by_target.get(t, [])
        flow_hash = hash_flowuid_list(flow_list)

        rows_out.append({
            "Layer": layer_name,
            "Target": t,
            "Num_Devices": len(devset),
            "Device_List": "|".join(sorted(list(devset)))[:2000],
            "Num_Vendors": len(vendors),
            "Vendors": "|".join(sorted(list(vendors)))[:2000],
            "Is_Implicit": True,

            "FlowUID_Count": len(flow_list),
            "FlowSet_Hash": flow_hash,

            # 如果你想 debug，可以保留下面这列；不想太大就注释掉
            "FlowUID_Set": ",".join(map(str, flow_list))[:50000],
        })

    stats = {
        "layer": layer_name,
        "targets_all": len(targets_all),
        "targets_implicit": len(targets_implicit),
        "dropped_explicit_targets": dropped_explicit_targets,
        "ignored_small_targets": ignored_small_targets,
        "dropped_generic_rows(pre-group)": dropped_generic_rows,
    }

    print(f"    > targets_all (>=MIN_DEVICES): {stats['targets_all']}")
    print(f"    > targets_implicit (cross-vendor): {stats['targets_implicit']}")
    print(f"    > dropped_explicit_targets: {stats['dropped_explicit_targets']}")
    print(f"    > ignored_small_targets: {stats['ignored_small_targets']}")
    print(f"    > dropped_generic_rows(pre-group): {stats['dropped_generic_rows(pre-group)']}")

    return targets_all, targets_implicit, rows_out, stats


def dedup_across_layers_by_flowhash(rows_all):
    """
    Cross-layer dedup by FlowSet_Hash.
    Key = (FlowUID_Count, FlowSet_Hash) to reduce collision risk.
    """
    key_to_gid = {}
    gid_to_count = defaultdict(int)
    next_gid = 1

    for r in rows_all:
        cnt = int(r.get("FlowUID_Count", 0))
        h = r.get("FlowSet_Hash", "")
        if not h:
            # Should not happen, but keep it safe
            h = f"EMPTY::{r.get('Layer','?')}::{r.get('Target','?')}"
            r["FlowSet_Hash"] = h

        key = (cnt, h)

        if key not in key_to_gid:
            gid = f"SET_{next_gid:06d}"
            key_to_gid[key] = gid
            next_gid += 1
        else:
            gid = key_to_gid[key]

        r["Dedup_Group_ID"] = gid
        gid_to_count[gid] += 1

    for r in rows_all:
        r["Dedup_Group_Size"] = gid_to_count.get(r["Dedup_Group_ID"], 0)

    return rows_all, len(key_to_gid)


# ==============================================================================
# 4) Main
# ==============================================================================
def main():
    print("=" * 80)
    print("[+] Candidate intersection mining (cross-vendor + optional generic) + dedup by Flow_UID hash")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Load traffic PKL
    df = pd.read_pickle(INPUT_FILE)
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Traffic PKL must be a DataFrame or list-of-dicts.")

    # Ensure column compatibility
    if COL_DEVICE not in df.columns and "Device_ID" in df.columns:
        df.rename(columns={"Device_ID": COL_DEVICE}, inplace=True)

    for c in [COL_DEVICE, COL_VENDOR, COL_REMOTE_IP, COL_SERVICE, COL_INFRA]:
        if c not in df.columns:
            df[c] = None

    # Attach Flow_UID (row index)
    df = df.reset_index(drop=True)
    df["Flow_UID"] = df.index.astype(int)

    # Build vendor map from traffic PKL Vendor column
    # (Device -> Vendor)
    vendor_map = (
        df[[COL_DEVICE, COL_VENDOR]]
        .dropna(subset=[COL_DEVICE])
        .drop_duplicates(subset=[COL_DEVICE])
        .set_index(COL_DEVICE)[COL_VENDOR]
        .to_dict()
    )

    print(f"[-] Vendor map built from Traffic PKL: {len(vendor_map)} devices (missing -> unknown)")

    # Generate candidates per layer
    results_implicit = {}
    all_rows = []
    stats_rows = []

    # IP
    _, implicit_ip, rows_ip, stats_ip = generate_layer_candidates_cross_vendor(
        df, "IP", COL_REMOTE_IP, vendor_map, MIN_DEVICES
    )
    results_implicit["IP"] = implicit_ip
    all_rows.extend(rows_ip)
    stats_rows.append(stats_ip)

    # Service
    _, implicit_svc, rows_svc, stats_svc = generate_layer_candidates_cross_vendor(
        df, "Service", COL_SERVICE, vendor_map, MIN_DEVICES
    )
    results_implicit["Service"] = implicit_svc
    all_rows.extend(rows_svc)
    stats_rows.append(stats_svc)

    # Infrastructure
    _, implicit_inf, rows_inf, stats_inf = generate_layer_candidates_cross_vendor(
        df, "Infrastructure", COL_INFRA, vendor_map, MIN_DEVICES
    )
    results_implicit["Infrastructure"] = implicit_inf
    all_rows.extend(rows_inf)
    stats_rows.append(stats_inf)

    # Cross-layer dedup
    all_rows, dedup_group_count = dedup_across_layers_by_flowhash(all_rows)

    # Save target-level CSV
    df_out = pd.DataFrame(all_rows)
    # Reorder columns for readability
    col_order = [
        "Layer", "Target",
        "Num_Devices", "Device_List",
        "Num_Vendors", "Vendors",
        "Is_Implicit",
        "FlowUID_Count", "FlowSet_Hash", "Dedup_Group_ID", "Dedup_Group_Size",
        "FlowUID_Set",
    ]
    col_order = [c for c in col_order if c in df_out.columns] + [c for c in df_out.columns if c not in col_order]
    df_out = df_out[col_order]

    df_out.to_csv(OUTPUT_TARGETLEVEL_CSV, index=False, encoding="utf-8-sig")
    print(f"[Out] Target-level implicit sets (dedup annotated): {OUTPUT_TARGETLEVEL_CSV}")

    # Save implicit candidates pkl
    with open(OUTPUT_CANDIDATES_PKL, "wb") as f:
        pickle.dump(results_implicit, f)
    print(f"[Out] Cross-vendor candidates saved: {OUTPUT_CANDIDATES_PKL}")

    # Save stats
    df_stats = pd.DataFrame(stats_rows)
    # Add summary numbers
    summary = {
        "layer": "ALL_SUMMARY",
        "targets_all": int(df_stats["targets_all"].sum()),
        "targets_implicit": int(df_stats["targets_implicit"].sum()),
        "dropped_explicit_targets": int(df_stats["dropped_explicit_targets"].sum()),
        "ignored_small_targets": int(df_stats["ignored_small_targets"].sum()),
        "dropped_generic_rows(pre-group)": int(df_stats["dropped_generic_rows(pre-group)"].sum()),
    }
    df_stats = pd.concat([df_stats, pd.DataFrame([summary])], ignore_index=True)
    df_stats.to_csv(OUTPUT_STATS_CSV, index=False, encoding="utf-8-sig")
    print(f"[Out] Stats saved: {OUTPUT_STATS_CSV}")

    # Print summary (cross-vendor after filters)
    print("\n" + "=" * 80)
    print("SUMMARY (cross-vendor, after filters)")
    print("=" * 80)
    ip_sets = stats_ip["targets_implicit"]
    svc_sets = stats_svc["targets_implicit"]
    inf_sets = stats_inf["targets_implicit"]
    print(f"IP sets:             {ip_sets}")
    print(f"Service sets:        {svc_sets}")
    print(f"Infrastructure sets: {inf_sets}")
    print("-" * 40)
    print(f"Total dedup groups (by FlowUID_Count+FlowSet_Hash): {dedup_group_count}")
    print("=" * 80)

    if ENABLE_GENERIC_FILTER:
        print(f"[Generic] ENABLED on layers: {sorted(list(GENERIC_FILTER_LAYERS))}")
    else:
        print("[Generic] DISABLED")
    print("[+] Done.")


if __name__ == "__main__":
    main()

