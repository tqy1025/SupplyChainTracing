import os
import re
import pickle
import pandas as pd
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP, DNS, DNSRR
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

# ================= 配置区域 =================
ROOT_DIRS = [
    "xxxxxxxx",
    "xxxxxxxx",
]

# 假设CSV文件的路径
CSV_FILE_PATH = "files_with_suggested_ips.csv"

# 结果保存路径
OUTPUT_PICKLE = "traffic_features_total_signed.pkl"
CHECKPOINT_LOG = "processed_files_total_signed1.log"

# 并行进程数
MAX_WORKERS = 64

# --- 新增配置 ---
# 流切分超时时间 (秒)。如果同一五元组的包间隔超过此时间，视为新流。
FLOW_TIMEOUT = 120
# 最小流长度 (包数量)。小于此数量的流将被丢弃(通常过滤只有握手的包)。
MIN_FLOW_LENGTH = 3


# ===========================================

def load_processed_files():
    """读取断点续传日志"""
    if not os.path.exists(CHECKPOINT_LOG):
        return set()
    with open(CHECKPOINT_LOG, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)


def append_processed_file(file_path):
    """写入断点日志"""
    with open(CHECKPOINT_LOG, 'a', encoding='utf-8') as f:
        f.write(file_path + "\n")


def parse_ip_from_filename(filename):
    """从文件名解析IP"""
    match = re.search(r'(\d{1,3}_\d{1,3}_\d{1,3}_\d{1,3})', filename)
    if match:
        return match.group(1).replace('_', '.')
    return None


def get_task_list(root_dirs, csv_path):
    """扫描目录并生成任务列表"""
    tasks = []
    csv_lookup = {}

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, dtype=str)
            for _, row in df.iterrows():
                norm_path = os.path.normpath(row['Full_Path'])
                csv_lookup[norm_path] = row['Manual_IP']
        except Exception as e:
            print(f"[Warning] 读取CSV失败: {e}")

    print("正在扫描文件构建任务列表...")
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            rel_path = os.path.relpath(root, root_dir)
            if rel_path == '.': continue

            device_name = rel_path.split(os.sep)[0]

            for file in files:
                if not file.endswith('.pcap'): continue

                full_path = os.path.join(root, file)
                norm_path = os.path.normpath(full_path)

                parts = file.split('_')
                if len(parts) >= 2 and parts[-2] == 'app':
                    continue

                device_ip = parse_ip_from_filename(file)

                if not device_ip:
                    if norm_path in csv_lookup:
                        manual_ip = csv_lookup[norm_path]
                        if str(manual_ip) == '0': continue
                        device_ip = manual_ip
                    else:
                        continue

                tasks.append({
                    'full_path': full_path,
                    'filename': file,
                    'device_name': device_name,
                    'device_ip': device_ip
                })
    return tasks

def process_single_pcap(task):
    """
    子进程执行单元：
    1. 提取全包长度 (Total Length)
    2. 使用 TCP FIN/RST + 时间窗口 进行流切分
    """
    pcap_path = task['full_path']
    device_ip = task['device_ip']
    device_name = task['device_name']
    filename = task['filename']

    flow_results = []

    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        return {'status': 'error', 'msg': str(e), 'path': pcap_path}

    # 1. DNS 映射 (保持不变)
    ip_to_domain = {}
    for pkt in packets:
        if pkt.haslayer(DNS) and pkt.haslayer(DNSRR):
            try:
                ans_count = pkt[DNS].ancount
                for i in range(ans_count):
                    ans = pkt[DNS].an[i]
                    if ans.type == 1:
                        rrname = ans.rrname.decode('utf-8').rstrip('.')
                        ip_to_domain[ans.rdata] = rrname
            except Exception:
                pass

    # 2. 初始数据提取 (Raw Data Extraction)
    raw_flows = defaultdict(list)

    for pkt in packets:
        if IP in pkt and (TCP in pkt or UDP in pkt):
            src = pkt[IP].src
            dst = pkt[IP].dst

            if src != device_ip and dst != device_ip:
                continue

            remote_ip = dst if src == device_ip else src
            proto_obj = TCP if TCP in pkt else UDP
            proto_name = "TCP" if TCP in pkt else "UDP"

            sport = pkt[proto_obj].sport
            dport = pkt[proto_obj].dport

            # Key: (Protocol, Remote_IP, Sorted_Ports)
            ports_key = tuple(sorted((sport, dport)))
            flow_key = (proto_name, remote_ip, ports_key)

            # 提取全包长度
            total_len = len(pkt)
            sign = 1 if src == device_ip else -1
            signed_size = total_len * sign

            # --- 新增: 提取 TCP Flags ---
            # Scapy flags: F=0x01, S=0x02, R=0x04, P=0x08, A=0x10, U=0x20
            tcp_flags = 0
            if TCP in pkt:
                # 获取整数形式的 flags
                tcp_flags = int(pkt[TCP].flags)

            raw_flows[flow_key].append({
                'ts': float(pkt.time),
                'signed_size': signed_size,
                'flags': tcp_flags  # 存储 flags 用于后续切分
            })

    # 3. 流切分 (基于 FIN/RST 和 Timeout)
    grouped_sessions = defaultdict(list)

    for (proto, remote_ip, ports), pkts in raw_flows.items():
        # 按时间排序
        pkts.sort(key=lambda x: x['ts'])

        current_session = []
        last_ts = -1.0
        force_new_session = False  # 标记是否需要强制开始新流

        for p in pkts:
            curr_ts = p['ts']
            flags = p['flags']

            # 判断是否切分：
            # 条件1: 上一个包是 FIN/RST，强制当前包开始新流
            # 条件2: 时间超时 (作为UDP或异常TCP的补充)
            is_timeout = (last_ts != -1.0) and ((curr_ts - last_ts) > FLOW_TIMEOUT)

            if force_new_session or is_timeout:
                if len(current_session) >= MIN_FLOW_LENGTH:
                    grouped_sessions[(proto, remote_ip)].append(current_session)
                current_session = []
                force_new_session = False  # 重置标记

            # 将当前包加入会话
            current_session.append(p['signed_size'])
            last_ts = curr_ts

            # --- 核心修改: 检查当前包是否包含 FIN(0x01) 或 RST(0x04) ---
            # 如果包含，说明当前流在这个包结束后应该终止
            # 下一个包 (if any) 将属于新流
            if proto == "TCP":
                # 位运算检查: 0x01=FIN, 0x04=RST
                if (flags & 0x01) or (flags & 0x04):
                    force_new_session = True

        # 循环结束，保存最后一个流
        if len(current_session) >= MIN_FLOW_LENGTH:
            grouped_sessions[(proto, remote_ip)].append(current_session)

    # 4. 结果组装 (保持不变)
    for (proto, remote_ip), sessions in grouped_sessions.items():
        domain = ip_to_domain.get(remote_ip, None)

        for idx, payload_sequence in enumerate(sessions):
            record = {
                'Domain': domain,
                'Remote_IP': remote_ip,
                'Device': device_name,
                'Protocol': proto,
                'Device_IP': device_ip,
                'Pcap_Filename': filename,
                'Flow_Index': idx,
                'Payload_Sequence': payload_sequence,
                'Source_Full_Path': pcap_path
            }
            flow_results.append(record)

    return {'status': 'success', 'data': flow_results, 'path': pcap_path}

def main():
    processed_files = load_processed_files()
    print(f"已处理文件断点: {len(processed_files)}")

    all_tasks = get_task_list(ROOT_DIRS, CSV_FILE_PATH)
    tasks_to_do = [t for t in all_tasks if t['full_path'] not in processed_files]
    print(f"总PCAP: {len(all_tasks)}, 待处理: {len(tasks_to_do)}")

    if not tasks_to_do:
        print("所有文件已处理完毕。")
        return

    final_results = []
    if os.path.exists(OUTPUT_PICKLE):
        print(f"加载现有 Pickle: {OUTPUT_PICKLE}")
        try:
            with open(OUTPUT_PICKLE, 'rb') as f:
                final_results = pickle.load(f)
        except Exception:
            final_results = []

    save_interval = 100
    counter = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_pcap, task): task for task in tasks_to_do}
        pbar = tqdm(total=len(tasks_to_do), desc="Processing")

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    if result['data']:
                        final_results.extend(result['data'])
                    append_processed_file(result['path'])
                    counter += 1
                else:
                    print(f"\nErr: {result.get('msg')} -> {result['path']}")
            except Exception as e:
                print(f"\nCritical: {e}")

            pbar.update(1)

            if counter % save_interval == 0 and counter > 0:
                temp_pkl = OUTPUT_PICKLE + ".tmp"
                with open(temp_pkl, 'wb') as f:
                    pickle.dump(final_results, f)
                if os.path.exists(OUTPUT_PICKLE):
                    os.remove(OUTPUT_PICKLE)
                os.rename(temp_pkl, OUTPUT_PICKLE)

        pbar.close()

    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump(final_results, f)
    print(f"完成! 结果保存至 {OUTPUT_PICKLE}, 共提取 {len(final_results)} 条流。")


if __name__ == '__main__':

    main()
