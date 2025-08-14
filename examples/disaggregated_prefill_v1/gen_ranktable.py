import argparse
import json
import os

import torch.distributed as dist

from vllm_ascend.utils import AscendSocVersion, init_ascend_soc_version, get_ascend_soc_version

parser = argparse.ArgumentParser(
    description="Arguments of rank table generator", )
parser.add_argument("--local-host", type=str, required=True, help="local ip")
parser.add_argument("--prefill-device-cnt",
                    type=int,
                    required=True,
                    help="number of prefill devices")
parser.add_argument("--decode-device-cnt",
                    type=int,
                    required=True,
                    help="number of decode devices")
parser.add_argument("--prefill-device-ids",
                    type=str,
                    help="comma-separated list of prefill device IDs (e.g., '0,1')")
parser.add_argument("--decode-device-ids",
                    type=str,
                    help="comma-separated list of decode device IDs (e.g., '4,5')")
args = parser.parse_args()
local_host = args.local_host
prefill_device_cnt = args.prefill_device_cnt
decode_device_cnt = args.decode_device_cnt
prefill_device_ids = args.prefill_device_ids
decode_device_ids = args.decode_device_ids

# Parse device IDs if provided
if prefill_device_ids:
    prefill_device_ids = [int(x.strip()) for x in prefill_device_ids.split(',')]
    if len(prefill_device_ids) != prefill_device_cnt:
        raise ValueError(f"Number of prefill device IDs ({len(prefill_device_ids)}) must match prefill_device_cnt ({prefill_device_cnt})")
else:
    prefill_device_ids = None

if decode_device_ids:
    decode_device_ids = [int(x.strip()) for x in decode_device_ids.split(',')]
    if len(decode_device_ids) != decode_device_cnt:
        raise ValueError(f"Number of decode device IDs ({len(decode_device_ids)}) must match decode_device_cnt ({decode_device_cnt})")
else:
    decode_device_ids = None

print("enter py")

hccn_tool_path = os.environ.get("HCCN_TOOL_PATH",
                                "/usr/local/Ascend/driver/tools/hccn_tool")
master_addr = os.environ.get("MASTER_ADDR")
master_port = os.environ.get("MASTER_PORT")
rank = os.environ.get("RANK")
local_rank = os.environ.get("LOCAL_RANK")
# This variable is set by torchrun,
# and is different from WORLD_SIZE in gen_rank_table.sh.
world_size = os.environ.get("WORLD_SIZE")

init_ascend_soc_version()
soc_info = get_ascend_soc_version()


def get_cmd_stdout(cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True,
                          shell=True).stdout.decode("utf-8").strip()


print(f"local_host: {local_host}")
print("gen ranktable.json")

num_cards = get_cmd_stdout("npu-smi info -l | grep \"Total Count\"").split(
    ":")[1].strip()
num_cards = int(num_cards)
chips_per_card = get_cmd_stdout("npu-smi info -l | grep \"Chip Count\"").split(
    "\n")[0].split(":")[1].strip()
chips_per_card = int(chips_per_card)

# generate local device list for local rank 0, and gather it to all ranks
local_device_list: list[dict[str, str]] = list()
if local_rank == "0":
    super_pod_id = "0"
    for card_id in range(num_cards):
        for chip_id in range(chips_per_card):
            device_id = card_id * chips_per_card + chip_id
            if soc_info == AscendSocVersion.A3:
                device_ip = get_cmd_stdout(
                    f"{hccn_tool_path} -i {device_id} -vnic -g | grep ipaddr"
                ).split(":")[1].strip()
                super_device_id = get_cmd_stdout(
                    f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep SDID"
                ).split(":")[1].strip()
                super_pod_id = get_cmd_stdout(
                    f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep \"Super Pod ID\""
                ).split(":")[1].strip()
            else:
                device_ip = get_cmd_stdout(
                    f"{hccn_tool_path} -i {device_id} -ip -g | grep ipaddr"
                ).split(":")[1].strip()

            device_info = {
                "server_id": local_host,
                "device_id": str(device_id),
                "device_ip": str(device_ip),
            }
            if soc_info == AscendSocVersion.A3:
                device_info.update({
                    "super_pod_id": str(super_pod_id),
                    "super_device_id": str(super_device_id)
                })
            local_device_list.append(device_info)

dist.init_process_group(backend=dist.Backend.GLOO)
global_device_list = [None] * dist.get_world_size()
dist.all_gather_object(global_device_list, local_device_list)
global_device_list = [
    device_info for device_list in global_device_list
    for device_info in device_list  # type: ignore[attr-defined]
]
cnt = 1
for device_info in global_device_list:  # type: ignore[assignment]
    device_info["cluster_id"] = str(cnt)
    cnt += 1
assert (prefill_device_cnt + decode_device_cnt) <= len(global_device_list), \
"prefill_device_cnt + decode_device_cnt must be less than or equal to number of all devices in cluster"

# Select devices based on specified IDs or default sequential allocation
if prefill_device_ids is not None:
    # Find devices by specified IDs
    prefill_devices = []
    for device_id in prefill_device_ids:
        device_found = False
        for device_info in global_device_list:
            if int(device_info["device_id"]) == device_id and device_info["server_id"] == local_host:
                prefill_devices.append(device_info)
                device_found = True
                break
        if not device_found:
            raise ValueError(f"Prefill device ID {device_id} not found in global device list for server {local_host}")
else:
    # Default sequential allocation
    prefill_devices = global_device_list[:prefill_device_cnt]

if decode_device_ids is not None:
    # Find devices by specified IDs
    decode_devices = []
    for device_id in decode_device_ids:
        device_found = False
        for device_info in global_device_list:
            if int(device_info["device_id"]) == device_id and device_info["server_id"] == local_host:
                decode_devices.append(device_info)
                device_found = True
                break
        if not device_found:
            raise ValueError(f"Decode device ID {device_id} not found in global device list for server {local_host}")
else:
    # Default sequential allocation
    decode_devices = global_device_list[prefill_device_cnt:prefill_device_cnt + decode_device_cnt]

# Add rank_id to prefill devices (starting from 0)
for rank_id, device_info in enumerate(prefill_devices):
    device_info["rank_id"] = rank_id

# Add rank_id to decode devices (starting from 0)
for rank_id, device_info in enumerate(decode_devices):
    device_info["rank_id"] = rank_id

ranktable = {
    "version":
    "1.2",
    "server_count":
    str(world_size),
    "prefill_device_list":
    prefill_devices,
    "decode_device_list":
    decode_devices,
    "status":
    "completed"
}

if local_rank == '0':
    with open("ranktable.json", "w") as f:
        json.dump(ranktable, f, indent=4)

    print("gen ranktable.json done")
