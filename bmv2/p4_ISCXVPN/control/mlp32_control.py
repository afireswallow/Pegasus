import sys,pickle
import signal
import os,time
from pathlib import Path
from p4runtime_lib.helper import P4InfoHelper
from p4runtime_lib.switch import ShutdownAllSwitchConnections
import p4runtime_lib.bmv2


current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'p4runtime_lib'))

current_dir = Path(__file__).parent
p4info_file_path = Path("../build/mlp_ISCXVPN.p4info.txt")
bmv2_json_path = Path("../build/mlp_ISCXVPN.json")
model_file_path = Path("/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/ISCXVPN/mlp_ISCXVPN.pkl")

'''
 p4c-bm2-ss --target bmv2 --arch v1model --p4runtime-files build/mlp_ISCXVPN.p4info.txt -o build/mlp_ISCXVPN.json basic.p4

 sudo simple_switch_grpc -i 0@veth0 -i 1@veth1 --log-console --no-p4  -- --grpc-server-addr 127.0.0.1:50051 build/mlp_ISCXVPN.json > bmv2logs/run_switch.log 2>&1
'''

p4info_helper = P4InfoHelper(str(p4info_file_path))
switch_conn = p4runtime_lib.bmv2.Bmv2SwitchConnection(
    name='bmv2-mlp32',
    address='127.0.0.1:50051',
    device_id=0
)

switch_conn.MasterArbitrationUpdate()
switch_conn.SetForwardingPipelineConfig(
    p4info=p4info_helper.p4info,
    bmv2_json_file_path=str(bmv2_json_path)
)

# max_table_size = 4096
max_table_size = 5325


with open(model_file_path,"rb") as f:
    [table1, table2_0_1, table2_1_16, table2_3, table3] = pickle.load(f)

#load_tb_Feat(table_data: data to write to table, table_name: table name, feat_name: table attribute name, action_name: function to execute)

def load_tb_ternary_feat2_parameter16(table_data, table_name, feat_name1, feat_name2, action_name):
    print("load_tb--{}.......".format(table_name))
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]; print("ERROR! ")
    for x in range(len(table_data)):
        priority = (2**31 - 1) - table_data[x][0]
        table_entry = p4info_helper.buildTableEntry(
            table_name=table_name,
            match_fields={
                feat_name1: (int(table_data[x][1]), int(table_data[x][3])),
                feat_name2: (int(table_data[x][2]), int(table_data[x][4]))
            },
            action_name=action_name,
            action_params={
                'in01': table_data[x][-1][0] & 0xffffffff,
                'in02': table_data[x][-1][1] & 0xffffffff,
                'in03': table_data[x][-1][2] & 0xffffffff,
                'in04': table_data[x][-1][3] & 0xffffffff,
                'in05': table_data[x][-1][4] & 0xffffffff,
                'in06': table_data[x][-1][5] & 0xffffffff,
                'in07': table_data[x][-1][6] & 0xffffffff,
                'in08': table_data[x][-1][7] & 0xffffffff,
                'in11': table_data[x][-1][8] & 0xffffffff,
                'in12': table_data[x][-1][9] & 0xffffffff,
                'in13': table_data[x][-1][10] & 0xffffffff,
                'in14': table_data[x][-1][11] & 0xffffffff,
                'in15': table_data[x][-1][12] & 0xffffffff,
                'in16': table_data[x][-1][13] & 0xffffffff,
                'in17': table_data[x][-1][14] & 0xffffffff,
                'in18': table_data[x][-1][15] & 0xffffffff
            },
            priority=priority
        )
        try:
            switch_conn.WriteTableEntry(table_entry)
        except Exception as e:
            print(f"[ERROR] Failed to write entry {x} in {table_name}: {e}")
    print("load_to_table--{} completed!".format(table_name))

def load_tb_ternary_feat1_parameter16(table_data, table_name, feat_name, action_name):
    print("load_tb--{}.......".format(table_name))
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]; print("ERROR! ")
    for x in range(len(table_data)):
        priority = (2**31 - 1) - table_data[x][0]
        table_entry = p4info_helper.buildTableEntry(
            table_name=table_name,
            match_fields={
                feat_name: (int(table_data[x][1]), int(table_data[x][2]))
            },
            action_name=action_name,
            action_params={
                'in01': table_data[x][-1][0] & 0xffffffff,
                'in02': table_data[x][-1][1] & 0xffffffff,
                'in03': table_data[x][-1][2] & 0xffffffff,
                'in04': table_data[x][-1][3] & 0xffffffff,
                'in05': table_data[x][-1][4] & 0xffffffff,
                'in06': table_data[x][-1][5] & 0xffffffff,
                'in07': table_data[x][-1][6] & 0xffffffff,
                'in08': table_data[x][-1][7] & 0xffffffff,
                'in11': table_data[x][-1][8] & 0xffffffff,
                'in12': table_data[x][-1][9] & 0xffffffff,
                'in13': table_data[x][-1][10] & 0xffffffff,
                'in14': table_data[x][-1][11] & 0xffffffff,
                'in15': table_data[x][-1][12] & 0xffffffff,
                'in16': table_data[x][-1][13] & 0xffffffff,
                'in17': table_data[x][-1][14] & 0xffffffff,
                'in18': table_data[x][-1][15] & 0xffffffff
            },
            priority=priority
        )
        try:
            switch_conn.WriteTableEntry(table_entry)
        except Exception as e:
            print(f"[ERROR] Failed to write entry {x} in {table_name}: {e}")
    print("load_to_table--{} completed!".format(table_name))

def load_tb_ternary_feat1_parameter8(table_data, table_name, feat_name, action_name):
    print("load_tb--{}.......".format(table_name))
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]; print("ERROR! ")
    for x in range(len(table_data)):
        priority = (2**31 - 1) - table_data[x][0]
        table_entry = p4info_helper.buildTableEntry(
            table_name=table_name,
            match_fields={
                feat_name: (int(table_data[x][1]), int(table_data[x][2]))
            },
            action_name=action_name,
            action_params={
                'in01': table_data[x][-1][0] & 0xffffffff,
                'in02': table_data[x][-1][1] & 0xffffffff,
                'in03': table_data[x][-1][2] & 0xffffffff,
                'in04': table_data[x][-1][3] & 0xffffffff,
                'in05': table_data[x][-1][4] & 0xffffffff,
                'in06': table_data[x][-1][5] & 0xffffffff,
                'in07': table_data[x][-1][6] & 0xffffffff,
                'in08': table_data[x][-1][7] & 0xffffffff
            },
            priority=priority
        )
        try:
            switch_conn.WriteTableEntry(table_entry)
        except Exception as e:
            print(f"[ERROR] Failed to write entry {x} in {table_name}: {e}")
    print("load_to_table--{} completed!".format(table_name))

def load_tb_ternary_feat1_parameter1(table_data, table_name, feat_name, action_name):
    print("load_tb--{}.......".format(table_name))
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]; print("ERROR! ")
    for x in range(len(table_data)):
        priority = (2**31 - 1) - table_data[x][0]
        table_entry = p4info_helper.buildTableEntry(
            table_name=table_name,
            match_fields={
                feat_name: (int(table_data[x][1]), int(table_data[x][2]))
            },
            action_name=action_name,
            action_params={
                'in01': table_data[x][-1] & 0xf
            },
            priority=priority
        )
        try:
            switch_conn.WriteTableEntry(table_entry)
        except Exception as e:
            print(f"[ERROR] Failed to write entry {x}: {e}")
    print("load_to_table--{} completed!".format(table_name))

def load_tb_exact_feat3_parameter8(table_data, table_name, feat_name1, feat_name2, feat_name3, action_name):
    print("load_tb--{}.......".format(table_name))
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]; print("ERROR! ")
    for x in range(len(table_data)):
        table_entry = p4info_helper.buildTableEntry(
            table_name=table_name,
            match_fields={
                feat_name1: int(table_data[x][0]),
                feat_name2: int(table_data[x][1]),
                feat_name3: int(table_data[x][2])
            },
            action_name=action_name,
            action_params={
                'in01': table_data[x][-1][0] & 0xffffffff,
                'in02': table_data[x][-1][1] & 0xffffffff,
                'in03': table_data[x][-1][2] & 0xffffffff,
                'in04': table_data[x][-1][3] & 0xffffffff,
                'in05': table_data[x][-1][4] & 0xffffffff,
                'in06': table_data[x][-1][5] & 0xffffffff,
                'in07': table_data[x][-1][6] & 0xffffffff,
                'in08': table_data[x][-1][7] & 0xffffffff
            }
        )
        try:
            switch_conn.WriteTableEntry(table_entry)
        except Exception as e:
            print(f"[ERROR] Failed to write entry {x}: {e}")
    print("load_to_table--{} completed!".format(table_name))

def load_tb_ternary_feat1_parameter6(table_data, table_name, feat_name, action_name):
    print("load_tb--{}.......".format(table_name))
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]; print("ERROR! ")
    for x in range(len(table_data)):
        priority = (2**31 - 1) - table_data[x][0]
        table_entry = p4info_helper.buildTableEntry(
            table_name=table_name,
            match_fields={
                feat_name: (int(table_data[x][1] & 0xffffffff), int(table_data[x][2] & 0xffffffff))
            },
            action_name=action_name,
            action_params={
                'in01': table_data[x][-1][0] & 0xffffffff,
                'in02': table_data[x][-1][1] & 0xffffffff,
                'in03': table_data[x][-1][2] & 0xffffffff,
                'in04': table_data[x][-1][3] & 0xffffffff,
                'in05': table_data[x][-1][4] & 0xffffffff,
                'in06': table_data[x][-1][5] & 0xffffffff
            },
            priority=priority
        )
        try:
            switch_conn.WriteTableEntry(table_entry)
        except Exception as e:
            print(f"[ERROR] Failed to write entry {x}: {e}")
        #print("{}   {}   {}".format(table_data[x][-1][0], table_data[x][-1][1], table_data[x][-1][2]))
    print("load_to_table--{} completed!".format(table_name))

# Load table data - same as original mlp32_control.py
load_tb_ternary_feat2_parameter16(table1[0],  "MyIngress.ip_len_total_len", "hdr.feature.ip_len",   "hdr.feature.ip_total_len", "MyIngress.set_ip_len_total_len")
load_tb_ternary_feat2_parameter16(table1[1],  "MyIngress.protocol_tos",     "hdr.feature.protocol", "hdr.feature.tos",          "MyIngress.set_protocol_tos")
load_tb_ternary_feat2_parameter16(table1[2],  "MyIngress.offset_max_byte",  "hdr.feature.offset",   "hdr.feature.max_byte",     "MyIngress.set_offset_max_byte")
load_tb_ternary_feat2_parameter16(table1[3],  "MyIngress.min_byte_max_ipd", "hdr.feature.min_byte", "hdr.feature.max_ipd",      "MyIngress.set_min_byte_max_ipd")
load_tb_ternary_feat2_parameter16(table1[4],  "MyIngress.min_ipd_ipd",      "hdr.feature.min_ipd",  "hdr.feature.ipd",          "MyIngress.set_min_ipd_ipd")

load_tb_ternary_feat1_parameter8(table2_0_1[0], "MyIngress.h1_1", "meta.linear1.l32_1", "MyIngress.set_h1_1")

load_tb_ternary_feat1_parameter1(table2_1_16[0], "MyIngress.h1_2", "meta.linear1.l32_2", "MyIngress.set_h1_2")
load_tb_ternary_feat1_parameter1(table2_1_16[1], "MyIngress.h1_3", "meta.linear1.l32_3", "MyIngress.set_h1_3")
load_tb_ternary_feat1_parameter1(table2_1_16[2], "MyIngress.h1_4", "meta.linear1.l32_4", "MyIngress.set_h1_4")
load_tb_ternary_feat1_parameter1(table2_1_16[3], "MyIngress.h1_5", "meta.linear1.l32_5", "MyIngress.set_h1_5")
load_tb_ternary_feat1_parameter1(table2_1_16[4], "MyIngress.h1_6", "meta.linear1.l32_6", "MyIngress.set_h1_6")
load_tb_ternary_feat1_parameter1(table2_1_16[5], "MyIngress.h1_7", "meta.linear1.l32_7", "MyIngress.set_h1_7")
load_tb_ternary_feat1_parameter1(table2_1_16[6], "MyIngress.h1_8", "meta.linear1.l32_8", "MyIngress.set_h1_8")
load_tb_ternary_feat1_parameter1(table2_1_16[7], "MyIngress.h1_9", "meta.linear2.l32_1", "MyIngress.set_h1_9")
load_tb_ternary_feat1_parameter1(table2_1_16[8], "MyIngress.h1_10", "meta.linear2.l32_2", "MyIngress.set_h1_10")
load_tb_ternary_feat1_parameter1(table2_1_16[9], "MyIngress.h1_11", "meta.linear2.l32_3", "MyIngress.set_h1_11")
load_tb_ternary_feat1_parameter1(table2_1_16[10], "MyIngress.h1_12", "meta.linear2.l32_4", "MyIngress.set_h1_12")
load_tb_ternary_feat1_parameter1(table2_1_16[11], "MyIngress.h1_13", "meta.linear2.l32_5", "MyIngress.set_h1_13")
load_tb_ternary_feat1_parameter1(table2_1_16[12], "MyIngress.h1_14", "meta.linear2.l32_6", "MyIngress.set_h1_14")
load_tb_ternary_feat1_parameter1(table2_1_16[13], "MyIngress.h1_15", "meta.linear2.l32_7", "MyIngress.set_h1_15")
load_tb_ternary_feat1_parameter1(table2_1_16[14], "MyIngress.h1_16", "meta.linear2.l32_8", "MyIngress.set_h1_16")

load_tb_exact_feat3_parameter8(table2_3[0], "MyIngress.h1_2_4", "meta.temp.l4_1", "meta.temp.l4_2", "meta.temp.l4_3", "MyIngress.set_h1_2_4")
load_tb_exact_feat3_parameter8(table2_3[1], "MyIngress.h1_5_7", "meta.temp.l4_4", "meta.temp.l4_5", "meta.temp.l4_6", "MyIngress.set_h1_5_7")
load_tb_exact_feat3_parameter8(table2_3[2], "MyIngress.h1_8_10", "meta.temp.l4_7", "meta.temp.l4_8", "meta.temp.l4_9", "MyIngress.set_h1_8_10")
load_tb_exact_feat3_parameter8(table2_3[3], "MyIngress.h1_11_13", "meta.temp.l4_10", "meta.temp.l4_11", "meta.temp.l4_12", "MyIngress.set_h1_11_13")
load_tb_exact_feat3_parameter8(table2_3[4], "MyIngress.h1_14_16", "meta.temp.l4_13", "meta.temp.l4_14", "meta.temp.l4_15", "MyIngress.set_h1_14_16")

load_tb_ternary_feat1_parameter6(table3[0], "MyIngress.h2_1", "meta.linear3.l32_1", "MyIngress.set_h2_1")
load_tb_ternary_feat1_parameter6(table3[1], "MyIngress.h2_2", "meta.linear3.l32_2", "MyIngress.set_h2_2")
load_tb_ternary_feat1_parameter6(table3[2], "MyIngress.h2_3", "meta.linear3.l32_3", "MyIngress.set_h2_3")
load_tb_ternary_feat1_parameter6(table3[3], "MyIngress.h2_4", "meta.linear3.l32_4", "MyIngress.set_h2_4")
load_tb_ternary_feat1_parameter6(table3[4], "MyIngress.h2_5", "meta.linear3.l32_5", "MyIngress.set_h2_5")
load_tb_ternary_feat1_parameter6(table3[5], "MyIngress.h2_6", "meta.linear3.l32_6", "MyIngress.set_h2_6")
load_tb_ternary_feat1_parameter6(table3[6], "MyIngress.h2_7", "meta.linear3.l32_7", "MyIngress.set_h2_7")
load_tb_ternary_feat1_parameter6(table3[7], "MyIngress.h2_8", "meta.linear3.l32_8", "MyIngress.set_h2_8")
