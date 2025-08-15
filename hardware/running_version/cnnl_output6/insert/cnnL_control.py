import sys,pickle                                                             
import signal
import numpy as np
import os,time
sys.path.append(os.path.expandvars('$SDE/install/lib/python3.6/site-packages/tofino/'))
sys.path.append(os.path.expandvars('/home/tangoic/oldz/xf/p4/p4/module'))
from bfrt_grpc import client
GRPC_CLIENT=client.ClientInterface(grpc_addr="localhost:50052", client_id=0, device_id=0)
bfrt_info=GRPC_CLIENT.bfrt_info_get(p4_name=None)
GRPC_CLIENT.bind_pipeline_config(p4_name=bfrt_info.p4_name)
target = client.Target()
#导入模型数据
with open("/home/tangoic/oldz/xf/p4/cnnl_test_0811/insert/cnnl-ISCXVPN.pkl","rb") as f:
    [table01, table02, table03, table04, table2, table11, table22, table23] = pickle.load(f)
#load_tb_Feat(table_data 写入table的数据,table_name 表名,feat_name 表的属性名,action_name 执行的函数)
def load_tb_feat2_parameter1(table_data, table_name, feat_name1, feat_name2, action_name):
    max_table_size = 4096
    print("load_tb--{}.......".format(table_name))
    tcam_table = bfrt_info.table_get(table_name)
    KeyTuple_list=[] 
    DataTuple_List=[] 
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]; print("ERROR! ")
    for x in range(len(table_data)):
        KeyTuple_list.append(tcam_table.make_key([
                                                client.KeyTuple('$MATCH_PRIORITY', table_data[x][0]),
                                                client.KeyTuple(feat_name1, int(table_data[x][1]), int(table_data[x][3])),
                                                client.KeyTuple(feat_name2, int(table_data[x][2]), int(table_data[x][4]))
                                                ]))
        DataTuple_List.append(tcam_table.make_data([client.DataTuple('in01', table_data[x][-1])],action_name))
    tcam_table.entry_add(target, KeyTuple_list, DataTuple_List)
    print("load_to_table--{} completed!".format(table_name))
def load_tb_feat3_parameter16(table_data, table_name, feat_name1, feat_name2, feat_name3, action_name):
    max_table_size = 4096
    print("load_tb--{}.......".format(table_name))
    sram_table = bfrt_info.table_get(table_name)
    KeyTuple_list=[] 
    DataTuple_List=[] 
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]
    for x in range(len(table_data)):
        KeyTuple_list.append(sram_table.make_key([
                                                client.KeyTuple(feat_name1, int(table_data[x][0])),
                                                client.KeyTuple(feat_name2, int(table_data[x][1])),
                                                client.KeyTuple(feat_name3, int(table_data[x][2]))
                                                ]))
        DataTuple_List.append(sram_table.make_data([client.DataTuple('in01', table_data[x][3][0]),
                                                    client.DataTuple('in02', table_data[x][3][1]),
                                                    client.DataTuple('in03', table_data[x][4][0]),
                                                    client.DataTuple('in04', table_data[x][4][1]),
                                                    client.DataTuple('in05', table_data[x][5][0]),
                                                    client.DataTuple('in06', table_data[x][5][1]),
                                                    client.DataTuple('in07', table_data[x][6][0]),
                                                    client.DataTuple('in08', table_data[x][6][1]),
                                                    client.DataTuple('in09', table_data[x][7][0]),
                                                    client.DataTuple('in10', table_data[x][7][1]),
                                                    client.DataTuple('in11', table_data[x][8][0]),
                                                    client.DataTuple('in12', table_data[x][8][1]),
                                                    client.DataTuple('in13', table_data[x][9][0]),
                                                    client.DataTuple('in14', table_data[x][9][1]),
                                                    client.DataTuple('in15', table_data[x][10][0]),
                                                    client.DataTuple('in16', table_data[x][10][1])],action_name))
    sram_table.entry_add(target, KeyTuple_list, DataTuple_List)
    print("load_to_table--{} completed!".format(table_name))
def load_tb_feat3_parameter2(table_data, table_name, feat_name1, feat_name2, feat_name3, action_name):
    max_table_size = 4096
    print("load_tb--{}.......".format(table_name))
    sram_table = bfrt_info.table_get(table_name)
    KeyTuple_list=[] 
    DataTuple_List=[] 
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]
    for x in range(len(table_data)):
        KeyTuple_list.append(sram_table.make_key([
                                                client.KeyTuple(feat_name1, int(table_data[x][0])),
                                                client.KeyTuple(feat_name2, int(table_data[x][1])),
                                                client.KeyTuple(feat_name3, int(table_data[x][2]))
                                                ]))
        DataTuple_List.append(sram_table.make_data([client.DataTuple('in01', table_data[x][3][0]),
                                                    client.DataTuple('in02', table_data[x][3][1])],action_name))
    sram_table.entry_add(target, KeyTuple_list, DataTuple_List)
    print("load_to_table--{} completed!".format(table_name))
def load_tb_feat3_parameter3(table_data, table_name, feat_name1, feat_name2, feat_name3, action_name):
    max_table_size = 4096
    print("load_tb--{}.......".format(table_name))
    sram_table = bfrt_info.table_get(table_name)
    KeyTuple_list=[] 
    DataTuple_List=[] 
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]; print("ERROR! ")
    for x in range(len(table_data)):
        KeyTuple_list.append(sram_table.make_key([
                                                client.KeyTuple(feat_name1, int(table_data[x][0])),
                                                client.KeyTuple(feat_name2, int(table_data[x][1])),
                                                client.KeyTuple(feat_name3, int(table_data[x][2]))
                                                ]))
        DataTuple_List.append(sram_table.make_data([client.DataTuple('in01', table_data[x][-1][0]),
                                                    client.DataTuple('in02', table_data[x][-1][1]),
                                                    client.DataTuple('in03', table_data[x][-1][2]),
                                                    client.DataTuple('in04', table_data[x][-1][3]),
                                                    client.DataTuple('in05', table_data[x][-1][4]),
                                                    client.DataTuple('in06', table_data[x][-1][5])
                                                    ],action_name))
    sram_table.entry_add(target, KeyTuple_list, DataTuple_List)
    print("load_to_table--{} completed!".format(table_name))
def load_tb_feat2_parameter3(table_data, table_name, feat_name1, feat_name2, action_name):
    max_table_size = 4096
    print("load_tb--{}.......".format(table_name))
    sram_table = bfrt_info.table_get(table_name)
    KeyTuple_list=[] 
    DataTuple_List=[] 
    if len(table_data) > max_table_size:
        table_data = table_data[:max_table_size]
    for x in range(len(table_data)):
        KeyTuple_list.append(sram_table.make_key([
                                                client.KeyTuple(feat_name1, int(table_data[x][0])),
                                                client.KeyTuple(feat_name2, int(table_data[x][1]))
                                                ]))
        DataTuple_List.append(sram_table.make_data([client.DataTuple('in01', table_data[x][-1][0]),
                                                    client.DataTuple('in02', table_data[x][-1][1]),
                                                    client.DataTuple('in03', table_data[x][-1][2]),
                                                    client.DataTuple('in04', table_data[x][-1][3]),
                                                    client.DataTuple('in05', table_data[x][-1][4]),
                                                    client.DataTuple('in06', table_data[x][-1][5])
                                                    ],action_name))
    sram_table.entry_add(target, KeyTuple_list, DataTuple_List)
    print("load_to_table--{} completed!".format(table_name))
for i in range(0, 10):
    load_tb_feat2_parameter1(table01[i], "SwitchIngress.p1_{}".format(i+1), "ig_md.b_{}.l8_1".format(i+1), "ig_md.b_{}.l8_2".format(i+1), "SwitchIngress.set_p1_{}".format(i+1))
for i in range(10, 27):
    load_tb_feat2_parameter1(table01[i], "SwitchIngress.p1_{}".format(i+1), "hdr.feature.b_{}.l8_1".format(i+1), "hdr.feature.b_{}.l8_2".format(i+1), "SwitchIngress.set_p1_{}".format(i+1))
for i in range(27, 28):
    load_tb_feat2_parameter1(table02[0], "SwitchIngress.p1_{}".format(i+1), "hdr.feature.b_{}.l8_1".format(i+1), "hdr.ipv4.total_len", "SwitchIngress.set_p1_{}".format(i+1))
for i in range(28, 29):
    load_tb_feat2_parameter1(table03[0], "SwitchIngress.p1_{}".format(i+1), "ig_md.interval", "hdr.ipv4.diffserv", "SwitchIngress.set_p1_{}".format(i+1))
for i in range(29, 30):
    load_tb_feat2_parameter1(table04[0], "SwitchIngress.p1_{}".format(i+1), "hdr.ipv4.ihl", "ig_md.data_offset", "SwitchIngress.set_p1_{}".format(i+1))

for i in range(10): #nospace
    try :
        load_tb_feat3_parameter2(table11[i], "SwitchIngress.p2_{}".format(i+1), "ig_md.temp.l4_{}".format(i*3+1), "ig_md.temp.l4_{}".format(i*3+2), "ig_md.temp.l4_{}".format(i*3+3), "SwitchIngress.set_p2_{}".format(i+1))
    except : 
        pass
for i in range(1):
    load_tb_feat2_parameter1(table2[i], "SwitchIngress.p3_{}".format(i+1), "ig_md.s_1.l16_{}".format(i*2+1),"ig_md.s_1.l16_{}".format(i*2+2), "SwitchIngress.set_p3_{}".format(i+1))
for i in range(2): #nospace
    try :
        load_tb_feat3_parameter3(table22[i], "SwitchIngress.x4_{}".format(i+1), "ig_md.temp.l4_{}".format(i*3+1), "ig_md.temp.l4_{}".format(i*3+2), "ig_md.temp.l4_{}".format(i*3+3), "SwitchIngress.set_x4_{}".format(i+1))
    except :
        pass
for i in range(1):
    load_tb_feat2_parameter3(table23[i], "SwitchIngress.x4_{}".format(i+3), "ig_md.temp.l4_{}".format(i*3+7), "ig_md.temp.l4_{}".format(i*3+8), "SwitchIngress.set_x4_{}".format(i+3))

