/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>
#include "headers.p4"
#include "util.p4"
#include "parsers.p4"

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) 
{
    @symmetric("hdr.ipv4.src_addr", "hdr.ipv4.dst_addr")
    @symmetric("ig_md.src_port", "ig_md.dst_port")
    Hash<bit<32>>(HashAlgorithm_t.CRC32) my_symmetric_hash;

    action noaction(){}
    action set_p1_1(bit<4> in01) {
        ig_md.temp.l4_1 = in01;
    }
    action set_p1_2(bit<4> in01) {
        ig_md.temp.l4_2 = in01;
    }
    action set_p1_3(bit<4> in01) {
        ig_md.temp.l4_3 = in01;
    }
    action set_p1_4(bit<4> in01) {
        ig_md.temp.l4_4 = in01;
    }
    action set_p1_5(bit<4> in01) {
        ig_md.temp.l4_5 = in01;
    }
    action set_p1_6(bit<4> in01) {
        ig_md.temp.l4_6 = in01;
    }
    action set_p1_7(bit<4> in01) {
        ig_md.temp.l4_7 = in01;
    }
    action set_p1_8(bit<4> in01) {
        ig_md.temp.l4_8 = in01;
    }
    action set_p1_9(bit<4> in01) {
        ig_md.temp.l4_9 = in01;
    }
    action set_p1_10(bit<4> in01) {
        ig_md.temp.l4_10 = in01;
    }
    action set_p1_11(bit<4> in01) {
        ig_md.temp.l4_11 = in01;
    }
    action set_p1_12(bit<4> in01) {
        ig_md.temp.l4_12 = in01;
    }
    action set_p1_13(bit<4> in01) {
        ig_md.temp.l4_13 = in01;
    }
    action set_p1_14(bit<4> in01) {
        ig_md.temp.l4_14 = in01;
    }
    action set_p1_15(bit<4> in01) {
        ig_md.temp.l4_15 = in01;
    }
    action set_p1_16(bit<4> in01) {
        ig_md.temp.l4_16 = in01;
    }
    action set_p1_17(bit<4> in01) {
        ig_md.temp.l4_17 = in01;
    }
    action set_p1_18(bit<4> in01) {
        ig_md.temp.l4_18 = in01;
    }
    action set_p1_19(bit<4> in01) {
        ig_md.temp.l4_19 = in01;
    }
    action set_p1_20(bit<4> in01) {
        ig_md.temp.l4_20 = in01;
    }
    action set_p1_21(bit<4> in01) {
        ig_md.temp.l4_21 = in01;
    }
    action set_p1_22(bit<4> in01) {
        ig_md.temp.l4_22 = in01;
    }
    action set_p1_23(bit<4> in01) {
        ig_md.temp.l4_23 = in01;
    }
    action set_p1_24(bit<4> in01) {
        ig_md.temp.l4_24 = in01;
    }
    action set_p1_25(bit<4> in01) {
        ig_md.temp.l4_25 = in01;
    }
    action set_p1_26(bit<4> in01) {
        ig_md.temp.l4_26 = in01;
    }
    action set_p1_27(bit<4> in01) {
        ig_md.temp.l4_27 = in01;
    }
    action set_p1_28(bit<4> in01) {
        ig_md.temp.l4_28 = in01;
    }
    action set_p1_29(bit<4> in01) {
        ig_md.temp.l4_29 = in01;
    }
    action set_p1_30(bit<4> in01) {
        ig_md.temp.l4_30 = in01;
    }
    table p1_1 {
        key = {
            ig_md.b_1.l8_1: ternary;
            ig_md.b_1.l8_2: ternary;
        }
        actions = {
            set_p1_1;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_2 {
        key = {
            ig_md.b_2.l8_1: ternary;
            ig_md.b_2.l8_2: ternary;
        }
        actions = {
            set_p1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_3 {
        key = {
            ig_md.b_3.l8_1: ternary;
            ig_md.b_3.l8_2: ternary;
        }
        actions = {
            set_p1_3;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_4 {
        key = {
            ig_md.b_4.l8_1: ternary;
            ig_md.b_4.l8_2: ternary;
        }
        actions = {
            set_p1_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_5 {
        key = {
            ig_md.b_5.l8_1: ternary;
            ig_md.b_5.l8_2: ternary;
        }
        actions = {
            set_p1_5;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_6 {
        key = {
            ig_md.b_6.l8_1: ternary;
            ig_md.b_6.l8_2: ternary;
        }
        actions = {
            set_p1_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_7 {
        key = {
            ig_md.b_7.l8_1: ternary;
            ig_md.b_7.l8_2: ternary;
        }
        actions = {
            set_p1_7;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_8 {
        key = {
            ig_md.b_8.l8_1: ternary;
            ig_md.b_8.l8_2: ternary;
        }
        actions = {
            set_p1_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_9 {
        key = {
            ig_md.b_9.l8_1: ternary;
            ig_md.b_9.l8_2: ternary;
        }
        actions = {
            set_p1_9;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_10 {
        key = {
            ig_md.b_10.l8_1: ternary;
            ig_md.b_10.l8_2: ternary;
        }
        actions = {
            set_p1_10;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_11 {
        key = {
            hdr.feature.b_11.l8_1: ternary;
            hdr.feature.b_11.l8_2: ternary;
        }
        actions = {
            set_p1_11;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_12 {
        key = {
            hdr.feature.b_12.l8_1: ternary;
            hdr.feature.b_12.l8_2: ternary;
        }
        actions = {
            set_p1_12;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_13 {
        key = {
            hdr.feature.b_13.l8_1: ternary;
            hdr.feature.b_13.l8_2: ternary;
        }
        actions = {
            set_p1_13;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_14 {
        key = {
            hdr.feature.b_14.l8_1: ternary;
            hdr.feature.b_14.l8_2: ternary;
        }
        actions = {
            set_p1_14;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_15 {
        key = {
            hdr.feature.b_15.l8_1: ternary;
            hdr.feature.b_15.l8_2: ternary;
        }
        actions = {
            set_p1_15;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_16 {
        key = {
            hdr.feature.b_16.l8_1: ternary;
            hdr.feature.b_16.l8_2: ternary;
        }
        actions = {
            set_p1_16;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_17 {
        key = {
            hdr.feature.b_17.l8_1: ternary;
            hdr.feature.b_17.l8_2: ternary;
        }
        actions = {
            set_p1_17;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_18 {
        key = {
            hdr.feature.b_18.l8_1: ternary;
            hdr.feature.b_18.l8_2: ternary;
        }
        actions = {
            set_p1_18;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_19 {
        key = {
            hdr.feature.b_19.l8_1: ternary;
            hdr.feature.b_19.l8_2: ternary;
        }
        actions = {
            set_p1_19;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_20 {
        key = {
            hdr.feature.b_20.l8_1: ternary;
            hdr.feature.b_20.l8_2: ternary;
        }
        actions = {
            set_p1_20;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_21 {
        key = {
            hdr.feature.b_21.l8_1: ternary;
            hdr.feature.b_21.l8_2: ternary;
        }
        actions = {
            set_p1_21;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_22 {
        key = {
            hdr.feature.b_22.l8_1: ternary;
            hdr.feature.b_22.l8_2: ternary;
        }
        actions = {
            set_p1_22;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_23 {
        key = {
            hdr.feature.b_23.l8_1: ternary;
            hdr.feature.b_23.l8_2: ternary;
        }
        actions = {
            set_p1_23;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_24 {
        key = {
            hdr.feature.b_24.l8_1: ternary;
            hdr.feature.b_24.l8_2: ternary;
        }
        actions = {
            set_p1_24;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_25 {
        key = {
            hdr.feature.b_25.l8_1: ternary;
            hdr.feature.b_25.l8_2: ternary;
        }
        actions = {
            set_p1_25;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_26 {
        key = {
            hdr.feature.b_26.l8_1: ternary;
            hdr.feature.b_26.l8_2: ternary;
        }
        actions = {
            set_p1_26;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_27 {
        key = {
            hdr.feature.b_27.l8_1: ternary;
            hdr.feature.b_27.l8_2: ternary;
        }
        actions = {
            set_p1_27;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_28 {
        key = {
            hdr.feature.b_28.l8_1: ternary;
            hdr.ipv4.total_len: ternary;
        }
        actions = {
            set_p1_28;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_29 {
        key = {
            ig_md.interval: ternary;
            hdr.ipv4.diffserv: ternary;
        }
        actions = {
            set_p1_29;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }
    table p1_30 {
        key = {
            hdr.ipv4.ihl: ternary;
            ig_md.data_offset: ternary;
        }
        actions = {
            set_p1_30;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }

    action set_p2_1(bit<16> in01, bit<16> in02) {
        ig_md.s_1.l16_1 = in01;
        ig_md.s_1.l16_2 = in02;
    }
    action set_p2_2(bit<16> in01, bit<16> in02) {
        ig_md.s_2.l16_1 = in01;
        ig_md.s_2.l16_2 = in02;
    }
    action set_p2_3(bit<16> in01, bit<16> in02) {
        ig_md.s_3.l16_1 = in01;
        ig_md.s_3.l16_2 = in02;
    }
    action set_p2_4(bit<16> in01, bit<16> in02) {
        ig_md.s_4.l16_1 = in01;
        ig_md.s_4.l16_2 = in02;
    }
    action set_p2_5(bit<16> in01, bit<16> in02) {
        ig_md.s_1.l16_1 = in01 + ig_md.s_1.l16_1;
        ig_md.s_1.l16_2 = in02 + ig_md.s_1.l16_2;
    }
    action set_p2_6(bit<16> in01, bit<16> in02) {
        ig_md.s_2.l16_1 = in01 + ig_md.s_2.l16_1;
        ig_md.s_2.l16_2 = in02 + ig_md.s_2.l16_2;
    }
    action set_p2_7(bit<16> in01, bit<16> in02) {
        ig_md.s_3.l16_1 = in01 + ig_md.s_3.l16_1;
        ig_md.s_3.l16_2 = in02 + ig_md.s_3.l16_2;
    }
    action set_p2_8(bit<16> in01, bit<16> in02) {
        ig_md.s_4.l16_1 = in01 + ig_md.s_4.l16_1;
        ig_md.s_4.l16_2 = in02 + ig_md.s_4.l16_2;
    }
    action set_p2_9(bit<16> in01, bit<16> in02) {
        ig_md.s_1.l16_1 = in01 + ig_md.s_1.l16_1;
        ig_md.s_1.l16_2 = in02 + ig_md.s_1.l16_2;
    }
    action set_p2_10(bit<16> in01, bit<16> in02) {
        ig_md.s_2.l16_1 = in01 + ig_md.s_2.l16_1;
        ig_md.s_2.l16_2 = in02 + ig_md.s_2.l16_2;
    }

    table p2_1 {
        key = {
            ig_md.temp.l4_1: exact;
            ig_md.temp.l4_2: exact;
            ig_md.temp.l4_3: exact;
        }
        actions = {
            set_p2_1;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table p2_2 {
        key = {
            ig_md.temp.l4_4: exact;
            ig_md.temp.l4_5: exact;
            ig_md.temp.l4_6: exact;
        }
        actions = {
            set_p2_2;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table p2_3 {
        key = {
            ig_md.temp.l4_7: exact;
            ig_md.temp.l4_8: exact;
            ig_md.temp.l4_9: exact;
        }
        actions = {
            set_p2_3;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table p2_4 {
        key = {
            ig_md.temp.l4_10: exact;
            ig_md.temp.l4_11: exact;
            ig_md.temp.l4_12: exact;
        }
        actions = {
            set_p2_4;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table p2_5 {
        key = {
            ig_md.temp.l4_13: exact;
            ig_md.temp.l4_14: exact;
            ig_md.temp.l4_15: exact;
        }
        actions = {
            set_p2_5;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table p2_6 {
        key = {
            ig_md.temp.l4_16: exact;
            ig_md.temp.l4_17: exact;
            ig_md.temp.l4_18: exact;
        }
        actions = {
            set_p2_6;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table p2_7 {
        key = {
            ig_md.temp.l4_19: exact;
            ig_md.temp.l4_20: exact;
            ig_md.temp.l4_21: exact;
        }
        actions = {
            set_p2_7;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table p2_8 {
        key = {
            ig_md.temp.l4_22: exact;
            ig_md.temp.l4_23: exact;
            ig_md.temp.l4_24: exact;
        }
        actions = {
            set_p2_8;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table p2_9 {
        key = {
            ig_md.temp.l4_25: exact;
            ig_md.temp.l4_26: exact;
            ig_md.temp.l4_27: exact;
        }
        actions = {
            set_p2_9;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table p2_10 {
        key = {
            ig_md.temp.l4_28: exact;
            ig_md.temp.l4_29: exact;
            ig_md.temp.l4_30: exact;
        }
        actions = {
            set_p2_10;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }

    action set_p3_1(bit<4> in01) {
        ig_md.temp.l4_1 = in01;
    }
    table p3_1 {
        key = {
            ig_md.s_1.l16_1: ternary;
            ig_md.s_1.l16_2: ternary;
        }
        actions = {
            set_p3_1;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

    @stage(0)
    action cal_flow_hash(){
        ig_md.flow_hash = my_symmetric_hash.get({hdr.ipv4.src_addr,hdr.ipv4.dst_addr,ig_md.src_port,ig_md.dst_port,hdr.ipv4.protocol});
        ig_md.flow_index = (bit<Register_Index_Size>)ig_md.flow_hash;
    }
    //对流id进行hash
    Register<bit<32>, bit<Register_Index_Size>>(Register_Table_Size) Register_full_flow_hash; //hash as flow id
    RegisterAction<bit<32>, bit<Register_Index_Size>, bit<32>>(Register_full_flow_hash) Update_Register_full_flow_hash = { 
        void apply(inout bit<32> value, out bit<32> read_value){
            read_value = value;
            value = ig_md.flow_hash;
        }
    };
    @stage(1)
    action Update_full_flow_hash(){
        ig_md.flow_hash1 = Update_Register_full_flow_hash.execute(ig_md.flow_index);
    }
    //统计总包数:返回上一次的总包数
    @stage(2)
    Register<bit<32>, bit<Register_Index_Size>>(Register_Table_Size) Register_pkt_count;
    RegisterAction<bit<32>, bit<Register_Index_Size>, bit<32>>(Register_pkt_count) Update_Register_pkt_count = { 
        void apply(inout bit<32> value, out bit<32> read_value){
            value = value + 1;
            read_value = value;
        }
    };
    action Update_pkt_count() {
        ig_md.pkt_count = Update_Register_pkt_count.execute(ig_md.flow_index);
    }
    RegisterAction<bit<32>, bit<Register_Index_Size>, bit<32>>(Register_pkt_count) Init_Register_pkt_count = { 
        void apply(inout bit<32> value, out bit<32> read_value){
            read_value = 1;
            value = 1;
        }
    };
    action Init_pkt_count(){
        ig_md.pkt_count = Init_Register_pkt_count.execute(ig_md.flow_index);
    }
    //统计总包数mod 8:即最新的数据包存放的序号
    @stage(2)
    Register<bit<size_of_pkt_count_mod_8>, bit<Register_Index_Size>>(Register_Table_Size) Register_pkt_count_mod_8;
    RegisterAction<bit<size_of_pkt_count_mod_8>, bit<Register_Index_Size>, bit<size_of_pkt_count_mod_8>>(Register_pkt_count_mod_8) Update_Register_pkt_count_mod_8 = { 
        void apply(inout bit<size_of_pkt_count_mod_8> value, out bit<size_of_pkt_count_mod_8> read_value){
            if (value == 7) {
                value = 0;
            }else {
                value = value + 1;
            }
            read_value = value;
        }
    };
    action Update_pkt_count_mod_8() {
        ig_md.pkt_count_mod_8 = Update_Register_pkt_count_mod_8.execute(ig_md.flow_index);
    }
    RegisterAction<bit<size_of_pkt_count_mod_8>, bit<Register_Index_Size>, bit<size_of_pkt_count_mod_8>>(Register_pkt_count_mod_8) Init_Register_pkt_count_mod_8 = { 
        void apply(inout bit<size_of_pkt_count_mod_8> value, out bit<size_of_pkt_count_mod_8> read_value){
            read_value = 0;
            value = 0;
        }
    };
    action Init_pkt_count_mod_8(){
        ig_md.pkt_count_mod_8 = Init_Register_pkt_count_mod_8.execute(ig_md.flow_index);
    }

    Register<bit<16>, bit<Register_Index_Size>>(Register_Table_Size) feature_ipd;
    RegisterAction<bit<16>, bit<Register_Index_Size>, bit<16>>(feature_ipd) access_timestamp_1 = {
        void apply(inout bit<16> value, out bit<16> read_value) {
            read_value = value;
            value =  ig_md.timestamp;
        }
    };
    Register<bit<Width>, bit<Register_Index_Size>>(Register_Table_Size) get_embed_1;
    RegisterAction<bit<Width>, bit<Register_Index_Size>, bit<Width>>(get_embed_1) access_embed_1 = {
        void apply(inout bit<Width> value, out bit<Width> read_value) {
            read_value = value;
            if( ig_md.pkt_count_mod_8 == 0) {
                value = ig_md.reg.l8_5;
            }
            else if( ig_md.pkt_count_mod_8 == 1) {
                value = value | ig_md.reg.l8_5;
            }
        }
    };
    Register<bit<Width>, bit<Register_Index_Size>>(Register_Table_Size) get_embed_2;
    RegisterAction<bit<Width>, bit<Register_Index_Size>, bit<Width>>(get_embed_2) access_embed_2 = {
        void apply(inout bit<Width> value, out bit<Width> read_value) {
            read_value = value;
            if( ig_md.pkt_count_mod_8 == 2) {
                value = ig_md.reg.l8_5;
            }
            else if( ig_md.pkt_count_mod_8 == 3) {
                value = value | ig_md.reg.l8_5;
            }
        }
    };
    Register<bit<Width>, bit<Register_Index_Size>>(Register_Table_Size) get_embed_3;
    RegisterAction<bit<Width>, bit<Register_Index_Size>, bit<Width>>(get_embed_3) access_embed_3 = {
        void apply(inout bit<Width> value, out bit<Width> read_value) {
            read_value = value;
            if( ig_md.pkt_count_mod_8 == 4) {
                value = ig_md.reg.l8_5;
            }
            else if( ig_md.pkt_count_mod_8 == 5) {
                value = value | ig_md.reg.l8_5;
            }
        }
    };
    Register<bit<Width>, bit<Register_Index_Size>>(Register_Table_Size) get_embed_4;
    RegisterAction<bit<Width>, bit<Register_Index_Size>, bit<Width>>(get_embed_4) access_embed_4 = {
        void apply(inout bit<Width> value, out bit<Width> read_value) {
            read_value = value;
            if( ig_md.pkt_count_mod_8 == 6) {
                value = ig_md.reg.l8_5;
            }
            else if( ig_md.pkt_count_mod_8 == 7) {
                value = value | ig_md.reg.l8_5;
            }
        }
    };
    action access_timestamp() {ig_md.timestamp = access_timestamp_1.execute(ig_md.flow_index);}
    action access_1() {ig_md.reg1.l8_1 = access_embed_1.execute(ig_md.flow_index);}
    action access_2() {ig_md.reg1.l8_2 = access_embed_2.execute(ig_md.flow_index);}
    action access_3() {ig_md.reg1.l8_3 = access_embed_3.execute(ig_md.flow_index);}
    action access_4() {ig_md.reg1.l8_4 = access_embed_4.execute(ig_md.flow_index);}

       
    action act_swap0() {
        ig_md.reg.l8_1[7:4] = ig_md.reg1.l8_2[7:4];
        ig_md.reg.l8_1[3:0] = ig_md.reg1.l8_2[3:0];
        ig_md.reg.l8_2[7:4] = ig_md.reg1.l8_3[7:4];
        ig_md.reg.l8_2[3:0] = ig_md.reg1.l8_3[3:0];
        ig_md.reg.l8_3[7:4] = ig_md.reg1.l8_4[7:4];
        ig_md.reg.l8_3[3:0] = ig_md.reg1.l8_4[3:0];
        ig_md.reg.l8_4[7:4] = ig_md.reg1.l8_1[7:4];
    }

    action act_swap1() {
        ig_md.reg.l8_1[7:4] = ig_md.reg1.l8_2[3:0];
        ig_md.reg.l8_1[3:0] = ig_md.reg1.l8_3[7:4];
        ig_md.reg.l8_2[7:4] = ig_md.reg1.l8_3[3:0];
        ig_md.reg.l8_2[3:0] = ig_md.reg1.l8_4[7:4];
        ig_md.reg.l8_3[7:4] = ig_md.reg1.l8_4[3:0];
        ig_md.reg.l8_3[3:0] = ig_md.reg1.l8_1[7:4];
        ig_md.reg.l8_4[7:4] = ig_md.reg1.l8_1[3:0];
    }

    action act_swap2() {
        ig_md.reg.l8_1[7:4] = ig_md.reg1.l8_3[7:4];
        ig_md.reg.l8_1[3:0] = ig_md.reg1.l8_3[3:0];
        ig_md.reg.l8_2[7:4] = ig_md.reg1.l8_4[7:4];
        ig_md.reg.l8_2[3:0] = ig_md.reg1.l8_4[3:0];
        ig_md.reg.l8_3[7:4] = ig_md.reg1.l8_1[7:4];
        ig_md.reg.l8_3[3:0] = ig_md.reg1.l8_1[3:0];
        ig_md.reg.l8_4[7:4] = ig_md.reg1.l8_2[7:4];
    }

    action act_swap3() {
        ig_md.reg.l8_1[7:4] = ig_md.reg1.l8_3[3:0];
        ig_md.reg.l8_1[3:0] = ig_md.reg1.l8_4[7:4];
        ig_md.reg.l8_2[7:4] = ig_md.reg1.l8_4[3:0];
        ig_md.reg.l8_2[3:0] = ig_md.reg1.l8_1[7:4];
        ig_md.reg.l8_3[7:4] = ig_md.reg1.l8_1[3:0];
        ig_md.reg.l8_3[3:0] = ig_md.reg1.l8_2[7:4];
        ig_md.reg.l8_4[7:4] = ig_md.reg1.l8_2[3:0];
    }
    action act_swap4() {
        ig_md.reg.l8_1[7:4] = ig_md.reg1.l8_4[7:4];
        ig_md.reg.l8_1[3:0] = ig_md.reg1.l8_4[3:0];
        ig_md.reg.l8_2[7:4] = ig_md.reg1.l8_1[7:4];
        ig_md.reg.l8_2[3:0] = ig_md.reg1.l8_1[3:0];
        ig_md.reg.l8_3[7:4] = ig_md.reg1.l8_2[7:4];
        ig_md.reg.l8_3[3:0] = ig_md.reg1.l8_2[3:0];
        ig_md.reg.l8_4[7:4] = ig_md.reg1.l8_3[7:4];
    }
    action act_swap5() {
        ig_md.reg.l8_1[7:4] = ig_md.reg1.l8_4[3:0];
        ig_md.reg.l8_1[3:0] = ig_md.reg1.l8_1[7:4];
        ig_md.reg.l8_2[7:4] = ig_md.reg1.l8_1[3:0];
        ig_md.reg.l8_2[3:0] = ig_md.reg1.l8_2[7:4];
        ig_md.reg.l8_3[7:4] = ig_md.reg1.l8_2[3:0];
        ig_md.reg.l8_3[3:0] = ig_md.reg1.l8_3[7:4];
        ig_md.reg.l8_4[7:4] = ig_md.reg1.l8_3[3:0];
    }
    action act_swap6() {
        ig_md.reg.l8_1[7:4] = ig_md.reg1.l8_1[7:4];
        ig_md.reg.l8_1[3:0] = ig_md.reg1.l8_1[3:0];
        ig_md.reg.l8_2[7:4] = ig_md.reg1.l8_2[7:4];
        ig_md.reg.l8_2[3:0] = ig_md.reg1.l8_2[3:0];
        ig_md.reg.l8_3[7:4] = ig_md.reg1.l8_3[7:4];
        ig_md.reg.l8_3[3:0] = ig_md.reg1.l8_3[3:0];
        ig_md.reg.l8_4[7:4] = ig_md.reg1.l8_4[7:4];
    }
    action act_swap7() {
        ig_md.reg.l8_1[7:4] = ig_md.reg1.l8_1[3:0];
        ig_md.reg.l8_1[3:0] = ig_md.reg1.l8_2[7:4];
        ig_md.reg.l8_2[7:4] = ig_md.reg1.l8_2[3:0];
        ig_md.reg.l8_2[3:0] = ig_md.reg1.l8_3[7:4];
        ig_md.reg.l8_3[7:4] = ig_md.reg1.l8_3[3:0];
        ig_md.reg.l8_3[3:0] = ig_md.reg1.l8_4[7:4];
        ig_md.reg.l8_4[7:4] = ig_md.reg1.l8_4[3:0];
    }
    table tab_swap {
        size = 8;
        key = { ig_md.pkt_count_mod_8: exact; }
        actions = { act_swap0; act_swap1; act_swap2; act_swap3; act_swap4; act_swap5; act_swap6; act_swap7; }
        const entries = {
            (0):act_swap0(); (1):act_swap1(); (2):act_swap2(); (3):act_swap3(); (4):act_swap4(); (5):act_swap5(); (6):act_swap6(); (7):act_swap7();
        }
        const default_action = act_swap0();
    }

    action set_x4_1(bit<32> in01, bit<32> in02, bit<32> in03) {
        hdr.output.linear1.l32_1 = in01;
        hdr.output.linear1.l32_2 = in02;
        hdr.output.linear1.l32_3 = in03;
    }
    action set_x4_2(bit<32> in01, bit<32> in02, bit<32> in03) {
        ig_md.linear1.l32_1 = in01;
        ig_md.linear1.l32_2 = in02;
        ig_md.linear1.l32_3 = in03;
    }
    action set_x4_3(bit<32> in01, bit<32> in02, bit<32> in03) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
    }
    table x4_1 {
        key = {
            ig_md.temp.l4_1: exact;
            ig_md.temp.l4_2: exact;
            ig_md.temp.l4_3: exact;
        }
        actions = {
            set_x4_1;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table x4_2 {
        key = {
            ig_md.temp.l4_4: exact;
            ig_md.temp.l4_5: exact;
            ig_md.temp.l4_6: exact;
        }
        actions = {
            set_x4_2;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table x4_3 {
        key = {
            ig_md.temp.l4_7: exact;
            ig_md.temp.l4_8: exact;
        }
        actions = {
            set_x4_3;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }

    action copy_data() {
        ig_md.b_1.l8_1 = hdr.tcp.src_port[15:8];
        ig_md.b_1.l8_2 = hdr.tcp.src_port[7:0];

        ig_md.b_2.l8_1 = hdr.tcp.dst_port[15:8];
        ig_md.b_2.l8_2 = hdr.tcp.dst_port[7:0];

        ig_md.b_3.l8_1 = hdr.tcp.seq_no[31:24];
        ig_md.b_3.l8_2 = hdr.tcp.seq_no[23:16];
        ig_md.b_4.l8_1 = hdr.tcp.seq_no[15:8];
        ig_md.b_4.l8_2 = hdr.tcp.seq_no[7:0];

        ig_md.b_5.l8_1 = hdr.tcp.ack_no[31:24];
        ig_md.b_5.l8_2 = hdr.tcp.ack_no[23:16];
        ig_md.b_6.l8_1 = hdr.tcp.ack_no[15:8];
        ig_md.b_6.l8_2 = hdr.tcp.ack_no[7:0];

        ig_md.b_7.l8_1 = ((bit<8>)hdr.tcp.data_offset << 4) | (bit<8>)hdr.tcp.res;

        ig_md.b_7.l8_2 = hdr.tcp.flags;

        ig_md.b_8.l8_1 = hdr.tcp.window[15:8];
        ig_md.b_8.l8_2 = hdr.tcp.window[7:0];

        ig_md.b_9.l8_1 = hdr.tcp.checksum[15:8];
        ig_md.b_9.l8_2 = hdr.tcp.checksum[7:0];

        ig_md.b_10.l8_1 = hdr.tcp.urgent_ptr[15:8];
        ig_md.b_10.l8_2 = hdr.tcp.urgent_ptr[7:0];
    }

    apply
    {
        copy_data();
        cal_flow_hash();
        if(hdr.ethernet.ether_type == ETHERTYPE_IPV4)
        {
            if(ig_md.src_port != 68)
            {
                ig_tm_md.ucast_egress_port = 0x120;
            }
        }
        Update_full_flow_hash();
        if(ig_md.flow_hash != ig_md.flow_hash1){
            Init_pkt_count();
            Init_pkt_count_mod_8();
        }
        else{
            Update_pkt_count();
            Update_pkt_count_mod_8();
        }

        if (hdr.tcp.isValid()) {
            ig_md.interval = hdr.ipv4.identification;
            ig_md.data_offset = hdr.tcp.data_offset;
        } else {
            ig_md.interval = hdr.ipv4.identification;
            ig_md.data_offset = 0;
        }

        // 时间间隔计算
        ig_md.timestamp = ig_md.interval;
        ig_md.last_timestamp = access_timestamp_1.execute(ig_md.flow_index);
        if (ig_md.pkt_count == 1) {
            ig_md.interval = 0;
        } else {
            ig_md.interval = ig_md.timestamp - ig_md.last_timestamp;
        }

        p1_1.apply();
        p1_2.apply();
        p1_3.apply();
        p1_4.apply();
        p1_5.apply();
        p1_6.apply();
        p1_7.apply();
        p1_8.apply();
        p1_9.apply();
        p1_10.apply();
        p1_11.apply();
        p1_12.apply();
        p1_13.apply();
        p1_14.apply();
        p1_15.apply();
        p1_16.apply();
        p1_17.apply();
        p1_18.apply();
        p1_19.apply();
        p1_20.apply();
        p1_21.apply();
        p1_22.apply();
        p1_23.apply();
        p1_24.apply();
        p1_25.apply();
        p1_26.apply();
        p1_27.apply();
        p1_28.apply();
        p1_29.apply();
        p1_30.apply();

        p2_1.apply();
        p2_2.apply();
        p2_3.apply();
        p2_4.apply();
        p2_5.apply();
        p2_6.apply();
        p2_7.apply();
        p2_8.apply();
        p2_9.apply();
        p2_10.apply();

        ig_md.s_1.l16_1 = ig_md.s_1.l16_1 + ig_md.s_3.l16_1;
        ig_md.s_1.l16_2 = ig_md.s_1.l16_2 + ig_md.s_3.l16_2;

        ig_md.s_2.l16_1 = ig_md.s_2.l16_1 + ig_md.s_4.l16_1;
        ig_md.s_2.l16_2 = ig_md.s_2.l16_2 + ig_md.s_4.l16_2;

        ig_md.s_1.l16_1 = ig_md.s_1.l16_1 + ig_md.s_2.l16_1;
        ig_md.s_1.l16_2 = ig_md.s_1.l16_2 + ig_md.s_2.l16_2;

        p3_1.apply();

        if (ig_md.pkt_count_mod_8 & 8w1 == 0){
            ig_md.reg.l8_5 = ((bit<8>)ig_md.temp.l4_1) << 4;
        } else {
            ig_md.reg.l8_5 = (bit<8>)ig_md.temp.l4_1;
        }
        access_1();
        access_2();
        access_3();
        access_4();
        tab_swap.apply();
        ig_md.reg.l8_4[3:0] = ig_md.temp.l4_1;

        ig_md.temp.l4_1 = ig_md.reg.l8_1[7:4];
        ig_md.temp.l4_2 = ig_md.reg.l8_1[3:0];
        ig_md.temp.l4_3 = ig_md.reg.l8_2[7:4];
        ig_md.temp.l4_4 = ig_md.reg.l8_2[3:0];
        ig_md.temp.l4_5 = ig_md.reg.l8_3[7:4];
        ig_md.temp.l4_6 = ig_md.reg.l8_3[3:0];
        ig_md.temp.l4_7 = ig_md.reg.l8_4[7:4];
        ig_md.temp.l4_8 = ig_md.reg.l8_4[3:0];

        ig_md.real_pkt_count = ig_md.pkt_count[7:0];

        x4_1.apply();
        x4_2.apply();
        x4_3.apply();

        hdr.output.linear1.l32_1 = ig_md.linear1.l32_1 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = ig_md.linear1.l32_2 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = ig_md.linear1.l32_3 + hdr.output.linear1.l32_3;

        if(ig_md.real_pkt_count<8){ig_dprsr_md.drop_ctl = 1;}

        hdr.output.setValid();
        ig_tm_md.ucast_egress_port = 288;
        ig_tm_md.bypass_egress = 1;
        hdr.ethernet.dst_addr = 0; //for filter
        hdr.tcp.dst_port = 9999;
        //hdr.udp.dst_port = 9999;
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;