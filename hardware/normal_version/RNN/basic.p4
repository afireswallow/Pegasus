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

// ***embed**************************************************************************

    action set_emb_pkl(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        hdr.output.x_8.l16_1 = in01;
        hdr.output.x_8.l16_2 = in02;
        hdr.output.x_8.l16_3 = in03;
        hdr.output.x_8.l16_4 = in04;
    }

    @stage (0)
    table emb_pkl {
        key = {
            hdr.feature.pkl: exact;
        }
        actions = {
            set_emb_pkl;
            noaction;
        }
        size = 4096;
        default_action = noaction();
    }

    action set_emb_ipd(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.x_temp.l16_1 = in01;
        ig_md.x_temp.l16_2 = in02;
        ig_md.x_temp.l16_3 = in03;
        ig_md.x_temp.l16_4 = in04;
    }

    @stage (0)
    table emb_ipd {
        key = {
            hdr.feature.ipd: exact;
        }
        actions = {
            set_emb_ipd;
            noaction;
        }
        size = 4096;
        default_action = noaction();
    }
    action set_x0_1_2(bit<8> in01) {
        hdr.output.t4_8.l8_1 = in01;
        // hdr.output.t_8.l32_1[31:24] = in01;
    }
    action set_x0_3_4(bit<8> in01) {
        hdr.output.t4_8.l8_2 = in01;
        // hdr.output.t_8.l32_1[23:16] = in01;
    }
    action set_x0_5_6(bit<8> in01) {
        hdr.output.t4_8.l8_3 = in01;
        // hdr.output.t_8.l32_1[15:8] = in01;
    }
    action set_x0_7_8(bit<8> in01) {
        // hdr.output.t4_8.l8_4 = in01;
        hdr.output.t_8.l32_1[7:0] = in01;
    }
    
    @stage (2)
    table x0_1_2 {
        key = {
            hdr.output.x_8.l16_1: ternary;
        }
        actions = {
            set_x0_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (2)
    table x0_3_4 {
        key = {
            hdr.output.x_8.l16_2: ternary;
        }
        actions = {
            set_x0_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (2)
    table x0_5_6 {
        key = {
            hdr.output.x_8.l16_3: ternary;
        }
        actions = {
            set_x0_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    //@stage (3)
    table x0_7_8 {
        key = {
            hdr.output.x_8.l16_4: ternary;
        }
        actions = {
            set_x0_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    // //计算流id
    // @stage(0)
    // action cal_flow_hash(){
    //     ig_md.flow_hash = my_symmetric_hash.get({hdr.ipv4.src_addr,hdr.ipv4.dst_addr,ig_md.src_port,ig_md.dst_port,hdr.ipv4.protocol});
    //     ig_md.flow_index = (bit<Register_Index_Size>)ig_md.flow_hash;
    // }
    // //对流id进行hash
    // Register<bit<32>, bit<Register_Index_Size>>(Register_Table_Size) Register_full_flow_hash; //hash as flow id
    // RegisterAction<bit<32>, bit<Register_Index_Size>, bit<32>>(Register_full_flow_hash) Update_Register_full_flow_hash = { 
    //     void apply(inout bit<32> value, out bit<32> read_value){
    //         read_value = value;
    //         value = ig_md.flow_hash;
    //     }
    // };
    // @stage(1)
    // action Update_full_flow_hash(){
    //     ig_md.flow_hash1 = Update_Register_full_flow_hash.execute(ig_md.flow_index);
    // }

    // //统计总包数:返回上一次的总包数
    // @stage(2)
    // Register<bit<32>, bit<Register_Index_Size>>(Register_Table_Size) Register_pkt_count;
    // RegisterAction<bit<32>, bit<Register_Index_Size>, bit<32>>(Register_pkt_count) Update_Register_pkt_count = { 
    //     void apply(inout bit<32> value, out bit<32> read_value){
    //         value = value + 1;
    //         read_value = value;
    //     }
    // };
    // action Update_pkt_count() {
    //     ig_md.pkt_count = Update_Register_pkt_count.execute(ig_md.flow_index);
    // }
    // RegisterAction<bit<32>, bit<Register_Index_Size>, bit<32>>(Register_pkt_count) Init_Register_pkt_count = { 
    //     void apply(inout bit<32> value, out bit<32> read_value){
    //         read_value = 1;
    //         value = 1;
    //     }
    // };
    // action Init_pkt_count(){
    //     ig_md.pkt_count = Init_Register_pkt_count.execute(ig_md.flow_index);
    // }

    // //统计总包数mod 7:即最新的数据包存放的序号
    // @stage(2)
    // Register<bit<size_of_pkt_count_mod_7>, bit<Register_Index_Size>>(Register_Table_Size) Register_pkt_count_mod_7;
    // RegisterAction<bit<size_of_pkt_count_mod_7>, bit<Register_Index_Size>, bit<size_of_pkt_count_mod_7>>(Register_pkt_count_mod_7) Update_Register_pkt_count_mod_7 = { 
    //     void apply(inout bit<size_of_pkt_count_mod_7> value, out bit<size_of_pkt_count_mod_7> read_value){
    //         if (value == 6) {
    //             value = 0;
    //         }else {
    //             value = value + 1;
    //         }
    //         read_value = value;
    //     }
    // };
    // action Update_pkt_count_mod_7() {
    //     ig_md.pkt_count_mod_7 = Update_Register_pkt_count_mod_7.execute(ig_md.flow_index);
    // }
    // RegisterAction<bit<size_of_pkt_count_mod_7>, bit<Register_Index_Size>, bit<size_of_pkt_count_mod_7>>(Register_pkt_count_mod_7) Init_Register_pkt_count_mod_7 = { 
    //     void apply(inout bit<size_of_pkt_count_mod_7> value, out bit<size_of_pkt_count_mod_7> read_value){
    //         read_value = 0;
    //         value = 0;
    //     }
    // };
    // action Init_pkt_count_mod_7(){
    //     ig_md.pkt_count_mod_7 = Init_Register_pkt_count_mod_7.execute(ig_md.flow_index);
    // }

    // // //缓存embed特征
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_0_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_0_1) access_pkt_embed_0_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7 == 0) { value =  hdr.output.t_8.l32_1; }
    //     } 
    // };
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_1_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_1_1) access_pkt_embed_1_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7 == 1) { value =  hdr.output.t_8.l32_1; }
    //     } 
    // };
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_2_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_2_1) access_pkt_embed_2_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7 == 2) { value =  hdr.output.t_8.l32_1; }
    //     } 
    // };
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_3_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_3_1) access_pkt_embed_3_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7 == 3) { value =  hdr.output.t_8.l32_1; }
    //     } 
    // };
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_4_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_4_1) access_pkt_embed_4_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7 == 4) { value =  hdr.output.t_8.l32_1; }
    //     } 
    // };
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_5_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_5_1) access_pkt_embed_5_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7 == 5) { value =  hdr.output.t_8.l32_1; }
    //     } 
    // };
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_6_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_6_1) access_pkt_embed_6_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7 == 6) { value =  hdr.output.t_8.l32_1; }
    //     } 
    // };
    // action access_pkt_embeded_feature_0_1() {
    //         ig_md.t_1.l32_1 = access_pkt_embed_0_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_1_1() {
    //         ig_md.t_2.l32_1 = access_pkt_embed_1_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_2_1() {
    //         ig_md.t_3.l32_1 = access_pkt_embed_2_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_3_1() {
    //         ig_md.t_4.l32_1 = access_pkt_embed_3_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_4_1() {
    //         ig_md.t_5.l32_1 = access_pkt_embed_4_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_5_1() {
    //         ig_md.t_6.l32_1 = access_pkt_embed_5_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_6_1() {
    //         ig_md.t_7.l32_1 = access_pkt_embed_6_1.execute(ig_md.flow_index);
    // }

    
    // action act_swap0() {
    //     hdr.output.s_1.l16_1 = ig_md.t_2.l32_1[31:16];
    //     hdr.output.s_1.l16_2 = ig_md.t_2.l32_1[15:0];
    //     hdr.output.s_2.l16_1 = ig_md.t_3.l32_1[31:16];
    //     hdr.output.s_2.l16_2 = ig_md.t_3.l32_1[15:0];
    //     hdr.output.s_3.l16_1 = ig_md.t_4.l32_1[31:16];
    //     hdr.output.s_3.l16_2 = ig_md.t_4.l32_1[15:0];
    //     hdr.output.s_4.l16_1 = ig_md.t_5.l32_1[31:16];
    //     hdr.output.s_4.l16_2 = ig_md.t_5.l32_1[15:0];
    //     hdr.output.s_5.l16_1 = ig_md.t_6.l32_1[31:16];
    //     hdr.output.s_5.l16_2 = ig_md.t_6.l32_1[15:0];
    //     hdr.output.s_6.l16_1 = ig_md.t_7.l32_1[31:16];
    //     hdr.output.s_6.l16_2 = ig_md.t_7.l32_1[15:0];
    //     hdr.output.s_7.l16_1 = ig_md.t_1.l32_1[31:16];
    //     hdr.output.s_7.l16_2 = ig_md.t_1.l32_1[15:0];
    // }    
    // action act_swap1() {
    //     hdr.output.s_1.l16_1 = ig_md.t_3.l32_1[31:16];
    //     hdr.output.s_1.l16_2 = ig_md.t_3.l32_1[15:0];
    //     hdr.output.s_2.l16_1 = ig_md.t_4.l32_1[31:16];
    //     hdr.output.s_2.l16_2 = ig_md.t_4.l32_1[15:0];
    //     hdr.output.s_3.l16_1 = ig_md.t_5.l32_1[31:16];
    //     hdr.output.s_3.l16_2 = ig_md.t_5.l32_1[15:0];
    //     hdr.output.s_4.l16_1 = ig_md.t_6.l32_1[31:16];
    //     hdr.output.s_4.l16_2 = ig_md.t_6.l32_1[15:0];
    //     hdr.output.s_5.l16_1 = ig_md.t_7.l32_1[31:16];
    //     hdr.output.s_5.l16_2 = ig_md.t_7.l32_1[15:0];
    //     hdr.output.s_6.l16_1 = ig_md.t_1.l32_1[31:16];
    //     hdr.output.s_6.l16_2 = ig_md.t_1.l32_1[15:0];
    //     hdr.output.s_7.l16_1 = ig_md.t_2.l32_1[31:16];
    //     hdr.output.s_7.l16_2 = ig_md.t_2.l32_1[15:0];
    // }    
    // action act_swap2() {
    //     hdr.output.s_1.l16_1 = ig_md.t_4.l32_1[31:16];
    //     hdr.output.s_1.l16_2 = ig_md.t_4.l32_1[15:0];
    //     hdr.output.s_2.l16_1 = ig_md.t_5.l32_1[31:16];
    //     hdr.output.s_2.l16_2 = ig_md.t_5.l32_1[15:0];
    //     hdr.output.s_3.l16_1 = ig_md.t_6.l32_1[31:16];
    //     hdr.output.s_3.l16_2 = ig_md.t_6.l32_1[15:0];
    //     hdr.output.s_4.l16_1 = ig_md.t_7.l32_1[31:16];
    //     hdr.output.s_4.l16_2 = ig_md.t_7.l32_1[15:0];
    //     hdr.output.s_5.l16_1 = ig_md.t_1.l32_1[31:16];
    //     hdr.output.s_5.l16_2 = ig_md.t_1.l32_1[15:0];
    //     hdr.output.s_6.l16_1 = ig_md.t_2.l32_1[31:16];
    //     hdr.output.s_6.l16_2 = ig_md.t_2.l32_1[15:0];
    //     hdr.output.s_7.l16_1 = ig_md.t_3.l32_1[31:16];
    //     hdr.output.s_7.l16_2 = ig_md.t_3.l32_1[15:0];
    // }    
    // action act_swap3() {
    //     hdr.output.s_1.l16_1 = ig_md.t_5.l32_1[31:16];
    //     hdr.output.s_1.l16_2 = ig_md.t_5.l32_1[15:0];
    //     hdr.output.s_2.l16_1 = ig_md.t_6.l32_1[31:16];
    //     hdr.output.s_2.l16_2 = ig_md.t_6.l32_1[15:0];
    //     hdr.output.s_3.l16_1 = ig_md.t_7.l32_1[31:16];
    //     hdr.output.s_3.l16_2 = ig_md.t_7.l32_1[15:0];
    //     hdr.output.s_4.l16_1 = ig_md.t_1.l32_1[31:16];
    //     hdr.output.s_4.l16_2 = ig_md.t_1.l32_1[15:0];
    //     hdr.output.s_5.l16_1 = ig_md.t_2.l32_1[31:16];
    //     hdr.output.s_5.l16_2 = ig_md.t_2.l32_1[15:0];
    //     hdr.output.s_6.l16_1 = ig_md.t_3.l32_1[31:16];
    //     hdr.output.s_6.l16_2 = ig_md.t_3.l32_1[15:0];
    //     hdr.output.s_7.l16_1 = ig_md.t_4.l32_1[31:16];
    //     hdr.output.s_7.l16_2 = ig_md.t_4.l32_1[15:0];
    // }    
    // action act_swap4() {
    //     hdr.output.s_1.l16_1 = ig_md.t_6.l32_1[31:16];
    //     hdr.output.s_1.l16_2 = ig_md.t_6.l32_1[15:0];
    //     hdr.output.s_2.l16_1 = ig_md.t_7.l32_1[31:16];
    //     hdr.output.s_2.l16_2 = ig_md.t_7.l32_1[15:0];
    //     hdr.output.s_3.l16_1 = ig_md.t_1.l32_1[31:16];
    //     hdr.output.s_3.l16_2 = ig_md.t_1.l32_1[15:0];
    //     hdr.output.s_4.l16_1 = ig_md.t_2.l32_1[31:16];
    //     hdr.output.s_4.l16_2 = ig_md.t_2.l32_1[15:0];
    //     hdr.output.s_5.l16_1 = ig_md.t_3.l32_1[31:16];
    //     hdr.output.s_5.l16_2 = ig_md.t_3.l32_1[15:0];
    //     hdr.output.s_6.l16_1 = ig_md.t_4.l32_1[31:16];
    //     hdr.output.s_6.l16_2 = ig_md.t_4.l32_1[15:0];
    //     hdr.output.s_7.l16_1 = ig_md.t_5.l32_1[31:16];
    //     hdr.output.s_7.l16_2 = ig_md.t_5.l32_1[15:0];
    // }    
    // action act_swap5() {
    //     hdr.output.s_1.l16_1 = ig_md.t_7.l32_1[31:16];
    //     hdr.output.s_1.l16_2 = ig_md.t_7.l32_1[15:0];
    //     hdr.output.s_2.l16_1 = ig_md.t_1.l32_1[31:16];
    //     hdr.output.s_2.l16_2 = ig_md.t_1.l32_1[15:0];
    //     hdr.output.s_3.l16_1 = ig_md.t_2.l32_1[31:16];
    //     hdr.output.s_3.l16_2 = ig_md.t_2.l32_1[15:0];
    //     hdr.output.s_4.l16_1 = ig_md.t_3.l32_1[31:16];
    //     hdr.output.s_4.l16_2 = ig_md.t_3.l32_1[15:0];
    //     hdr.output.s_5.l16_1 = ig_md.t_4.l32_1[31:16];
    //     hdr.output.s_5.l16_2 = ig_md.t_4.l32_1[15:0];
    //     hdr.output.s_6.l16_1 = ig_md.t_5.l32_1[31:16];
    //     hdr.output.s_6.l16_2 = ig_md.t_5.l32_1[15:0];
    //     hdr.output.s_7.l16_1 = ig_md.t_6.l32_1[31:16];
    //     hdr.output.s_7.l16_2 = ig_md.t_6.l32_1[15:0];
    // }    
    // action act_swap6() {
    //     hdr.output.s_1.l16_1 = ig_md.t_1.l32_1[31:16];
    //     hdr.output.s_1.l16_2 = ig_md.t_1.l32_1[15:0];
    //     hdr.output.s_2.l16_1 = ig_md.t_2.l32_1[31:16];
    //     hdr.output.s_2.l16_2 = ig_md.t_2.l32_1[15:0];
    //     hdr.output.s_3.l16_1 = ig_md.t_3.l32_1[31:16];
    //     hdr.output.s_3.l16_2 = ig_md.t_3.l32_1[15:0];
    //     hdr.output.s_4.l16_1 = ig_md.t_4.l32_1[31:16];
    //     hdr.output.s_4.l16_2 = ig_md.t_4.l32_1[15:0];
    //     hdr.output.s_5.l16_1 = ig_md.t_5.l32_1[31:16];
    //     hdr.output.s_5.l16_2 = ig_md.t_5.l32_1[15:0];
    //     hdr.output.s_6.l16_1 = ig_md.t_6.l32_1[31:16];
    //     hdr.output.s_6.l16_2 = ig_md.t_6.l32_1[15:0];
    //     hdr.output.s_7.l16_1 = ig_md.t_7.l32_1[31:16];
    //     hdr.output.s_7.l16_2 = ig_md.t_7.l32_1[15:0];
    // }
    // @stage(10)
    // table tab_swap {
    //     size = 7;
    //     key = { ig_md.pkt_count_mod_7: exact; }
    //     actions = { act_swap0; act_swap1; act_swap2; act_swap3; act_swap4; act_swap5; act_swap6; }
    //     const entries = {
    //         (0):act_swap0(); (1):act_swap1(); (2):act_swap2(); (3):act_swap3(); (4):act_swap4(); (5):act_swap5(); (6):act_swap6();
    //     }
    //     const default_action = act_swap0();
    // }

// *********cal: h1****************************************************************

    action set_x1_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    action set_x1_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear2.l32_1 = in01 + ig_md.linear2.l32_1;
        ig_md.linear2.l32_2 = in02 + ig_md.linear2.l32_2;
        ig_md.linear2.l32_3 = in03 + ig_md.linear2.l32_3;
        ig_md.linear2.l32_4 = in04 + ig_md.linear2.l32_4;
    }
    
    action set_x1_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear3.l32_1 = in01 + ig_md.linear3.l32_1;
        ig_md.linear3.l32_2 = in02 + ig_md.linear3.l32_2;
        ig_md.linear3.l32_3 = in03 + ig_md.linear3.l32_3;
        ig_md.linear3.l32_4 = in04 + ig_md.linear3.l32_4;
    }
    
    action set_x1_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear4.l32_1 = in01 + ig_md.linear4.l32_1;
        ig_md.linear4.l32_2 = in02 + ig_md.linear4.l32_2;
        ig_md.linear4.l32_3 = in03 + ig_md.linear4.l32_3;
        ig_md.linear4.l32_4 = in04 + ig_md.linear4.l32_4;
    }
    
    action set_h0_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01;
        hdr.output.linear1.l32_2 = in02;
        hdr.output.linear1.l32_3 = in03;
        hdr.output.linear1.l32_4 = in04;
    }
    
    action set_h0_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear2.l32_1 = in01;
        ig_md.linear2.l32_2 = in02;
        ig_md.linear2.l32_3 = in03;
        ig_md.linear2.l32_4 = in04;
    }
    
    action set_h0_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear3.l32_1 = in01;
        ig_md.linear3.l32_2 = in02;
        ig_md.linear3.l32_3 = in03;
        ig_md.linear3.l32_4 = in04;
    }
    
    action set_h0_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear4.l32_1 = in01;
        ig_md.linear4.l32_2 = in02;
        ig_md.linear4.l32_3 = in03;
        ig_md.linear4.l32_4 = in04;
    }
    
    @stage (11)
    table x1_1_2 {
        key = {
            hdr.output.s_1.l16_1: ternary;
        }
        actions = {
            set_x1_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }    
    @stage (11)
    table x1_3_4 {
        key = {
            hdr.output.s_1.l16_2: ternary;
        }
        actions = {
            set_x1_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (10)
    table h0_1_2 {
        key = {
            ig_md.linear0.l32_1: ternary;
        }
        actions = {
            set_h0_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (10)
    table h0_3_4 {
        key = {
            ig_md.linear0.l32_2: ternary;
        }
        actions = {
            set_h0_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (10)
    table h0_5_6 {
        key = {
            ig_md.linear0.l32_3: ternary;
        }
        actions = {
            set_h0_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (10)
    table h0_7_8 {
        key = {
            ig_md.linear0.l32_4: ternary;
        }
        actions = {
            set_h0_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
// *** cal h2

    action set_x2_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear0.l32_1 = in01;
        ig_md.linear0.l32_2 = in02;
        ig_md.linear0.l32_3 = in03;
        ig_md.linear0.l32_4 = in04;
    }
    
    action set_x2_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear1.l32_1 = in01;
        ig_md.linear1.l32_2 = in02;
        ig_md.linear1.l32_3 = in03;
        ig_md.linear1.l32_4 = in04;
    }
    
    action set_x2_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear3.l32_1 = in01;
        ig_md.linear3.l32_2 = in02;
        ig_md.linear3.l32_3 = in03;
        ig_md.linear3.l32_4 = in04;
    }
    
    action set_x2_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear4.l32_1 = in01;
        ig_md.linear4.l32_2 = in02;
        ig_md.linear4.l32_3 = in03;
        ig_md.linear4.l32_4 = in04;
    }
    
    action set_h1_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear0.l32_1 = in01 + ig_md.linear0.l32_1;
        ig_md.linear0.l32_2 = in02 + ig_md.linear0.l32_2;
        ig_md.linear0.l32_3 = in03 + ig_md.linear0.l32_3;
        ig_md.linear0.l32_4 = in04 + ig_md.linear0.l32_4;
    }
    
    action set_h1_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear1.l32_1 = in01 + ig_md.linear1.l32_1;
        ig_md.linear1.l32_2 = in02 + ig_md.linear1.l32_2;
        ig_md.linear1.l32_3 = in03 + ig_md.linear1.l32_3;
        ig_md.linear1.l32_4 = in04 + ig_md.linear1.l32_4;
    }
    
    action set_h1_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear3.l32_1 = in01 + ig_md.linear3.l32_1;
        ig_md.linear3.l32_2 = in02 + ig_md.linear3.l32_2;
        ig_md.linear3.l32_3 = in03 + ig_md.linear3.l32_3;
        ig_md.linear3.l32_4 = in04 + ig_md.linear3.l32_4;
    }
    
    action set_h1_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear4.l32_1 = in01 + ig_md.linear4.l32_1;
        ig_md.linear4.l32_2 = in02 + ig_md.linear4.l32_2;
        ig_md.linear4.l32_3 = in03 + ig_md.linear4.l32_3;
        ig_md.linear4.l32_4 = in04 + ig_md.linear4.l32_4;
    }
    
    @stage (13)
    table x2_1_2 {
        key = {
            hdr.output.s_2.l16_1: ternary;
        }
        actions = {
            set_x2_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }    
    @stage (13)
    table x2_3_4 {
        key = {
            hdr.output.s_2.l16_2: ternary;
        }
        actions = {
            set_x2_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (14)
    table h1_1_2 {
        key = {
            hdr.output.linear1.l32_1: ternary;
        }
        actions = {
            set_h1_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (14)
    table h1_3_4 {
        key = {
            hdr.output.linear1.l32_2: ternary;
        }
        actions = {
            set_h1_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (14)
    table h1_5_6 {
        key = {
            hdr.output.linear1.l32_3: ternary;
        }
        actions = {
            set_h1_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (14)
    table h1_7_8 {
        key = {
            hdr.output.linear1.l32_4: ternary;
        }
        actions = {
            set_h1_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
// *** cal h3

    action set_x3_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01;
        hdr.output.linear1.l32_2 = in02;
        hdr.output.linear1.l32_3 = in03;
        hdr.output.linear1.l32_4 = in04;
    }
    
    action set_x3_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear2.l32_1 = in01;
        ig_md.linear2.l32_2 = in02;
        ig_md.linear2.l32_3 = in03;
        ig_md.linear2.l32_4 = in04;
    }
    
    action set_x3_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear3.l32_1 = in01;
        ig_md.linear3.l32_2 = in02;
        ig_md.linear3.l32_3 = in03;
        ig_md.linear3.l32_4 = in04;
    }
    
    action set_x3_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear4.l32_1 = in01;
        ig_md.linear4.l32_2 = in02;
        ig_md.linear4.l32_3 = in03;
        ig_md.linear4.l32_4 = in04;
    }
    
    action set_h2_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    action set_h2_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear2.l32_1 = in01 + ig_md.linear2.l32_1;
        ig_md.linear2.l32_2 = in02 + ig_md.linear2.l32_2;
        ig_md.linear2.l32_3 = in03 + ig_md.linear2.l32_3;
        ig_md.linear2.l32_4 = in04 + ig_md.linear2.l32_4;
    }
    
    action set_h2_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear3.l32_1 = in01 + ig_md.linear3.l32_1;
        ig_md.linear3.l32_2 = in02 + ig_md.linear3.l32_2;
        ig_md.linear3.l32_3 = in03 + ig_md.linear3.l32_3;
        ig_md.linear3.l32_4 = in04 + ig_md.linear3.l32_4;
    }
    
    action set_h2_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear4.l32_1 = in01 + ig_md.linear4.l32_1;
        ig_md.linear4.l32_2 = in02 + ig_md.linear4.l32_2;
        ig_md.linear4.l32_3 = in03 + ig_md.linear4.l32_3;
        ig_md.linear4.l32_4 = in04 + ig_md.linear4.l32_4;
    }
    
    @stage (16)
    table x3_1_2 {
        key = {
            hdr.output.s_3.l16_1: ternary;
        }
        actions = {
            set_x3_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }    
    @stage (16)
    table x3_3_4 {
        key = {
            hdr.output.s_3.l16_2: ternary;
        }
        actions = {
            set_x3_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

    @stage (17)
    table h2_1_2 {
        key = {
            ig_md.linear0.l32_1: ternary;
        }
        actions = {
            set_h2_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (17)
    table h2_3_4 {
        key = {
            ig_md.linear0.l32_2: ternary;
        }
        actions = {
            set_h2_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (17)
    table h2_5_6 {
        key = {
            ig_md.linear0.l32_3: ternary;
        }
        actions = {
            set_h2_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (17)
    table h2_7_8 {
        key = {
            ig_md.linear0.l32_4: ternary;
        }
        actions = {
            set_h2_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

// ***********************

    apply
    {
// *****获取嵌入层******
        emb_pkl.apply();
        emb_ipd.apply();
        hdr.output.x_8.l16_1 = hdr.output.x_8.l16_1 + ig_md.x_temp.l16_1;
        hdr.output.x_8.l16_2 = hdr.output.x_8.l16_2 + ig_md.x_temp.l16_2;
        hdr.output.x_8.l16_3 = hdr.output.x_8.l16_3 + ig_md.x_temp.l16_3;
        hdr.output.x_8.l16_4 = hdr.output.x_8.l16_4 + ig_md.x_temp.l16_4;
        x0_1_2.apply();
        x0_3_4.apply();
        x0_5_6.apply();
        x0_7_8.apply();
        
        // cal_flow_hash(); //stage0
        // if(hdr.ethernet.ether_type == ETHERTYPE_IPV4)
        // {
        //     if(ig_md.src_port != 68)
        //     {
        //         ig_tm_md.ucast_egress_port = 0x120; //stage0
        //     }
        // }
        // Update_full_flow_hash(); //stage1
        // if(ig_md.flow_hash != ig_md.flow_hash1){
        //     Init_pkt_count(); //stage2
        //     Init_pkt_count_mod_7(); //stage2
        // }
        // else{
        //     Update_pkt_count(); //stage2
        //     Update_pkt_count_mod_7(); //stage2
        // }
        // hdr.output.t_8.l32_1[31:24] = hdr.output.t4_8.l8_1;
        // hdr.output.t_8.l32_1[23:16] = hdr.output.t4_8.l8_2;
        // hdr.output.t_8.l32_1[15:8] = hdr.output.t4_8.l8_3;
        // // hdr.output.t_8.l32_1[7:0] = hdr.output.t4_8.l8_4;
        // access_pkt_embeded_feature_0_1();
        // access_pkt_embeded_feature_1_1();
        // access_pkt_embeded_feature_2_1();
        // access_pkt_embeded_feature_3_1();
        // access_pkt_embeded_feature_4_1();
        // access_pkt_embeded_feature_5_1();
        // access_pkt_embeded_feature_6_1();
        // tab_swap.apply();

// *****cal h1 as linear1

        h0_1_2.apply();
        h0_3_4.apply();
        h0_5_6.apply();
        h0_7_8.apply();
        x1_1_2.apply();
        x1_3_4.apply();
        // x1_5_6.apply();
        // x1_7_8.apply();

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + ig_md.linear3.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + ig_md.linear3.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + ig_md.linear3.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + ig_md.linear3.l32_4;

        ig_md.linear2.l32_1 = ig_md.linear2.l32_1 + ig_md.linear4.l32_1;
        ig_md.linear2.l32_2 = ig_md.linear2.l32_2 + ig_md.linear4.l32_2;
        ig_md.linear2.l32_3 = ig_md.linear2.l32_3 + ig_md.linear4.l32_3;
        ig_md.linear2.l32_4 = ig_md.linear2.l32_4 + ig_md.linear4.l32_4;

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + ig_md.linear2.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + ig_md.linear2.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + ig_md.linear2.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + ig_md.linear2.l32_4;
//linear2 used
// *****cal h2 as linear0
        x2_1_2.apply();
        x2_3_4.apply();
        // x2_5_6.apply();
        // x2_7_8.apply();

        h1_1_2.apply();
        h1_3_4.apply();
        h1_5_6.apply();
        h1_7_8.apply();

        ig_md.linear0.l32_1 = ig_md.linear0.l32_1 + ig_md.linear3.l32_1;
        ig_md.linear0.l32_2 = ig_md.linear0.l32_2 + ig_md.linear3.l32_2;
        ig_md.linear0.l32_3 = ig_md.linear0.l32_3 + ig_md.linear3.l32_3;
        ig_md.linear0.l32_4 = ig_md.linear0.l32_4 + ig_md.linear3.l32_4;

        ig_md.linear1.l32_1 = ig_md.linear1.l32_1 + ig_md.linear4.l32_1;
        ig_md.linear1.l32_2 = ig_md.linear1.l32_2 + ig_md.linear4.l32_2;
        ig_md.linear1.l32_3 = ig_md.linear1.l32_3 + ig_md.linear4.l32_3;
        ig_md.linear1.l32_4 = ig_md.linear1.l32_4 + ig_md.linear4.l32_4;

        ig_md.linear0.l32_1 = ig_md.linear0.l32_1 + ig_md.linear1.l32_1;
        ig_md.linear0.l32_2 = ig_md.linear0.l32_2 + ig_md.linear1.l32_2;
        ig_md.linear0.l32_3 = ig_md.linear0.l32_3 + ig_md.linear1.l32_3;
        ig_md.linear0.l32_4 = ig_md.linear0.l32_4 + ig_md.linear1.l32_4;
//linear1 used
// *****cal h3 as linear1
        x3_1_2.apply();
        x3_3_4.apply();
        // x3_5_6.apply();
        // x3_7_8.apply();

        h2_1_2.apply();
        h2_3_4.apply();
        h2_5_6.apply();
        h2_7_8.apply();

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + ig_md.linear3.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + ig_md.linear3.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + ig_md.linear3.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + ig_md.linear3.l32_4;

        ig_md.linear2.l32_1 = ig_md.linear2.l32_1 + ig_md.linear4.l32_1;
        ig_md.linear2.l32_2 = ig_md.linear2.l32_2 + ig_md.linear4.l32_2;
        ig_md.linear2.l32_3 = ig_md.linear2.l32_3 + ig_md.linear4.l32_3;
        ig_md.linear2.l32_4 = ig_md.linear2.l32_4 + ig_md.linear4.l32_4;

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + ig_md.linear2.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + ig_md.linear2.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + ig_md.linear2.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + ig_md.linear2.l32_4;  



        hdr.output.setValid();

        hdr.ethernet.dst_addr = 0; //for filter
    }
}

control SwitchEgress(
        inout header_t hdr,
        inout eg_metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t eg_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) 
{

    action noaction(){}

 
// *** cal h4

    action set_x4_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear0.l32_1 = in01;
        eg_md.linear0.l32_2 = in02;
        eg_md.linear0.l32_3 = in03;
        eg_md.linear0.l32_4 = in04;
    }
    
    action set_x4_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01;
        eg_md.linear2.l32_2 = in02;
        eg_md.linear2.l32_3 = in03;
        eg_md.linear2.l32_4 = in04;
    }
    
    action set_x4_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01;
        eg_md.linear3.l32_2 = in02;
        eg_md.linear3.l32_3 = in03;
        eg_md.linear3.l32_4 = in04;
    }
    
    action set_x4_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01;
        eg_md.linear4.l32_2 = in02;
        eg_md.linear4.l32_3 = in03;
        eg_md.linear4.l32_4 = in04;
    }
    
    action set_h3_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear0.l32_1 = in01 + eg_md.linear0.l32_1;
        eg_md.linear0.l32_2 = in02 + eg_md.linear0.l32_2;
        eg_md.linear0.l32_3 = in03 + eg_md.linear0.l32_3;
        eg_md.linear0.l32_4 = in04 + eg_md.linear0.l32_4;
    }
    
    action set_h3_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01 + eg_md.linear2.l32_1;
        eg_md.linear2.l32_2 = in02 + eg_md.linear2.l32_2;
        eg_md.linear2.l32_3 = in03 + eg_md.linear2.l32_3;
        eg_md.linear2.l32_4 = in04 + eg_md.linear2.l32_4;
    }
    
    action set_h3_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01 + eg_md.linear3.l32_1;
        eg_md.linear3.l32_2 = in02 + eg_md.linear3.l32_2;
        eg_md.linear3.l32_3 = in03 + eg_md.linear3.l32_3;
        eg_md.linear3.l32_4 = in04 + eg_md.linear3.l32_4;
    }
    
    action set_h3_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01 + eg_md.linear4.l32_1;
        eg_md.linear4.l32_2 = in02 + eg_md.linear4.l32_2;
        eg_md.linear4.l32_3 = in03 + eg_md.linear4.l32_3;
        eg_md.linear4.l32_4 = in04 + eg_md.linear4.l32_4;
    }

    
    @stage (0)
    table x4_1_2 {
        key = {
            hdr.output.s_4.l16_1: ternary;
        }
        actions = {
            set_x4_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (0)
    table x4_3_4 {
        key = {
            hdr.output.s_4.l16_2: ternary;
        }
        actions = {
            set_x4_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (1)
    table h3_1_2 {
        key = {
            hdr.output.linear1.l32_1: ternary;
        }
        actions = {
            set_h3_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (1)
    table h3_3_4 {
        key = {
            hdr.output.linear1.l32_2: ternary;
        }
        actions = {
            set_h3_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (1)
    table h3_5_6 {
        key = {
            hdr.output.linear1.l32_3: ternary;
        }
        actions = {
            set_h3_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (1)
    table h3_7_8 {
        key = {
            hdr.output.linear1.l32_4: ternary;
        }
        actions = {
            set_h3_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
// *** cal h5

    action set_x5_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01;
        hdr.output.linear1.l32_2 = in02;
        hdr.output.linear1.l32_3 = in03;
        hdr.output.linear1.l32_4 = in04;
    }
    
    action set_x5_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01;
        eg_md.linear2.l32_2 = in02;
        eg_md.linear2.l32_3 = in03;
        eg_md.linear2.l32_4 = in04;
    }
    
    action set_x5_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01;
        eg_md.linear3.l32_2 = in02;
        eg_md.linear3.l32_3 = in03;
        eg_md.linear3.l32_4 = in04;
    }
    
    action set_x5_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01;
        eg_md.linear4.l32_2 = in02;
        eg_md.linear4.l32_3 = in03;
        eg_md.linear4.l32_4 = in04;
    }
    
    action set_h4_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    action set_h4_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01 + eg_md.linear2.l32_1;
        eg_md.linear2.l32_2 = in02 + eg_md.linear2.l32_2;
        eg_md.linear2.l32_3 = in03 + eg_md.linear2.l32_3;
        eg_md.linear2.l32_4 = in04 + eg_md.linear2.l32_4;
    }
    
    action set_h4_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01 + eg_md.linear3.l32_1;
        eg_md.linear3.l32_2 = in02 + eg_md.linear3.l32_2;
        eg_md.linear3.l32_3 = in03 + eg_md.linear3.l32_3;
        eg_md.linear3.l32_4 = in04 + eg_md.linear3.l32_4;
    }
    
    action set_h4_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01 + eg_md.linear4.l32_1;
        eg_md.linear4.l32_2 = in02 + eg_md.linear4.l32_2;
        eg_md.linear4.l32_3 = in03 + eg_md.linear4.l32_3;
        eg_md.linear4.l32_4 = in04 + eg_md.linear4.l32_4;
    }
    
    @stage (3)
    table x5_1_2 {
        key = {
            hdr.output.s_5.l16_1: ternary;
        }
        actions = {
            set_x5_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }    
    @stage (3)
    table x5_3_4 {
        key = {
            hdr.output.s_5.l16_2: ternary;
        }
        actions = {
            set_x5_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

    @stage (4)
    table h4_1_2 {
        key = {
            eg_md.linear0.l32_1: ternary;
        }
        actions = {
            set_h4_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (4)
    table h4_3_4 {
        key = {
            eg_md.linear0.l32_2: ternary;
        }
        actions = {
            set_h4_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (4)
    table h4_5_6 {
        key = {
            eg_md.linear0.l32_3: ternary;
        }
        actions = {
            set_h4_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (4)
    table h4_7_8 {
        key = {
            eg_md.linear0.l32_4: ternary;
        }
        actions = {
            set_h4_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

// *****cal h6

    action set_x6_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear0.l32_1 = in01;
        eg_md.linear0.l32_2 = in02;
        eg_md.linear0.l32_3 = in03;
        eg_md.linear0.l32_4 = in04;
    }
    
    action set_x6_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01;
        eg_md.linear2.l32_2 = in02;
        eg_md.linear2.l32_3 = in03;
        eg_md.linear2.l32_4 = in04;
    }
    
    action set_x6_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01;
        eg_md.linear3.l32_2 = in02;
        eg_md.linear3.l32_3 = in03;
        eg_md.linear3.l32_4 = in04;
    }
    
    action set_x6_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01;
        eg_md.linear4.l32_2 = in02;
        eg_md.linear4.l32_3 = in03;
        eg_md.linear4.l32_4 = in04;
    }
    
    action set_h5_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear0.l32_1 = in01 + eg_md.linear0.l32_1;
        eg_md.linear0.l32_2 = in02 + eg_md.linear0.l32_2;
        eg_md.linear0.l32_3 = in03 + eg_md.linear0.l32_3;
        eg_md.linear0.l32_4 = in04 + eg_md.linear0.l32_4;
    }
    
    action set_h5_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01 + eg_md.linear2.l32_1;
        eg_md.linear2.l32_2 = in02 + eg_md.linear2.l32_2;
        eg_md.linear2.l32_3 = in03 + eg_md.linear2.l32_3;
        eg_md.linear2.l32_4 = in04 + eg_md.linear2.l32_4;
    }
    
    action set_h5_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01 + eg_md.linear3.l32_1;
        eg_md.linear3.l32_2 = in02 + eg_md.linear3.l32_2;
        eg_md.linear3.l32_3 = in03 + eg_md.linear3.l32_3;
        eg_md.linear3.l32_4 = in04 + eg_md.linear3.l32_4;
    }
    
    action set_h5_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01 + eg_md.linear4.l32_1;
        eg_md.linear4.l32_2 = in02 + eg_md.linear4.l32_2;
        eg_md.linear4.l32_3 = in03 + eg_md.linear4.l32_3;
        eg_md.linear4.l32_4 = in04 + eg_md.linear4.l32_4;
    }
    
    @stage (6)
    table x6_1_2 {
        key = {
            hdr.output.s_6.l16_1: ternary;
        }
        actions = {
            set_x6_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }    
    @stage (6)
    table x6_3_4 {
        key = {
            hdr.output.s_6.l16_2: ternary;
        }
        actions = {
            set_x6_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

    @stage (7)
    table h5_1_2 {
        key = {
            hdr.output.linear1.l32_1: ternary;
        }
        actions = {
            set_h5_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (7)
    table h5_3_4 {
        key = {
            hdr.output.linear1.l32_2: ternary;
        }
        actions = {
            set_h5_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (7)
    table h5_5_6 {
        key = {
            hdr.output.linear1.l32_3: ternary;
        }
        actions = {
            set_h5_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (7)
    table h5_7_8 {
        key = {
            hdr.output.linear1.l32_4: ternary;
        }
        actions = {
            set_h5_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
// *****cal h7

    action set_x7_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01;
        hdr.output.linear1.l32_2 = in02;
        hdr.output.linear1.l32_3 = in03;
        hdr.output.linear1.l32_4 = in04;
    }
    
    action set_x7_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01;
        eg_md.linear2.l32_2 = in02;
        eg_md.linear2.l32_3 = in03;
        eg_md.linear2.l32_4 = in04;
    }
    
    action set_x7_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01;
        eg_md.linear3.l32_2 = in02;
        eg_md.linear3.l32_3 = in03;
        eg_md.linear3.l32_4 = in04;
    }
    
    action set_x7_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01;
        eg_md.linear4.l32_2 = in02;
        eg_md.linear4.l32_3 = in03;
        eg_md.linear4.l32_4 = in04;
    }
    
    action set_h6_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    action set_h6_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01 + eg_md.linear2.l32_1;
        eg_md.linear2.l32_2 = in02 + eg_md.linear2.l32_2;
        eg_md.linear2.l32_3 = in03 + eg_md.linear2.l32_3;
        eg_md.linear2.l32_4 = in04 + eg_md.linear2.l32_4;
    }
    
    action set_h6_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01 + eg_md.linear3.l32_1;
        eg_md.linear3.l32_2 = in02 + eg_md.linear3.l32_2;
        eg_md.linear3.l32_3 = in03 + eg_md.linear3.l32_3;
        eg_md.linear3.l32_4 = in04 + eg_md.linear3.l32_4;
    }
    
    action set_h6_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01 + eg_md.linear4.l32_1;
        eg_md.linear4.l32_2 = in02 + eg_md.linear4.l32_2;
        eg_md.linear4.l32_3 = in03 + eg_md.linear4.l32_3;
        eg_md.linear4.l32_4 = in04 + eg_md.linear4.l32_4;
    }
    
    @stage (9)
    table x7_1_2 {
        key = {
            hdr.output.s_7.l16_1: ternary;
        }
        actions = {
            set_x7_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }    
    @stage (9)
    table x7_3_4 {
        key = {
            hdr.output.s_7.l16_2: ternary;
        }
        actions = {
            set_x7_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

    @stage (10)
    table h6_1_2 {
        key = {
            eg_md.linear0.l32_1: ternary;
        }
        actions = {
            set_h6_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (10)
    table h6_3_4 {
        key = {
            eg_md.linear0.l32_2: ternary;
        }
        actions = {
            set_h6_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (10)
    table h6_5_6 {
        key = {
            eg_md.linear0.l32_3: ternary;
        }
        actions = {
            set_h6_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (10)
    table h6_7_8 {
        key = {
            eg_md.linear0.l32_4: ternary;
        }
        actions = {
            set_h6_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
// *****cal h8

    action set_x8_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear0.l32_1 = in01;
        eg_md.linear0.l32_2 = in02;
        eg_md.linear0.l32_3 = in03;
        eg_md.linear0.l32_4 = in04;
    }
    
    action set_x8_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01;
        eg_md.linear2.l32_2 = in02;
        eg_md.linear2.l32_3 = in03;
        eg_md.linear2.l32_4 = in04;
    }
    
    action set_x8_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01;
        eg_md.linear3.l32_2 = in02;
        eg_md.linear3.l32_3 = in03;
        eg_md.linear3.l32_4 = in04;
    }
    
    action set_x8_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01;
        eg_md.linear4.l32_2 = in02;
        eg_md.linear4.l32_3 = in03;
        eg_md.linear4.l32_4 = in04;
    }
    
    action set_h7_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear0.l32_1 = in01 + eg_md.linear0.l32_1;
        eg_md.linear0.l32_2 = in02 + eg_md.linear0.l32_2;
        eg_md.linear0.l32_3 = in03 + eg_md.linear0.l32_3;
        eg_md.linear0.l32_4 = in04 + eg_md.linear0.l32_4;
    }
    
    action set_h7_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01 + eg_md.linear2.l32_1;
        eg_md.linear2.l32_2 = in02 + eg_md.linear2.l32_2;
        eg_md.linear2.l32_3 = in03 + eg_md.linear2.l32_3;
        eg_md.linear2.l32_4 = in04 + eg_md.linear2.l32_4;
    }
    
    action set_h7_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01 + eg_md.linear3.l32_1;
        eg_md.linear3.l32_2 = in02 + eg_md.linear3.l32_2;
        eg_md.linear3.l32_3 = in03 + eg_md.linear3.l32_3;
        eg_md.linear3.l32_4 = in04 + eg_md.linear3.l32_4;
    }
    
    action set_h7_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01 + eg_md.linear4.l32_1;
        eg_md.linear4.l32_2 = in02 + eg_md.linear4.l32_2;
        eg_md.linear4.l32_3 = in03 + eg_md.linear4.l32_3;
        eg_md.linear4.l32_4 = in04 + eg_md.linear4.l32_4;
    }
    
    @stage (12)
    table x8_1_2 {
        key = {
            hdr.output.s_8.l16_1: ternary;
        }
        actions = {
            set_x8_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (12)
    table x8_3_4 {
        key = {
            hdr.output.s_8.l16_1: ternary;
        }
        actions = {
            set_x8_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

    @stage (13)
    table h7_1_2 {
        key = {
            hdr.output.linear1.l32_1: ternary;
        }
        actions = {
            set_h7_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (13)
    table h7_3_4 {
        key = {
            hdr.output.linear1.l32_2: ternary;
        }
        actions = {
            set_h7_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (13)
    table h7_5_6 {
        key = {
            hdr.output.linear1.l32_3: ternary;
        }
        actions = {
            set_h7_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (13)
    table h7_7_8 {
        key = {
            hdr.output.linear1.l32_4: ternary;
        }
        actions = {
            set_h7_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

// *****cal ans

    action set_h8_1_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01;
        hdr.output.linear1.l32_2 = in02;
        hdr.output.linear1.l32_3 = in03;
        hdr.output.linear1.l32_4 = in04;
    }
    
    action set_h8_3_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear2.l32_1 = in01;
        eg_md.linear2.l32_2 = in02;
        eg_md.linear2.l32_3 = in03;
        eg_md.linear2.l32_4 = in04;
    }
    
    action set_h8_5_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear3.l32_1 = in01;
        eg_md.linear3.l32_2 = in02;
        eg_md.linear3.l32_3 = in03;
        eg_md.linear3.l32_4 = in04;
    }
    
    action set_h8_7_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        eg_md.linear4.l32_1 = in01;
        eg_md.linear4.l32_2 = in02;
        eg_md.linear4.l32_3 = in03;
        eg_md.linear4.l32_4 = in04;
    }
    
    @stage (16)
    table h8_1_2 {
        key = {
            eg_md.linear0.l32_1: ternary;
        }
        actions = {
            set_h8_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16)
    table h8_3_4 {
        key = {
            eg_md.linear0.l32_2: ternary;
        }
        actions = {
            set_h8_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16)
    table h8_5_6 {
        key = {
            eg_md.linear0.l32_3: ternary;
        }
        actions = {
            set_h8_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (17)
    table h8_7_8 {
        key = {
            eg_md.linear0.l32_4: ternary;
        }
        actions = {
            set_h8_7_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
// *** begin
    
    apply {

// *****cal h4 as linear0
        x4_1_2.apply();
        x4_3_4.apply();
        // x4_5_6.apply();
        // x4_7_8.apply();

        h3_1_2.apply();
        h3_3_4.apply();
        h3_5_6.apply();
        h3_7_8.apply();

        eg_md.linear0.l32_1 = eg_md.linear0.l32_1 + eg_md.linear3.l32_1;
        eg_md.linear0.l32_2 = eg_md.linear0.l32_2 + eg_md.linear3.l32_2;
        eg_md.linear0.l32_3 = eg_md.linear0.l32_3 + eg_md.linear3.l32_3;
        eg_md.linear0.l32_4 = eg_md.linear0.l32_4 + eg_md.linear3.l32_4;

        eg_md.linear2.l32_1 = eg_md.linear2.l32_1 + eg_md.linear4.l32_1;
        eg_md.linear2.l32_2 = eg_md.linear2.l32_2 + eg_md.linear4.l32_2;
        eg_md.linear2.l32_3 = eg_md.linear2.l32_3 + eg_md.linear4.l32_3;
        eg_md.linear2.l32_4 = eg_md.linear2.l32_4 + eg_md.linear4.l32_4;

        eg_md.linear0.l32_1 = eg_md.linear0.l32_1 + eg_md.linear2.l32_1;
        eg_md.linear0.l32_2 = eg_md.linear0.l32_2 + eg_md.linear2.l32_2;
        eg_md.linear0.l32_3 = eg_md.linear0.l32_3 + eg_md.linear2.l32_3;
        eg_md.linear0.l32_4 = eg_md.linear0.l32_4 + eg_md.linear2.l32_4;

// *****cal h5 as linear1
        x5_1_2.apply();
        x5_3_4.apply();
        // x5_5_6.apply();
        // x5_7_8.apply();

        h4_1_2.apply();
        h4_3_4.apply();
        h4_5_6.apply();
        h4_7_8.apply();

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + eg_md.linear3.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + eg_md.linear3.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + eg_md.linear3.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + eg_md.linear3.l32_4;

        eg_md.linear2.l32_1 = eg_md.linear2.l32_1 + eg_md.linear4.l32_1;
        eg_md.linear2.l32_2 = eg_md.linear2.l32_2 + eg_md.linear4.l32_2;
        eg_md.linear2.l32_3 = eg_md.linear2.l32_3 + eg_md.linear4.l32_3;
        eg_md.linear2.l32_4 = eg_md.linear2.l32_4 + eg_md.linear4.l32_4;

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + eg_md.linear2.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + eg_md.linear2.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + eg_md.linear2.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + eg_md.linear2.l32_4;
// *****cal h6 as linear0
        x6_1_2.apply();
        x6_3_4.apply();
        // x6_5_6.apply();
        // x6_7_8.apply();

        h5_1_2.apply();
        h5_3_4.apply();
        h5_5_6.apply();
        h5_7_8.apply();

        eg_md.linear0.l32_1 = eg_md.linear0.l32_1 + eg_md.linear3.l32_1;
        eg_md.linear0.l32_2 = eg_md.linear0.l32_2 + eg_md.linear3.l32_2;
        eg_md.linear0.l32_3 = eg_md.linear0.l32_3 + eg_md.linear3.l32_3;
        eg_md.linear0.l32_4 = eg_md.linear0.l32_4 + eg_md.linear3.l32_4;

        eg_md.linear2.l32_1 = eg_md.linear2.l32_1 + eg_md.linear4.l32_1;
        eg_md.linear2.l32_2 = eg_md.linear2.l32_2 + eg_md.linear4.l32_2;
        eg_md.linear2.l32_3 = eg_md.linear2.l32_3 + eg_md.linear4.l32_3;
        eg_md.linear2.l32_4 = eg_md.linear2.l32_4 + eg_md.linear4.l32_4;

        eg_md.linear0.l32_1 = eg_md.linear0.l32_1 + eg_md.linear2.l32_1;
        eg_md.linear0.l32_2 = eg_md.linear0.l32_2 + eg_md.linear2.l32_2;
        eg_md.linear0.l32_3 = eg_md.linear0.l32_3 + eg_md.linear2.l32_3;
        eg_md.linear0.l32_4 = eg_md.linear0.l32_4 + eg_md.linear2.l32_4;

// *****cal h7 as linear1
        x7_1_2.apply();
        x7_3_4.apply();
        // x7_5_6.apply();
        // x7_7_8.apply();

        h6_1_2.apply();
        h6_3_4.apply();
        h6_5_6.apply();
        h6_7_8.apply();

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + eg_md.linear3.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + eg_md.linear3.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + eg_md.linear3.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + eg_md.linear3.l32_4;

        eg_md.linear2.l32_1 = eg_md.linear2.l32_1 + eg_md.linear4.l32_1;
        eg_md.linear2.l32_2 = eg_md.linear2.l32_2 + eg_md.linear4.l32_2;
        eg_md.linear2.l32_3 = eg_md.linear2.l32_3 + eg_md.linear4.l32_3;
        eg_md.linear2.l32_4 = eg_md.linear2.l32_4 + eg_md.linear4.l32_4;

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + eg_md.linear2.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + eg_md.linear2.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + eg_md.linear2.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + eg_md.linear2.l32_4;  

// *****cal h8 as linear0
        x8_1_2.apply();
        x8_3_4.apply();
        // x8_5_6.apply();
        // x8_7_8.apply();

        h7_1_2.apply();
        h7_3_4.apply();
        h7_5_6.apply();
        h7_7_8.apply();

        eg_md.linear0.l32_1 = eg_md.linear0.l32_1 + eg_md.linear3.l32_1;
        eg_md.linear0.l32_2 = eg_md.linear0.l32_2 + eg_md.linear3.l32_2;
        eg_md.linear0.l32_3 = eg_md.linear0.l32_3 + eg_md.linear3.l32_3;
        eg_md.linear0.l32_4 = eg_md.linear0.l32_4 + eg_md.linear3.l32_4;

        eg_md.linear2.l32_1 = eg_md.linear2.l32_1 + eg_md.linear4.l32_1;
        eg_md.linear2.l32_2 = eg_md.linear2.l32_2 + eg_md.linear4.l32_2;
        eg_md.linear2.l32_3 = eg_md.linear2.l32_3 + eg_md.linear4.l32_3;
        eg_md.linear2.l32_4 = eg_md.linear2.l32_4 + eg_md.linear4.l32_4;

        eg_md.linear0.l32_1 = eg_md.linear0.l32_1 + eg_md.linear2.l32_1;
        eg_md.linear0.l32_2 = eg_md.linear0.l32_2 + eg_md.linear2.l32_2;
        eg_md.linear0.l32_3 = eg_md.linear0.l32_3 + eg_md.linear2.l32_3;
        eg_md.linear0.l32_4 = eg_md.linear0.l32_4 + eg_md.linear2.l32_4;

// *****cal ans as linear1
        h8_1_2.apply();
        h8_3_4.apply();
        h8_5_6.apply();
        h8_7_8.apply();

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + eg_md.linear3.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + eg_md.linear3.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + eg_md.linear3.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + eg_md.linear3.l32_4;

        eg_md.linear2.l32_1 = eg_md.linear2.l32_1 + eg_md.linear4.l32_1;
        eg_md.linear2.l32_2 = eg_md.linear2.l32_2 + eg_md.linear4.l32_2;
        eg_md.linear2.l32_3 = eg_md.linear2.l32_3 + eg_md.linear4.l32_3;
        eg_md.linear2.l32_4 = eg_md.linear2.l32_4 + eg_md.linear4.l32_4;

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + eg_md.linear2.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + eg_md.linear2.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + eg_md.linear2.l32_3;
        hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + eg_md.linear2.l32_4;

        hdr.ethernet.dst_addr = 0; //for filter
    }

}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         SwitchEgressParser(),
         SwitchEgress(),
         SwitchEgressDeparser()) pipe;

Switch(pipe) main;