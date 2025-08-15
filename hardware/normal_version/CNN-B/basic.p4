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

    // ***embedding**************************************************************************

    action set_emb_pkl(bit<16> in01, bit<16> in02) {
        ig_md.hidden_7.l16_1 = in01;
        ig_md.hidden_7.l16_2 = in02;
    }

    @stage (0)
    table emb_pkl {
        key = {
            hdr.input_feature.pkl: exact;
        }
        actions = {
            set_emb_pkl;
            noaction;
        }
        size = 4096;
        default_action = noaction();
    }

    action set_emb_ipd(bit<16> in01, bit<16> in02) {
        ig_md.hidden_6.l16_1 = in01;
        ig_md.hidden_6.l16_2 = in02;
    }

    @stage (0)
    table emb_ipd {
        key = {
            hdr.input_feature.ipd: exact;
        }
        actions = {
            set_emb_ipd;
            noaction;
        }
        size = 4096;
        default_action = noaction();
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
    // @stage(1)
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_0_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_0_1) access_pkt_embed_0_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7_0_0 == 0) { value =  ig_md.v; }
    //     } 
    // };
    // @stage(1)
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_1_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_1_1) access_pkt_embed_1_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7_1_0 == 1) { value =  ig_md.v; }
    //     } 
    // };
    // @stage(2)
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_2_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_2_1) access_pkt_embed_2_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7_2_0 == 2) { value =  ig_md.v; }
    //     } 
    // };
    // @stage(2)
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_3_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_3_1) access_pkt_embed_3_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7_3_0 == 3) { value =  ig_md.v; }
    //     } 
    // };
    // @stage(3)
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_4_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_4_1) access_pkt_embed_4_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7_4_0 == 4) { value =  ig_md.v; }
    //     } 
    // };
    // @stage(3)
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_5_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_5_1) access_pkt_embed_5_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7_5_0 == 5) { value =  ig_md.v; }
    //     } 
    // };
    // @stage(4)
    // Register<bit<WIDTH>, bit<Register_Index_Size>>(Register_Table_Size) pkt_embed_6_1; 
    // RegisterAction<bit<WIDTH>, bit<Register_Index_Size>, bit<WIDTH>>(pkt_embed_6_1) access_pkt_embed_6_1 = {
    //     void apply(inout bit<WIDTH> value, out bit<WIDTH> read_value) {
    //         read_value = value;
    //         if( ig_md.pkt_count_mod_7_6_0 == 6) { value =  ig_md.v; }
    //     } 
    // };
    // action access_pkt_embeded_feature_0_1() {
    //         ig_md.pkt_embeded_0.l16_1 = access_pkt_embed_0_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_1_1() {
    //         ig_md.pkt_embeded_1.l16_1 = access_pkt_embed_1_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_2_1() {
    //         ig_md.pkt_embeded_2.l16_1 = access_pkt_embed_2_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_3_1() {
    //         ig_md.pkt_embeded_3.l16_1 = access_pkt_embed_3_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_4_1() {
    //         ig_md.pkt_embeded_4.l16_1 = access_pkt_embed_4_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_5_1() {
    //         ig_md.pkt_embeded_5.l16_1 = access_pkt_embed_5_1.execute(ig_md.flow_index);
    // }
    // action access_pkt_embeded_feature_6_1() {
    //         ig_md.pkt_embeded_6.l16_1 = access_pkt_embed_6_1.execute(ig_md.flow_index);
    // }
    
    // action act_swap0() {
    //     ig_md.temp.l4_1  = ig_md.pkt_embeded_1.l16_1[7:4];
    //     ig_md.temp.l4_2  = ig_md.pkt_embeded_1.l16_1[3:0];
    //     ig_md.temp.l4_3  = ig_md.pkt_embeded_2.l16_1[7:4];
    //     ig_md.temp.l4_4  = ig_md.pkt_embeded_2.l16_1[3:0];
    //     ig_md.temp.l4_5  = ig_md.pkt_embeded_3.l16_1[7:4];
    //     ig_md.temp.l4_6  = ig_md.pkt_embeded_3.l16_1[3:0];
    //     ig_md.temp.l4_7  = ig_md.pkt_embeded_4.l16_1[7:4];
    //     ig_md.temp.l4_8  = ig_md.pkt_embeded_4.l16_1[3:0];
    //     ig_md.temp.l4_9  = ig_md.pkt_embeded_5.l16_1[7:4];
    //     ig_md.temp.l4_10 = ig_md.pkt_embeded_5.l16_1[3:0];
    //     ig_md.temp.l4_11 = ig_md.pkt_embeded_6.l16_1[7:4];
    //     ig_md.temp.l4_12 = ig_md.pkt_embeded_6.l16_1[3:0];
    //     ig_md.temp.l4_13 = ig_md.pkt_embeded_0.l16_1[7:4];
    //     ig_md.temp.l4_14 = ig_md.pkt_embeded_0.l16_1[3:0];
    // }

    // action act_swap1() {
    //     ig_md.temp.l4_1  = ig_md.pkt_embeded_2.l16_1[7:4];
    //     ig_md.temp.l4_2  = ig_md.pkt_embeded_2.l16_1[3:0];
    //     ig_md.temp.l4_3  = ig_md.pkt_embeded_3.l16_1[7:4];
    //     ig_md.temp.l4_4  = ig_md.pkt_embeded_3.l16_1[3:0];
    //     ig_md.temp.l4_5  = ig_md.pkt_embeded_4.l16_1[7:4];
    //     ig_md.temp.l4_6  = ig_md.pkt_embeded_4.l16_1[3:0];
    //     ig_md.temp.l4_7  = ig_md.pkt_embeded_5.l16_1[7:4];
    //     ig_md.temp.l4_8  = ig_md.pkt_embeded_5.l16_1[3:0];
    //     ig_md.temp.l4_9  = ig_md.pkt_embeded_6.l16_1[7:4];
    //     ig_md.temp.l4_10 = ig_md.pkt_embeded_6.l16_1[3:0];
    //     ig_md.temp.l4_11 = ig_md.pkt_embeded_0.l16_1[7:4];
    //     ig_md.temp.l4_12 = ig_md.pkt_embeded_0.l16_1[3:0];
    //     ig_md.temp.l4_13 = ig_md.pkt_embeded_1.l16_1[7:4];
    //     ig_md.temp.l4_14 = ig_md.pkt_embeded_1.l16_1[3:0];
    // }

    // action act_swap2() {
    //     ig_md.temp.l4_1  = ig_md.pkt_embeded_3.l16_1[7:4];
    //     ig_md.temp.l4_2  = ig_md.pkt_embeded_3.l16_1[3:0];
    //     ig_md.temp.l4_3  = ig_md.pkt_embeded_4.l16_1[7:4];
    //     ig_md.temp.l4_4  = ig_md.pkt_embeded_4.l16_1[3:0];
    //     ig_md.temp.l4_5  = ig_md.pkt_embeded_5.l16_1[7:4];
    //     ig_md.temp.l4_6  = ig_md.pkt_embeded_5.l16_1[3:0];
    //     ig_md.temp.l4_7  = ig_md.pkt_embeded_6.l16_1[7:4];
    //     ig_md.temp.l4_8  = ig_md.pkt_embeded_6.l16_1[3:0];
    //     ig_md.temp.l4_9  = ig_md.pkt_embeded_0.l16_1[7:4];
    //     ig_md.temp.l4_10 = ig_md.pkt_embeded_0.l16_1[3:0];
    //     ig_md.temp.l4_11 = ig_md.pkt_embeded_1.l16_1[7:4];
    //     ig_md.temp.l4_12 = ig_md.pkt_embeded_1.l16_1[3:0];
    //     ig_md.temp.l4_13 = ig_md.pkt_embeded_2.l16_1[7:4];
    //     ig_md.temp.l4_14 = ig_md.pkt_embeded_2.l16_1[3:0];
    // }

    // action act_swap3() {
    //     ig_md.temp.l4_1  = ig_md.pkt_embeded_4.l16_1[7:4];
    //     ig_md.temp.l4_2  = ig_md.pkt_embeded_4.l16_1[3:0];
    //     ig_md.temp.l4_3  = ig_md.pkt_embeded_5.l16_1[7:4];
    //     ig_md.temp.l4_4  = ig_md.pkt_embeded_5.l16_1[3:0];
    //     ig_md.temp.l4_5  = ig_md.pkt_embeded_6.l16_1[7:4];
    //     ig_md.temp.l4_6  = ig_md.pkt_embeded_6.l16_1[3:0];
    //     ig_md.temp.l4_7  = ig_md.pkt_embeded_0.l16_1[7:4];
    //     ig_md.temp.l4_8  = ig_md.pkt_embeded_0.l16_1[3:0];
    //     ig_md.temp.l4_9  = ig_md.pkt_embeded_1.l16_1[7:4];
    //     ig_md.temp.l4_10 = ig_md.pkt_embeded_1.l16_1[3:0];
    //     ig_md.temp.l4_11 = ig_md.pkt_embeded_2.l16_1[7:4];
    //     ig_md.temp.l4_12 = ig_md.pkt_embeded_2.l16_1[3:0];
    //     ig_md.temp.l4_13 = ig_md.pkt_embeded_3.l16_1[7:4];
    //     ig_md.temp.l4_14 = ig_md.pkt_embeded_3.l16_1[3:0];
    // }
    // action act_swap4() {
    //     ig_md.temp.l4_1  = ig_md.pkt_embeded_5.l16_1[7:4];
    //     ig_md.temp.l4_2  = ig_md.pkt_embeded_5.l16_1[3:0];
    //     ig_md.temp.l4_3  = ig_md.pkt_embeded_6.l16_1[7:4];
    //     ig_md.temp.l4_4  = ig_md.pkt_embeded_6.l16_1[3:0];
    //     ig_md.temp.l4_5  = ig_md.pkt_embeded_0.l16_1[7:4];
    //     ig_md.temp.l4_6  = ig_md.pkt_embeded_0.l16_1[3:0];
    //     ig_md.temp.l4_7  = ig_md.pkt_embeded_1.l16_1[7:4];
    //     ig_md.temp.l4_8  = ig_md.pkt_embeded_1.l16_1[3:0];
    //     ig_md.temp.l4_9  = ig_md.pkt_embeded_2.l16_1[7:4];
    //     ig_md.temp.l4_10 = ig_md.pkt_embeded_2.l16_1[3:0];
    //     ig_md.temp.l4_11 = ig_md.pkt_embeded_3.l16_1[7:4];
    //     ig_md.temp.l4_12 = ig_md.pkt_embeded_3.l16_1[3:0];
    //     ig_md.temp.l4_13 = ig_md.pkt_embeded_4.l16_1[7:4];
    //     ig_md.temp.l4_14 = ig_md.pkt_embeded_4.l16_1[3:0];
    // }
    // action act_swap5() {
    //     ig_md.temp.l4_1  = ig_md.pkt_embeded_6.l16_1[7:4];
    //     ig_md.temp.l4_2  = ig_md.pkt_embeded_6.l16_1[3:0];
    //     ig_md.temp.l4_3  = ig_md.pkt_embeded_0.l16_1[7:4];
    //     ig_md.temp.l4_4  = ig_md.pkt_embeded_0.l16_1[3:0];
    //     ig_md.temp.l4_5  = ig_md.pkt_embeded_1.l16_1[7:4];
    //     ig_md.temp.l4_6  = ig_md.pkt_embeded_1.l16_1[3:0];
    //     ig_md.temp.l4_7  = ig_md.pkt_embeded_2.l16_1[7:4];
    //     ig_md.temp.l4_8  = ig_md.pkt_embeded_2.l16_1[3:0];
    //     ig_md.temp.l4_9  = ig_md.pkt_embeded_3.l16_1[7:4];
    //     ig_md.temp.l4_10 = ig_md.pkt_embeded_3.l16_1[3:0];
    //     ig_md.temp.l4_11 = ig_md.pkt_embeded_4.l16_1[7:4];
    //     ig_md.temp.l4_12 = ig_md.pkt_embeded_4.l16_1[3:0];
    //     ig_md.temp.l4_13 = ig_md.pkt_embeded_5.l16_1[7:4];
    //     ig_md.temp.l4_14 = ig_md.pkt_embeded_5.l16_1[3:0];
    // }
    // action act_swap6() {
    //     ig_md.temp.l4_1  = ig_md.pkt_embeded_0.l16_1[7:4];
    //     ig_md.temp.l4_2  = ig_md.pkt_embeded_0.l16_1[3:0];
    //     ig_md.temp.l4_3  = ig_md.pkt_embeded_1.l16_1[7:4];
    //     ig_md.temp.l4_4  = ig_md.pkt_embeded_1.l16_1[3:0];
    //     ig_md.temp.l4_5  = ig_md.pkt_embeded_2.l16_1[7:4];
    //     ig_md.temp.l4_6  = ig_md.pkt_embeded_2.l16_1[3:0];
    //     ig_md.temp.l4_7  = ig_md.pkt_embeded_3.l16_1[7:4];
    //     ig_md.temp.l4_8  = ig_md.pkt_embeded_3.l16_1[3:0];
    //     ig_md.temp.l4_9  = ig_md.pkt_embeded_4.l16_1[7:4];
    //     ig_md.temp.l4_10 = ig_md.pkt_embeded_4.l16_1[3:0];
    //     ig_md.temp.l4_11 = ig_md.pkt_embeded_5.l16_1[7:4];
    //     ig_md.temp.l4_12 = ig_md.pkt_embeded_5.l16_1[3:0];
    //     ig_md.temp.l4_13 = ig_md.pkt_embeded_6.l16_1[7:4];
    //     ig_md.temp.l4_14 = ig_md.pkt_embeded_6.l16_1[3:0];
    // }
    // @stage(7)
    // table tab_swap {
    //     size = 7;
    //     key = { ig_md.pkt_count_mod_7: exact; }
    //     actions = { act_swap0; act_swap1; act_swap2; act_swap3; act_swap4; act_swap5; act_swap6; }
    //     const entries = {
    //         (0):act_swap0(); (1):act_swap1(); (2):act_swap2(); (3):act_swap3(); (4):act_swap4(); (5):act_swap5(); (6):act_swap6();
    //     }
    //     const default_action = act_swap0();
    // }

    // ****cal cnn first
    action set_x8_1_2(bit<4> in01) {
        ig_md.v[7:4] = in01;
    }
    
    action set_x8_3_4(bit<4> in01) {
        ig_md.v[3:0] = in01;
    }
    @stage (2) //@stage (0)
    table x8_1_2 {
        key = {
            ig_md.hidden_7.l16_1: ternary;
        }
        actions = {
            set_x8_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (2) //@stage (0)
    table x8_3_4 {
        key = {
            ig_md.hidden_7.l16_2: ternary;
        }
        actions = {
            set_x8_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }

    // *****cal cnn second

    action set_x1_1_4(bit<16> in0, bit<16> in1, bit<16> in2, bit<16> in3, bit<16> in4, bit<16> in5) {
		ig_md.cnn_1_1.l16_1 = in0 + ig_md.cnn_1_1.l16_1;
		ig_md.cnn_1_2.l16_1 = in1 + ig_md.cnn_1_2.l16_1;
		ig_md.cnn_1_3.l16_1 = in2 + ig_md.cnn_1_3.l16_1;
		ig_md.cnn_2_1.l16_1 = in3 + ig_md.cnn_2_1.l16_1;
		ig_md.cnn_2_2.l16_1 = in4 + ig_md.cnn_2_2.l16_1;
		ig_md.cnn_2_3.l16_1 = in5 + ig_md.cnn_2_3.l16_1;

    }
    
    action set_x2_1_4(bit<16> in0, bit<16> in1, bit<16> in2, bit<16> in3, bit<16> in4, bit<16> in5, bit<16> in6, bit<16> in7, bit<16> in8, bit<16> in9, bit<16> in10, bit<16> in11) {
		ig_md.cnn_1_1.l16_1 = in0 + ig_md.cnn_1_1.l16_1;
		ig_md.cnn_1_1.l16_2 = in1 + ig_md.cnn_1_1.l16_2;
		ig_md.cnn_1_2.l16_1 = in2 + ig_md.cnn_1_2.l16_1;
		ig_md.cnn_1_2.l16_2 = in3 + ig_md.cnn_1_2.l16_2;
		ig_md.cnn_1_3.l16_1 = in4 + ig_md.cnn_1_3.l16_1;
		ig_md.cnn_1_3.l16_2 = in5 + ig_md.cnn_1_3.l16_2;
		ig_md.cnn_2_1.l16_1 = in6 + ig_md.cnn_2_1.l16_1;
		ig_md.cnn_2_1.l16_2 = in7 + ig_md.cnn_2_1.l16_2;
		ig_md.cnn_2_2.l16_1 = in8 + ig_md.cnn_2_2.l16_1;
		ig_md.cnn_2_2.l16_2 = in9 + ig_md.cnn_2_2.l16_2;
		ig_md.cnn_2_3.l16_1 = in10 + ig_md.cnn_2_3.l16_1;
		ig_md.cnn_2_3.l16_2 = in11 + ig_md.cnn_2_3.l16_2;

    }
    
    action set_x3_1_4(bit<16> in0, bit<16> in1, bit<16> in2, bit<16> in3, bit<16> in4, bit<16> in5, bit<16> in6, bit<16> in7, bit<16> in8, bit<16> in9, bit<16> in10, bit<16> in11, bit<16> in12, bit<16> in13, bit<16> in14, bit<16> in15, bit<16> in16, bit<16> in17) {
		ig_md.cnn_1_1.l16_1 = in0 + ig_md.cnn_1_1.l16_1;
		ig_md.cnn_1_1.l16_2 = in1 + ig_md.cnn_1_1.l16_2;
		ig_md.cnn_1_1.l16_3 = in2 + ig_md.cnn_1_1.l16_3;
		ig_md.cnn_1_2.l16_1 = in3 + ig_md.cnn_1_2.l16_1;
		ig_md.cnn_1_2.l16_2 = in4 + ig_md.cnn_1_2.l16_2;
		ig_md.cnn_1_2.l16_3 = in5 + ig_md.cnn_1_2.l16_3;
		ig_md.cnn_1_3.l16_1 = in6 + ig_md.cnn_1_3.l16_1;
		ig_md.cnn_1_3.l16_2 = in7 + ig_md.cnn_1_3.l16_2;
		ig_md.cnn_1_3.l16_3 = in8 + ig_md.cnn_1_3.l16_3;
		ig_md.cnn_2_1.l16_1 = in9 + ig_md.cnn_2_1.l16_1;
		ig_md.cnn_2_1.l16_2 = in10 + ig_md.cnn_2_1.l16_2;
		ig_md.cnn_2_1.l16_3 = in11 + ig_md.cnn_2_1.l16_3;
		ig_md.cnn_2_2.l16_1 = in12 + ig_md.cnn_2_2.l16_1;
		ig_md.cnn_2_2.l16_2 = in13 + ig_md.cnn_2_2.l16_2;
		ig_md.cnn_2_2.l16_3 = in14 + ig_md.cnn_2_2.l16_3;
		ig_md.cnn_2_3.l16_1 = in15 + ig_md.cnn_2_3.l16_1;
		ig_md.cnn_2_3.l16_2 = in16 + ig_md.cnn_2_3.l16_2;
		ig_md.cnn_2_3.l16_3 = in17 + ig_md.cnn_2_3.l16_3;

    }
    
    action set_x4_1_4(bit<16> in0, bit<16> in1, bit<16> in2, bit<16> in3, bit<16> in4, bit<16> in5, bit<16> in6, bit<16> in7, bit<16> in8, bit<16> in9, bit<16> in10, bit<16> in11, bit<16> in12, bit<16> in13, bit<16> in14, bit<16> in15, bit<16> in16, bit<16> in17, bit<16> in18, bit<16> in19, bit<16> in20, bit<16> in21) {
		ig_md.cnn_1_1.l16_2 = in0 + ig_md.cnn_1_1.l16_2;
		ig_md.cnn_1_1.l16_3 = in1 + ig_md.cnn_1_1.l16_3;
		ig_md.cnn_1_1.l16_4 = in2 + ig_md.cnn_1_1.l16_4;
		ig_md.cnn_1_2.l16_1 = in3 + ig_md.cnn_1_2.l16_1;
		ig_md.cnn_1_2.l16_2 = in4 + ig_md.cnn_1_2.l16_2;
		ig_md.cnn_1_2.l16_3 = in5 + ig_md.cnn_1_2.l16_3;
		ig_md.cnn_1_2.l16_4 = in6 + ig_md.cnn_1_2.l16_4;
		ig_md.cnn_1_3.l16_1 = in7 + ig_md.cnn_1_3.l16_1;
		ig_md.cnn_1_3.l16_2 = in8 + ig_md.cnn_1_3.l16_2;
		ig_md.cnn_1_3.l16_3 = in9 + ig_md.cnn_1_3.l16_3;
		ig_md.cnn_1_3.l16_4 = in10 + ig_md.cnn_1_3.l16_4;
		ig_md.cnn_2_1.l16_2 = in11 + ig_md.cnn_2_1.l16_2;
		ig_md.cnn_2_1.l16_3 = in12 + ig_md.cnn_2_1.l16_3;
		ig_md.cnn_2_1.l16_4 = in13 + ig_md.cnn_2_1.l16_4;
		ig_md.cnn_2_2.l16_1 = in14 + ig_md.cnn_2_2.l16_1;
		ig_md.cnn_2_2.l16_2 = in15 + ig_md.cnn_2_2.l16_2;
		ig_md.cnn_2_2.l16_3 = in16 + ig_md.cnn_2_2.l16_3;
		ig_md.cnn_2_2.l16_4 = in17 + ig_md.cnn_2_2.l16_4;
		ig_md.cnn_2_3.l16_1 = in18 + ig_md.cnn_2_3.l16_1;
		ig_md.cnn_2_3.l16_2 = in19 + ig_md.cnn_2_3.l16_2;
		ig_md.cnn_2_3.l16_3 = in20 + ig_md.cnn_2_3.l16_3;
		ig_md.cnn_2_3.l16_4 = in21 + ig_md.cnn_2_3.l16_4;

    }
    
    action set_x5_1_4(bit<16> in0, bit<16> in1, bit<16> in2, bit<16> in3, bit<16> in4, bit<16> in5, bit<16> in6, bit<16> in7, bit<16> in8, bit<16> in9, bit<16> in10, bit<16> in11, bit<16> in12, bit<16> in13, bit<16> in14, bit<16> in15, bit<16> in16, bit<16> in17, bit<16> in18, bit<16> in19, bit<16> in20, bit<16> in21) {
		ig_md.cnn_1_1.l16_3 = in0 + ig_md.cnn_1_1.l16_3;
		ig_md.cnn_1_1.l16_4 = in1 + ig_md.cnn_1_1.l16_4;
		ig_md.cnn_1_1.l16_5 = in2 + ig_md.cnn_1_1.l16_5;
		ig_md.cnn_1_2.l16_2 = in3 + ig_md.cnn_1_2.l16_2;
		ig_md.cnn_1_2.l16_3 = in4 + ig_md.cnn_1_2.l16_3;
		ig_md.cnn_1_2.l16_4 = in5 + ig_md.cnn_1_2.l16_4;
		ig_md.cnn_1_2.l16_5 = in6 + ig_md.cnn_1_2.l16_5;
		ig_md.cnn_1_3.l16_1 = in7 + ig_md.cnn_1_3.l16_1;
		ig_md.cnn_1_3.l16_2 = in8 + ig_md.cnn_1_3.l16_2;
		ig_md.cnn_1_3.l16_3 = in9 + ig_md.cnn_1_3.l16_3;
		ig_md.cnn_1_3.l16_4 = in10 + ig_md.cnn_1_3.l16_4;
		ig_md.cnn_2_1.l16_3 = in11 + ig_md.cnn_2_1.l16_3;
		ig_md.cnn_2_1.l16_4 = in12 + ig_md.cnn_2_1.l16_4;
		ig_md.cnn_2_1.l16_5 = in13 + ig_md.cnn_2_1.l16_5;
		ig_md.cnn_2_2.l16_2 = in14 + ig_md.cnn_2_2.l16_2;
		ig_md.cnn_2_2.l16_3 = in15 + ig_md.cnn_2_2.l16_3;
		ig_md.cnn_2_2.l16_4 = in16 + ig_md.cnn_2_2.l16_4;
		ig_md.cnn_2_2.l16_5 = in17 + ig_md.cnn_2_2.l16_5;
		ig_md.cnn_2_3.l16_1 = in18 + ig_md.cnn_2_3.l16_1;
		ig_md.cnn_2_3.l16_2 = in19 + ig_md.cnn_2_3.l16_2;
		ig_md.cnn_2_3.l16_3 = in20 + ig_md.cnn_2_3.l16_3;
		ig_md.cnn_2_3.l16_4 = in21 + ig_md.cnn_2_3.l16_4;

    }
    
    action set_x6_1_4(bit<16> in0, bit<16> in1, bit<16> in2, bit<16> in3, bit<16> in4, bit<16> in5, bit<16> in6, bit<16> in7, bit<16> in8, bit<16> in9, bit<16> in10, bit<16> in11, bit<16> in12, bit<16> in13, bit<16> in14, bit<16> in15, bit<16> in16, bit<16> in17) {
		ig_md.cnn_1_1.l16_4 = in0 + ig_md.cnn_1_1.l16_4;
		ig_md.cnn_1_1.l16_5 = in1 + ig_md.cnn_1_1.l16_5;
		ig_md.cnn_1_1.l16_6 = in2 + ig_md.cnn_1_1.l16_6;
		ig_md.cnn_1_2.l16_3 = in3 + ig_md.cnn_1_2.l16_3;
		ig_md.cnn_1_2.l16_4 = in4 + ig_md.cnn_1_2.l16_4;
		ig_md.cnn_1_2.l16_5 = in5 + ig_md.cnn_1_2.l16_5;
		ig_md.cnn_1_3.l16_2 = in6 + ig_md.cnn_1_3.l16_2;
		ig_md.cnn_1_3.l16_3 = in7 + ig_md.cnn_1_3.l16_3;
		ig_md.cnn_1_3.l16_4 = in8 + ig_md.cnn_1_3.l16_4;
		ig_md.cnn_2_1.l16_4 = in9 + ig_md.cnn_2_1.l16_4;
		ig_md.cnn_2_1.l16_5 = in10 + ig_md.cnn_2_1.l16_5;
		ig_md.cnn_2_1.l16_6 = in11 + ig_md.cnn_2_1.l16_6;
		ig_md.cnn_2_2.l16_3 = in12 + ig_md.cnn_2_2.l16_3;
		ig_md.cnn_2_2.l16_4 = in13 + ig_md.cnn_2_2.l16_4;
		ig_md.cnn_2_2.l16_5 = in14 + ig_md.cnn_2_2.l16_5;
		ig_md.cnn_2_3.l16_2 = in15 + ig_md.cnn_2_3.l16_2;
		ig_md.cnn_2_3.l16_3 = in16 + ig_md.cnn_2_3.l16_3;
		ig_md.cnn_2_3.l16_4 = in17 + ig_md.cnn_2_3.l16_4;

    }
    
    action set_x7_1_4(bit<16> in0, bit<16> in1, bit<16> in2, bit<16> in3, bit<16> in4, bit<16> in5, bit<16> in6, bit<16> in7, bit<16> in8, bit<16> in9, bit<16> in10, bit<16> in11) {
		ig_md.cnn_1_1.l16_5 = in0 + ig_md.cnn_1_1.l16_5;
		ig_md.cnn_1_1.l16_6 = in1 + ig_md.cnn_1_1.l16_6;
		ig_md.cnn_1_2.l16_4 = in2 + ig_md.cnn_1_2.l16_4;
		ig_md.cnn_1_2.l16_5 = in3 + ig_md.cnn_1_2.l16_5;
		ig_md.cnn_1_3.l16_3 = in4 + ig_md.cnn_1_3.l16_3;
		ig_md.cnn_1_3.l16_4 = in5 + ig_md.cnn_1_3.l16_4;
		ig_md.cnn_2_1.l16_5 = in6 + ig_md.cnn_2_1.l16_5;
		ig_md.cnn_2_1.l16_6 = in7 + ig_md.cnn_2_1.l16_6;
		ig_md.cnn_2_2.l16_4 = in8 + ig_md.cnn_2_2.l16_4;
		ig_md.cnn_2_2.l16_5 = in9 + ig_md.cnn_2_2.l16_5;
		ig_md.cnn_2_3.l16_3 = in10 + ig_md.cnn_2_3.l16_3;
		ig_md.cnn_2_3.l16_4 = in11 + ig_md.cnn_2_3.l16_4;

    }
    
    action set_x8_1_4(bit<16> in0, bit<16> in1, bit<16> in2, bit<16> in3, bit<16> in4, bit<16> in5) {
		ig_md.cnn_1_1.l16_6 = in0 + ig_md.cnn_1_1.l16_6;
		ig_md.cnn_1_2.l16_5 = in1 + ig_md.cnn_1_2.l16_5;
		ig_md.cnn_1_3.l16_4 = in2 + ig_md.cnn_1_3.l16_4;
		ig_md.cnn_2_1.l16_6 = in3 + ig_md.cnn_2_1.l16_6;
		ig_md.cnn_2_2.l16_5 = in4 + ig_md.cnn_2_2.l16_5;
		ig_md.cnn_2_3.l16_4 = in5 + ig_md.cnn_2_3.l16_4;

    }
    
    @stage (8) //@stage (1)
    table x1_1_4 {
        key = {
            ig_md.temp.l4_1: exact;
            ig_md.temp.l4_2: exact;
        }
        actions = {
            set_x1_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (10) //@stage (2)
    table x2_1_4 {
        key = {
            ig_md.temp.l4_3: exact;
            ig_md.temp.l4_4: exact;
        }
        actions = {
            set_x2_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (9) //@stage (3)
    table x3_1_4 {
        key = {
            ig_md.temp.l4_5: exact;
            ig_md.temp.l4_6: exact;
        }
        actions = {
            set_x3_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (12) //@stage (4)
    table x4_1_4 {
        key = {
            ig_md.temp.l4_7: exact;
            ig_md.temp.l4_8: exact;
        }
        actions = {
            set_x4_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (13) //@stage (5)
    table x5_1_4 {
        key = {
            ig_md.temp.l4_9: exact;
            ig_md.temp.l4_10: exact;
        }
        actions = {
            set_x5_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (8) //@stage (6)
    table x6_1_4 {
        key = {
            ig_md.temp.l4_11: exact;
            ig_md.temp.l4_12: exact;
        }
        actions = {
            set_x6_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (10) //@stage (2)
    table x7_1_4 {
        key = {
            ig_md.temp.l4_13: exact;
            ig_md.temp.l4_14: exact;
        }
        actions = {
            set_x7_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (9) //@stage (1)
    table x8_1_4 {
        key = {
            ig_md.temp.l4_15: exact;
            ig_md.temp.l4_16: exact;
        }
        actions = {
            set_x8_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
// *****cal fc first


    action set_cnn_1_1_1_2(bit<4> in01) {
        ig_md.temp.l4_1 = in01;
    }
    
    action set_cnn_1_1_3_4(bit<4> in01) {
        ig_md.temp.l4_2 = in01;
    }
    
    action set_cnn_1_1_5_6(bit<4> in01) {
        ig_md.temp.l4_3 = in01;
    }
    
    action set_cnn_2_1_1_2(bit<4> in01) {
        ig_md.temp.l4_4 = in01;
    }
    
    action set_cnn_2_1_3_4(bit<4> in01) {
        ig_md.temp.l4_5 = in01;
    }
    
    action set_cnn_2_1_5_6(bit<4> in01) {
        ig_md.temp.l4_6 = in01;
    }
    
    action set_cnn_1_2_1_2(bit<4> in01) {
        ig_md.temp.l4_7 = in01;
    }
    
    action set_cnn_1_2_3_4(bit<4> in01) {
        ig_md.temp.l4_8 = in01;
    }
    
    action set_cnn_2_2_1_2(bit<4> in01) {
        ig_md.temp.l4_9 = in01;
    }
    
    action set_cnn_2_2_3_4(bit<4> in01) {
        ig_md.temp.l4_10 = in01;
    }
    
    action set_cnn_1_2_2_5(bit<4> in01) {
        ig_md.temp.l4_11 = in01;
    }
    
    action set_cnn_1_3_1_2(bit<4> in01) {
        ig_md.temp.l4_12 = in01;
    }
    
    action set_cnn_1_3_3_4(bit<4> in01) {
        ig_md.temp.l4_13 = in01;
    }
    
    action set_cnn_2_3_1_2(bit<4> in01) {
        ig_md.temp.l4_14 = in01;
    }
    
    action set_cnn_2_3_3_4(bit<4> in01) {
        ig_md.temp.l4_15 = in01;
    }
    
    @stage (15) //@stage (7)
    table cnn_1_1_1_2 {
        key = {
            ig_md.cnn_1_1.l16_1: ternary;
            ig_md.cnn_1_1.l16_2: ternary;
        }
        actions = {
            set_cnn_1_1_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (15) //@stage (7)
    table cnn_1_1_3_4 {
        key = {
            ig_md.cnn_1_1.l16_3: ternary;
            ig_md.cnn_1_1.l16_4: ternary;
        }
        actions = {
            set_cnn_1_1_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (15) //@stage (7)
    table cnn_1_1_5_6 {
        key = {
            ig_md.cnn_1_1.l16_5: ternary;
            ig_md.cnn_1_1.l16_6: ternary;
        }
        actions = {
            set_cnn_1_1_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (15) //@stage (7)
    table cnn_2_1_1_2 {
        key = {
            ig_md.cnn_2_1.l16_1: ternary;
            ig_md.cnn_2_1.l16_2: ternary;
        }
        actions = {
            set_cnn_2_1_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (15) //@stage (7)
    table cnn_2_1_3_4 {
        key = {
            ig_md.cnn_2_1.l16_3: ternary;
            ig_md.cnn_2_1.l16_4: ternary;
        }
        actions = {
            set_cnn_2_1_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (15) //@stage (7)
    table cnn_2_1_5_6 {
        key = {
            ig_md.cnn_2_1.l16_5: ternary;
            ig_md.cnn_2_1.l16_6: ternary;
        }
        actions = {
            set_cnn_2_1_5_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (15) //@stage (7)
    table cnn_1_2_1_2 {
        key = {
            ig_md.cnn_1_2.l16_1: ternary;
            ig_md.cnn_1_2.l16_2: ternary;
        }
        actions = {
            set_cnn_1_2_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (15) //@stage (7)
    table cnn_1_2_3_4 {
        key = {
            ig_md.cnn_1_2.l16_3: ternary;
            ig_md.cnn_1_2.l16_4: ternary;
        }
        actions = {
            set_cnn_1_2_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16) //@stage (8)
    table cnn_2_2_1_2 {
        key = {
            ig_md.cnn_1_2.l16_5: ternary;
            ig_md.cnn_2_2.l16_1: ternary;
        }
        actions = {
            set_cnn_2_2_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16) //@stage (8)
    table cnn_2_2_3_4 {
        key = {
            ig_md.cnn_2_2.l16_2: ternary;
            ig_md.cnn_2_2.l16_3: ternary;
        }
        actions = {
            set_cnn_2_2_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16) //@stage (8)
    table cnn_1_2_2_5 {
        key = {
            ig_md.cnn_2_2.l16_4: ternary;
            ig_md.cnn_2_2.l16_5: ternary;
        }
        actions = {
            set_cnn_1_2_2_5;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16) //@stage (8)
    table cnn_1_3_1_2 {
        key = {
            ig_md.cnn_1_3.l16_1: ternary;
            ig_md.cnn_1_3.l16_2: ternary;
        }
        actions = {
            set_cnn_1_3_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16) //@stage (8)
    table cnn_1_3_3_4 {
        key = {
            ig_md.cnn_1_3.l16_3: ternary;
            ig_md.cnn_1_3.l16_4: ternary;
        }
        actions = {
            set_cnn_1_3_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16) //@stage (8)
    table cnn_2_3_1_2 {
        key = {
            ig_md.cnn_2_3.l16_1: ternary;
            ig_md.cnn_2_3.l16_2: ternary;
        }
        actions = {
            set_cnn_2_3_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16) //@stage (8)
    table cnn_2_3_3_4 {
        key = {
            ig_md.cnn_2_3.l16_3: ternary;
            ig_md.cnn_2_3.l16_4: ternary;
        }
        actions = {
            set_cnn_2_3_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    

// cal fc second


    action set_get_fc_1(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01;
        hdr.output.linear1.l32_2 = in02;
        hdr.output.linear1.l32_3 = in03;
        hdr.output.linear1.l32_4 = in04;
    }
    
    action set_get_fc_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear1.l32_1 = in01;
        ig_md.linear1.l32_2 = in02;
        ig_md.linear1.l32_3 = in03;
        ig_md.linear1.l32_4 = in04;
    }
    
    action set_get_fc_3(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    action set_get_fc_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear1.l32_1 = in01 + ig_md.linear1.l32_1;
        ig_md.linear1.l32_2 = in02 + ig_md.linear1.l32_2;
        ig_md.linear1.l32_3 = in03 + ig_md.linear1.l32_3;
        ig_md.linear1.l32_4 = in04 + ig_md.linear1.l32_4;
    }
    
    action set_get_fc_5(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    @stage (16) //@stage (8)
    table get_fc_1 {
        key = {
            ig_md.temp.l4_1: exact;
            ig_md.temp.l4_2: exact;
            ig_md.temp.l4_3: exact;
        }
        actions = {
            set_get_fc_1;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16) //@stage (8)
    table get_fc_2 {
        key = {
            ig_md.temp.l4_4: exact;
            ig_md.temp.l4_5: exact;
            ig_md.temp.l4_6: exact;
        }
        actions = {
            set_get_fc_2;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (17) //@stage (9)
    table get_fc_3 {
        key = {
            ig_md.temp.l4_7: exact;
            ig_md.temp.l4_8: exact;
            ig_md.temp.l4_9: exact;
        }
        actions = {
            set_get_fc_3;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (17) //@stage (9)
    table get_fc_4 {
        key = {
            ig_md.temp.l4_10: exact;
            ig_md.temp.l4_11: exact;
            ig_md.temp.l4_12: exact;
        }
        actions = {
            set_get_fc_4;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (18) //@stage (18)
    table get_fc_5 {
        key = {
            ig_md.temp.l4_13: exact;
            ig_md.temp.l4_14: exact;
            ig_md.temp.l4_15: exact;
        }
        actions = {
            set_get_fc_5;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }

    apply
    {
        // *****获取嵌入层******
        emb_pkl.apply(); //stage0
        emb_ipd.apply(); //stage0
        ig_md.hidden_7.l16_1 = ig_md.hidden_7.l16_1 + ig_md.hidden_6.l16_1; //stage1
        ig_md.hidden_7.l16_2 = ig_md.hidden_7.l16_2 + ig_md.hidden_6.l16_2; //stage1
        // cal_flow_hash(); //stage0
        x8_1_2.apply();
        x8_3_4.apply();
        // if(hdr.ethernet.ether_type == ETHERTYPE_IPV4)
        // {
        //     if(ig_md.src_port != 68)
        //     {
        //         ig_tm_md.ucast_egress_port = 0x120; //stage0
        //     }
        // }
        // Update_full_flow_hash(); //stage2
        // if(ig_md.flow_hash != ig_md.flow_hash1){
        //     Init_pkt_count(); //stage3
        //     Init_pkt_count_mod_7(); //stage3
        // }
        // else{
        //     Update_pkt_count(); //stage3
        //     Update_pkt_count_mod_7(); //stage3
        // }
        // access_pkt_embeded_feature_0_1();
        // access_pkt_embeded_feature_1_1();
        // access_pkt_embeded_feature_2_1();
        // access_pkt_embeded_feature_3_1();
        // access_pkt_embeded_feature_4_1();
        // access_pkt_embeded_feature_5_1();
        // access_pkt_embeded_feature_6_1();
        // tab_swap.apply();
        // *****cal cnn
            x1_1_4.apply();
            x6_1_4.apply();
            x3_1_4.apply();
            x8_1_4.apply();
            x2_1_4.apply();
            x7_1_4.apply();
            x4_1_4.apply();
            x5_1_4.apply();
            

    // ****cal fc
            cnn_1_1_1_2.apply();
            cnn_1_1_3_4.apply();
            cnn_1_1_5_6.apply();
            cnn_2_1_1_2.apply();
            cnn_2_1_3_4.apply();
            cnn_2_1_5_6.apply();
            cnn_1_2_1_2.apply();
            cnn_1_2_3_4.apply();
            cnn_2_2_1_2.apply();
            cnn_2_2_3_4.apply();
            cnn_1_2_2_5.apply();
            cnn_1_3_1_2.apply();
            cnn_1_3_3_4.apply();
            cnn_2_3_1_2.apply();
            cnn_2_3_3_4.apply();
            get_fc_1.apply();
            get_fc_2.apply();
            get_fc_3.apply();
            get_fc_4.apply();
            get_fc_5.apply();

            hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + ig_md.linear1.l32_1;
            hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + ig_md.linear1.l32_2;
            hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + ig_md.linear1.l32_3;
            hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + ig_md.linear1.l32_4;

            hdr.output.setValid();

            hdr.ethernet.dst_addr = 0; //for filter
            ig_tm_md.bypass_egress = 1;
            if(ig_md.pkt_count > 7) ig_dprsr_md.drop_ctl = 1; //stage4
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;