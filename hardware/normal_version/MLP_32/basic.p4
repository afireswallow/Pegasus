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

// ****cal h1

    action set_ip_len_total_len(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        ig_md.linear1.l32_1 = in01;
        ig_md.linear1.l32_2 = in02;
        ig_md.linear1.l32_3 = in03;
        ig_md.linear1.l32_4 = in04;
        ig_md.linear1.l32_5 = in05;
        ig_md.linear1.l32_6 = in06;
        ig_md.linear1.l32_7 = in07;
        ig_md.linear1.l32_8 = in08;

        ig_md.linear2.l32_1 = in11;
        ig_md.linear2.l32_2 = in12;
        ig_md.linear2.l32_3 = in13;
        ig_md.linear2.l32_4 = in14;
        ig_md.linear2.l32_5 = in15;
        ig_md.linear2.l32_6 = in16;
        ig_md.linear2.l32_7 = in17;
        ig_md.linear2.l32_8 = in18;
    }
    
    action set_protocol_tos(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        ig_md.linear3.l32_1 = in01;
        ig_md.linear3.l32_2 = in02;
        ig_md.linear3.l32_3 = in03;
        ig_md.linear3.l32_4 = in04;
        ig_md.linear3.l32_5 = in05;
        ig_md.linear3.l32_6 = in06;
        ig_md.linear3.l32_7 = in07;
        ig_md.linear3.l32_8 = in08;

        ig_md.linear4.l32_1 = in11;
        ig_md.linear4.l32_2 = in12;
        ig_md.linear4.l32_3 = in13;
        ig_md.linear4.l32_4 = in14;
        ig_md.linear4.l32_5 = in15;
        ig_md.linear4.l32_6 = in16;
        ig_md.linear4.l32_7 = in17;
        ig_md.linear4.l32_8 = in18;
    }
    
    action set_ttl_offset(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        ig_md.linear1.l32_1 = in01 + ig_md.linear1.l32_1;
        ig_md.linear1.l32_2 = in02 + ig_md.linear1.l32_2;
        ig_md.linear1.l32_3 = in03 + ig_md.linear1.l32_3;
        ig_md.linear1.l32_4 = in04 + ig_md.linear1.l32_4;
        ig_md.linear1.l32_5 = in05 + ig_md.linear1.l32_5;
        ig_md.linear1.l32_6 = in06 + ig_md.linear1.l32_6;
        ig_md.linear1.l32_7 = in07 + ig_md.linear1.l32_7;
        ig_md.linear1.l32_8 = in08 + ig_md.linear1.l32_8;

        ig_md.linear2.l32_1 = in11 + ig_md.linear2.l32_1;
        ig_md.linear2.l32_2 = in12 + ig_md.linear2.l32_2;
        ig_md.linear2.l32_3 = in13 + ig_md.linear2.l32_3;
        ig_md.linear2.l32_4 = in14 + ig_md.linear2.l32_4;
        ig_md.linear2.l32_5 = in15 + ig_md.linear2.l32_5;
        ig_md.linear2.l32_6 = in16 + ig_md.linear2.l32_6;
        ig_md.linear2.l32_7 = in17 + ig_md.linear2.l32_7;
        ig_md.linear2.l32_8 = in18 + ig_md.linear2.l32_8;
    }
    
    action set_max_min_byte(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        ig_md.linear3.l32_1 = in01 + ig_md.linear3.l32_1;
        ig_md.linear3.l32_2 = in02 + ig_md.linear3.l32_2;
        ig_md.linear3.l32_3 = in03 + ig_md.linear3.l32_3;
        ig_md.linear3.l32_4 = in04 + ig_md.linear3.l32_4;
        ig_md.linear3.l32_5 = in05 + ig_md.linear3.l32_5;
        ig_md.linear3.l32_6 = in06 + ig_md.linear3.l32_6;
        ig_md.linear3.l32_7 = in07 + ig_md.linear3.l32_7;
        ig_md.linear3.l32_8 = in08 + ig_md.linear3.l32_8;

        ig_md.linear4.l32_1 = in11 + ig_md.linear4.l32_1;
        ig_md.linear4.l32_2 = in12 + ig_md.linear4.l32_2;
        ig_md.linear4.l32_3 = in13 + ig_md.linear4.l32_3;
        ig_md.linear4.l32_4 = in14 + ig_md.linear4.l32_4;
        ig_md.linear4.l32_5 = in15 + ig_md.linear4.l32_5;
        ig_md.linear4.l32_6 = in16 + ig_md.linear4.l32_6;
        ig_md.linear4.l32_7 = in17 + ig_md.linear4.l32_7;
        ig_md.linear4.l32_8 = in18 + ig_md.linear4.l32_8;
    }
    
    action set_max_min_ipd(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        ig_md.linear1.l32_1 = in01 + ig_md.linear1.l32_1;
        ig_md.linear1.l32_2 = in02 + ig_md.linear1.l32_2;
        ig_md.linear1.l32_3 = in03 + ig_md.linear1.l32_3;
        ig_md.linear1.l32_4 = in04 + ig_md.linear1.l32_4;
        ig_md.linear1.l32_5 = in05 + ig_md.linear1.l32_5;
        ig_md.linear1.l32_6 = in06 + ig_md.linear1.l32_6;
        ig_md.linear1.l32_7 = in07 + ig_md.linear1.l32_7;
        ig_md.linear1.l32_8 = in08 + ig_md.linear1.l32_8;

        ig_md.linear2.l32_1 = in11 + ig_md.linear2.l32_1;
        ig_md.linear2.l32_2 = in12 + ig_md.linear2.l32_2;
        ig_md.linear2.l32_3 = in13 + ig_md.linear2.l32_3;
        ig_md.linear2.l32_4 = in14 + ig_md.linear2.l32_4;
        ig_md.linear2.l32_5 = in15 + ig_md.linear2.l32_5;
        ig_md.linear2.l32_6 = in16 + ig_md.linear2.l32_6;
        ig_md.linear2.l32_7 = in17 + ig_md.linear2.l32_7;
        ig_md.linear2.l32_8 = in18 + ig_md.linear2.l32_8;
    }
    
    action set_ipd(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        ig_md.linear3.l32_1 = in01 + ig_md.linear3.l32_1;
        ig_md.linear3.l32_2 = in02 + ig_md.linear3.l32_2;
        ig_md.linear3.l32_3 = in03 + ig_md.linear3.l32_3;
        ig_md.linear3.l32_4 = in04 + ig_md.linear3.l32_4;
        ig_md.linear3.l32_5 = in05 + ig_md.linear3.l32_5;
        ig_md.linear3.l32_6 = in06 + ig_md.linear3.l32_6;
        ig_md.linear3.l32_7 = in07 + ig_md.linear3.l32_7;
        ig_md.linear3.l32_8 = in08 + ig_md.linear3.l32_8;

        ig_md.linear4.l32_1 = in11 + ig_md.linear4.l32_1;
        ig_md.linear4.l32_2 = in12 + ig_md.linear4.l32_2;
        ig_md.linear4.l32_3 = in13 + ig_md.linear4.l32_3;
        ig_md.linear4.l32_4 = in14 + ig_md.linear4.l32_4;
        ig_md.linear4.l32_5 = in15 + ig_md.linear4.l32_5;
        ig_md.linear4.l32_6 = in16 + ig_md.linear4.l32_6;
        ig_md.linear4.l32_7 = in17 + ig_md.linear4.l32_7;
        ig_md.linear4.l32_8 = in18 + ig_md.linear4.l32_8;
    }
    
    @stage (0)
    table ip_len_total_len {
        key = {
            hdr.feature.ip_len: ternary;
            hdr.feature.ip_total_len: ternary;
        }
        actions = {
            set_ip_len_total_len;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (0)
    table protocol_tos {
        key = {
            hdr.feature.protocol: ternary;
            hdr.feature.tos: ternary;
        }
        actions = {
            set_protocol_tos;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (1)
    table ttl_offset {
        key = {
            hdr.feature.ttl: ternary;
            hdr.feature.offset: ternary;
        }
        actions = {
            set_ttl_offset;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (1)
    table max_min_byte {
        key = {
            hdr.feature.max_byte: ternary;
            hdr.feature.min_byte: ternary;
        }
        actions = {
            set_max_min_byte;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (2)
    table max_min_ipd {
        key = {
            hdr.feature.max_ipd: ternary;
            hdr.feature.min_ipd: ternary;
        }
        actions = {
            set_max_min_ipd;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (2)
    table ipd {
        key = {
            hdr.feature.ipd: ternary;
        }
        actions = {
            set_ipd;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
// ****cal h2 first

    action set_h1_1(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        ig_md.linear3.l32_1 = in01;
        ig_md.linear3.l32_2 = in02;
        ig_md.linear3.l32_3 = in03;
        ig_md.linear3.l32_4 = in04;
        ig_md.linear3.l32_5 = in05;
        ig_md.linear3.l32_6 = in06;
        ig_md.linear3.l32_7 = in07;
        ig_md.linear3.l32_8 = in08;
    }
    
    action set_h1_2(bit<4> in01) {
        ig_md.temp.l4_1 = in01;
    }
    
    action set_h1_3(bit<4> in01) {
        ig_md.temp.l4_2 = in01;
    }
    
    action set_h1_4(bit<4> in01) {
        ig_md.temp.l4_3 = in01;
    }
    
    action set_h1_5(bit<4> in01) {
        ig_md.temp.l4_4 = in01;
    }
    
    action set_h1_6(bit<4> in01) {
        ig_md.temp.l4_5 = in01;
    }
    
    action set_h1_7(bit<4> in01) {
        ig_md.temp.l4_6 = in01;
    }
    
    action set_h1_8(bit<4> in01) {
        ig_md.temp.l4_7 = in01;
    }
    
    action set_h1_9(bit<4> in01) {
        ig_md.temp.l4_8 = in01;
    }
    
    action set_h1_10(bit<4> in01) {
        ig_md.temp.l4_9 = in01;
    }
    
    action set_h1_11(bit<4> in01) {
        ig_md.temp.l4_10 = in01;
    }
    
    action set_h1_12(bit<4> in01) {
        ig_md.temp.l4_11 = in01;
    }
    
    action set_h1_13(bit<4> in01) {
        ig_md.temp.l4_12 = in01;
    }
    
    action set_h1_14(bit<4> in01) {
        ig_md.temp.l4_13 = in01;
    }
    
    action set_h1_15(bit<4> in01) {
        ig_md.temp.l4_14 = in01;
    }
    
    action set_h1_16(bit<4> in01) {
        ig_md.temp.l4_15 = in01;
    }
    
    @stage (5)
    table h1_1 {
        key = {
            ig_md.linear1.l32_1: ternary;
        }
        actions = {
            set_h1_1;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (5)
    table h1_2 {
        key = {
            ig_md.linear1.l32_2: ternary;
        }
        actions = {
            set_h1_2;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (5)
    table h1_3 {
        key = {
            ig_md.linear1.l32_3: ternary;
        }
        actions = {
            set_h1_3;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (5)
    table h1_4 {
        key = {
            ig_md.linear1.l32_4: ternary;
        }
        actions = {
            set_h1_4;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (5)
    table h1_5 {
        key = {
            ig_md.linear1.l32_5: ternary;
        }
        actions = {
            set_h1_5;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (5)
    table h1_6 {
        key = {
            ig_md.linear1.l32_6: ternary;
        }
        actions = {
            set_h1_6;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (5)
    table h1_7 {
        key = {
            ig_md.linear1.l32_7: ternary;
        }
        actions = {
            set_h1_7;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (5)
    table h1_8 {
        key = {
            ig_md.linear1.l32_8: ternary;
        }
        actions = {
            set_h1_8;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (6)
    table h1_9 {
        key = {
            ig_md.linear2.l32_1: ternary;
        }
        actions = {
            set_h1_9;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (6)
    table h1_10 {
        key = {
            ig_md.linear2.l32_2: ternary;
        }
        actions = {
            set_h1_10;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (6)
    table h1_11 {
        key = {
            ig_md.linear2.l32_3: ternary;
        }
        actions = {
            set_h1_11;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (6)
    table h1_12 {
        key = {
            ig_md.linear2.l32_4: ternary;
        }
        actions = {
            set_h1_12;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (6)
    table h1_13 {
        key = {
            ig_md.linear2.l32_5: ternary;
        }
        actions = {
            set_h1_13;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (6)
    table h1_14 {
        key = {
            ig_md.linear2.l32_6: ternary;
        }
        actions = {
            set_h1_14;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (6)
    table h1_15 {
        key = {
            ig_md.linear2.l32_7: ternary;
        }
        actions = {
            set_h1_15;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (6)
    table h1_16 {
        key = {
            ig_md.linear2.l32_8: ternary;
        }
        actions = {
            set_h1_16;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
// ****cal h2 second

    action set_h1_2_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        ig_md.linear3.l32_1 = in01 + ig_md.linear3.l32_1;
        ig_md.linear3.l32_2 = in02 + ig_md.linear3.l32_2;
        ig_md.linear3.l32_3 = in03 + ig_md.linear3.l32_3;
        ig_md.linear3.l32_4 = in04 + ig_md.linear3.l32_4;
        ig_md.linear3.l32_5 = in05 + ig_md.linear3.l32_5;
        ig_md.linear3.l32_6 = in06 + ig_md.linear3.l32_6;
        ig_md.linear3.l32_7 = in07 + ig_md.linear3.l32_7;
        ig_md.linear3.l32_8 = in08 + ig_md.linear3.l32_8;
    }
    
    action set_h1_5_7(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        ig_md.linear4.l32_1 = in01;
        ig_md.linear4.l32_2 = in02;
        ig_md.linear4.l32_3 = in03;
        ig_md.linear4.l32_4 = in04;
        ig_md.linear4.l32_5 = in05;
        ig_md.linear4.l32_6 = in06;
        ig_md.linear4.l32_7 = in07;
        ig_md.linear4.l32_8 = in08;
    }
    
    action set_h1_8_10(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        ig_md.linear3.l32_1 = in01 + ig_md.linear3.l32_1;
        ig_md.linear3.l32_2 = in02 + ig_md.linear3.l32_2;
        ig_md.linear3.l32_3 = in03 + ig_md.linear3.l32_3;
        ig_md.linear3.l32_4 = in04 + ig_md.linear3.l32_4;
        ig_md.linear3.l32_5 = in05 + ig_md.linear3.l32_5;
        ig_md.linear3.l32_6 = in06 + ig_md.linear3.l32_6;
        ig_md.linear3.l32_7 = in07 + ig_md.linear3.l32_7;
        ig_md.linear3.l32_8 = in08 + ig_md.linear3.l32_8;
    }
    
    action set_h1_11_13(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        ig_md.linear4.l32_1 = in01 + ig_md.linear4.l32_1;
        ig_md.linear4.l32_2 = in02 + ig_md.linear4.l32_2;
        ig_md.linear4.l32_3 = in03 + ig_md.linear4.l32_3;
        ig_md.linear4.l32_4 = in04 + ig_md.linear4.l32_4;
        ig_md.linear4.l32_5 = in05 + ig_md.linear4.l32_5;
        ig_md.linear4.l32_6 = in06 + ig_md.linear4.l32_6;
        ig_md.linear4.l32_7 = in07 + ig_md.linear4.l32_7;
        ig_md.linear4.l32_8 = in08 + ig_md.linear4.l32_8;
    }
    
    action set_h1_14_16(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        ig_md.linear3.l32_1 = in01 + ig_md.linear3.l32_1;
        ig_md.linear3.l32_2 = in02 + ig_md.linear3.l32_2;
        ig_md.linear3.l32_3 = in03 + ig_md.linear3.l32_3;
        ig_md.linear3.l32_4 = in04 + ig_md.linear3.l32_4;
        ig_md.linear3.l32_5 = in05 + ig_md.linear3.l32_5;
        ig_md.linear3.l32_6 = in06 + ig_md.linear3.l32_6;
        ig_md.linear3.l32_7 = in07 + ig_md.linear3.l32_7;
        ig_md.linear3.l32_8 = in08 + ig_md.linear3.l32_8;
    }
    
    @stage (7)
    table h1_2_4 {
        key = {
            ig_md.temp.l4_1: exact;
            ig_md.temp.l4_2: exact;
            ig_md.temp.l4_3: exact;
        }
        actions = {
            set_h1_2_4;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (7)
    table h1_5_7 {
        key = {
            ig_md.temp.l4_4: exact;
            ig_md.temp.l4_5: exact;
            ig_md.temp.l4_6: exact;
        }
        actions = {
            set_h1_5_7;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (8)
    table h1_8_10 {
        key = {
            ig_md.temp.l4_7: exact;
            ig_md.temp.l4_8: exact;
            ig_md.temp.l4_9: exact;
        }
        actions = {
            set_h1_8_10;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (8)
    table h1_11_13 {
        key = {
            ig_md.temp.l4_10: exact;
            ig_md.temp.l4_11: exact;
            ig_md.temp.l4_12: exact;
        }
        actions = {
            set_h1_11_13;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (9)
    table h1_14_16 {
        key = {
            ig_md.temp.l4_13: exact;
            ig_md.temp.l4_14: exact;
            ig_md.temp.l4_15: exact;
        }
        actions = {
            set_h1_14_16;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
// ****cal output

    action set_h2_1(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear2.l32_1 = in01;
        ig_md.linear2.l32_2 = in02;
        ig_md.linear2.l32_3 = in03;
        //ig_md.linear2.l32_4 = in04;
    }
    
    action set_h2_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01;
        hdr.output.linear1.l32_2 = in02;
        hdr.output.linear1.l32_3 = in03;
        //hdr.output.linear1.l32_4 = in04;
    }
    
    action set_h2_3(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear2.l32_1 = in01 + ig_md.linear2.l32_1;
        ig_md.linear2.l32_2 = in02 + ig_md.linear2.l32_2;
        ig_md.linear2.l32_3 = in03 + ig_md.linear2.l32_3;
        //ig_md.linear2.l32_4 = in04 + ig_md.linear2.l32_4;
    }
    
    action set_h2_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        //hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    action set_h2_5(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear2.l32_1 = in01 + ig_md.linear2.l32_1;
        ig_md.linear2.l32_2 = in02 + ig_md.linear2.l32_2;
        ig_md.linear2.l32_3 = in03 + ig_md.linear2.l32_3;
        //ig_md.linear2.l32_4 = in04 + ig_md.linear2.l32_4;
    }
    
    action set_h2_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        //hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    action set_h2_7(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        ig_md.linear2.l32_1 = in01 + ig_md.linear2.l32_1;
        ig_md.linear2.l32_2 = in02 + ig_md.linear2.l32_2;
        ig_md.linear2.l32_3 = in03 + ig_md.linear2.l32_3;
        //ig_md.linear2.l32_4 = in04 + ig_md.linear2.l32_4;
    }
    
    action set_h2_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        //hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    @stage (15)
    table h2_1 {
        key = {
            ig_md.linear3.l32_1: ternary;
        }
        actions = {
            set_h2_1;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (15)
    table h2_2 {
        key = {
            ig_md.linear3.l32_2: ternary;
        }
        actions = {
            set_h2_2;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16)
    table h2_3 {
        key = {
            ig_md.linear3.l32_3: ternary;
        }
        actions = {
            set_h2_3;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (16)
    table h2_4 {
        key = {
            ig_md.linear3.l32_4: ternary;
        }
        actions = {
            set_h2_4;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (17)
    table h2_5 {
        key = {
            ig_md.linear3.l32_5: ternary;
        }
        actions = {
            set_h2_5;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (17)
    table h2_6 {
        key = {
            ig_md.linear3.l32_6: ternary;
        }
        actions = {
            set_h2_6;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (18)
    table h2_7 {
        key = {
            ig_md.linear3.l32_7: ternary;
        }
        actions = {
            set_h2_7;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (18)
    table h2_8 {
        key = {
            ig_md.linear3.l32_8: ternary;
        }
        actions = {
            set_h2_8;
            noaction;
        }
        size = 2048;
        default_action = noaction();
        idle_timeout = false;
    }
    
// ************************    

    apply
    {
// *****cal h1 as linear1 linear2
        ip_len_total_len.apply();
        protocol_tos.apply();
        ttl_offset.apply();
        max_min_byte.apply();
        max_min_ipd.apply();
        ipd.apply();

        ig_md.linear1.l32_1 = ig_md.linear1.l32_1 + ig_md.linear3.l32_1;
        ig_md.linear1.l32_2 = ig_md.linear1.l32_2 + ig_md.linear3.l32_2;
        ig_md.linear1.l32_3 = ig_md.linear1.l32_3 + ig_md.linear3.l32_3;
        ig_md.linear1.l32_4 = ig_md.linear1.l32_4 + ig_md.linear3.l32_4;
        ig_md.linear1.l32_5 = ig_md.linear1.l32_5 + ig_md.linear3.l32_5;
        ig_md.linear1.l32_6 = ig_md.linear1.l32_6 + ig_md.linear3.l32_6;
        ig_md.linear1.l32_7 = ig_md.linear1.l32_7 + ig_md.linear3.l32_7;
        ig_md.linear1.l32_8 = ig_md.linear1.l32_8 + ig_md.linear3.l32_8;
        ig_md.linear2.l32_1 = ig_md.linear2.l32_1 + ig_md.linear4.l32_1;
        ig_md.linear2.l32_2 = ig_md.linear2.l32_2 + ig_md.linear4.l32_2;
        ig_md.linear2.l32_3 = ig_md.linear2.l32_3 + ig_md.linear4.l32_3;
        ig_md.linear2.l32_4 = ig_md.linear2.l32_4 + ig_md.linear4.l32_4;
        ig_md.linear2.l32_5 = ig_md.linear2.l32_5 + ig_md.linear4.l32_5;
        ig_md.linear2.l32_6 = ig_md.linear2.l32_6 + ig_md.linear4.l32_6;
        ig_md.linear2.l32_7 = ig_md.linear2.l32_7 + ig_md.linear4.l32_7;
        ig_md.linear2.l32_8 = ig_md.linear2.l32_8 + ig_md.linear4.l32_8;

// *****cal h2 as linear3
        h1_1.apply();
        h1_2.apply();
        h1_3.apply();
        h1_4.apply();
        h1_5.apply();
        h1_6.apply();
        h1_7.apply();
        h1_8.apply();
        h1_9.apply();
        h1_10.apply();
        h1_11.apply();
        h1_12.apply();
        h1_13.apply();
        h1_14.apply();
        h1_15.apply();
        h1_16.apply();

        h1_2_4.apply();
        h1_5_7.apply();
        h1_8_10.apply();
        h1_11_13.apply();
        h1_14_16.apply();

        ig_md.linear3.l32_1 = ig_md.linear3.l32_1 + ig_md.linear4.l32_1;
        ig_md.linear3.l32_2 = ig_md.linear3.l32_2 + ig_md.linear4.l32_2;
        ig_md.linear3.l32_3 = ig_md.linear3.l32_3 + ig_md.linear4.l32_3;
        ig_md.linear3.l32_4 = ig_md.linear3.l32_4 + ig_md.linear4.l32_4;
        ig_md.linear3.l32_5 = ig_md.linear3.l32_5 + ig_md.linear4.l32_5;
        ig_md.linear3.l32_6 = ig_md.linear3.l32_6 + ig_md.linear4.l32_6;
        ig_md.linear3.l32_7 = ig_md.linear3.l32_7 + ig_md.linear4.l32_7;
        ig_md.linear3.l32_8 = ig_md.linear3.l32_8 + ig_md.linear4.l32_8;

// *****cal output as linear3
        h2_1.apply();
        h2_2.apply();
        h2_3.apply();
        h2_4.apply();
        h2_5.apply();
        h2_6.apply();
        h2_7.apply();
        h2_8.apply();

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + ig_md.linear2.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + ig_md.linear2.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + ig_md.linear2.l32_3;
        //hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + ig_md.linear2.l32_4;

        hdr.output.setValid();

        ig_tm_md.ucast_egress_port = 288;
        ig_tm_md.bypass_egress = 1;

        hdr.ethernet.dst_addr = 0; //for filter
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;