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
            hdr.feature.b_1.l8_1: ternary;
            hdr.feature.b_1.l8_2: ternary;
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
            hdr.feature.b_2.l8_1: ternary;
            hdr.feature.b_2.l8_2: ternary;
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
            hdr.feature.b_3.l8_1: ternary;
            hdr.feature.b_3.l8_2: ternary;
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
            hdr.feature.b_4.l8_1: ternary;
            hdr.feature.b_4.l8_2: ternary;
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
            hdr.feature.b_5.l8_1: ternary;
            hdr.feature.b_5.l8_2: ternary;
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
            hdr.feature.b_6.l8_1: ternary;
            hdr.feature.b_6.l8_2: ternary;
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
            hdr.feature.b_7.l8_1: ternary;
            hdr.feature.b_7.l8_2: ternary;
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
            hdr.feature.b_8.l8_1: ternary;
            hdr.feature.b_8.l8_2: ternary;
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
            hdr.feature.b_9.l8_1: ternary;
            hdr.feature.b_9.l8_2: ternary;
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
            hdr.feature.b_10.l8_1: ternary;
            hdr.feature.b_10.l8_2: ternary;
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
            hdr.feature.b_28.l8_2: ternary;
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
            hdr.feature.b_29.l8_1: ternary;
            hdr.feature.b_29.l8_2: ternary;
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
            hdr.feature.b_30.l8_1: ternary;
            hdr.feature.b_30.l8_2: ternary;
        }
        actions = {
            set_p1_30;
            noaction;
        }
        size = 1024;
        default_action = noaction();
    }

    action set_p2_1(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_1.l16_1 = in01;
        ig_md.s_1.l16_2 = in02;
        ig_md.s_1.l16_3 = in03;
        ig_md.s_1.l16_4 = in04;
    }
    action set_p2_2(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_2.l16_1 = in01;
        ig_md.s_2.l16_2 = in02;
        ig_md.s_2.l16_3 = in03;
        ig_md.s_2.l16_4 = in04;
    }
    action set_p2_3(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_3.l16_1 = in01;
        ig_md.s_3.l16_2 = in02;
        ig_md.s_3.l16_3 = in03;
        ig_md.s_3.l16_4 = in04;
    }
    action set_p2_4(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_4.l16_1 = in01;
        ig_md.s_4.l16_2 = in02;
        ig_md.s_4.l16_3 = in03;
        ig_md.s_4.l16_4 = in04;
    }
    action set_p2_5(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_1.l16_1 = in01 + ig_md.s_1.l16_1;
        ig_md.s_1.l16_2 = in02 + ig_md.s_1.l16_2;
        ig_md.s_1.l16_3 = in03 + ig_md.s_1.l16_3;
        ig_md.s_1.l16_4 = in04 + ig_md.s_1.l16_4;
    }
    action set_p2_6(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_2.l16_1 = in01 + ig_md.s_2.l16_1;
        ig_md.s_2.l16_2 = in02 + ig_md.s_2.l16_2;
        ig_md.s_2.l16_3 = in03 + ig_md.s_2.l16_3;
        ig_md.s_2.l16_4 = in04 + ig_md.s_2.l16_4;
    }
    action set_p2_7(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_3.l16_1 = in01 + ig_md.s_3.l16_1;
        ig_md.s_3.l16_2 = in02 + ig_md.s_3.l16_2;
        ig_md.s_3.l16_3 = in03 + ig_md.s_3.l16_3;
        ig_md.s_3.l16_4 = in04 + ig_md.s_3.l16_4;
    }
    action set_p2_8(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_4.l16_1 = in01 + ig_md.s_4.l16_1;
        ig_md.s_4.l16_2 = in02 + ig_md.s_4.l16_2;
        ig_md.s_4.l16_3 = in03 + ig_md.s_4.l16_3;
        ig_md.s_4.l16_4 = in04 + ig_md.s_4.l16_4;
    }
    action set_p2_9(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_1.l16_1 = in01 + ig_md.s_1.l16_1;
        ig_md.s_1.l16_2 = in02 + ig_md.s_1.l16_2;
        ig_md.s_1.l16_3 = in03 + ig_md.s_1.l16_3;
        ig_md.s_1.l16_4 = in04 + ig_md.s_1.l16_4;
    }
    action set_p2_10(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04) {
        ig_md.s_2.l16_1 = in01 + ig_md.s_2.l16_1;
        ig_md.s_2.l16_2 = in02 + ig_md.s_2.l16_2;
        ig_md.s_2.l16_3 = in03 + ig_md.s_2.l16_3;
        ig_md.s_2.l16_4 = in04 + ig_md.s_2.l16_4;
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
        ig_md.temp.l4_15 = in01;
    }
    action set_p3_2(bit<4> in01) {
        ig_md.temp.l4_16 = in01;
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
    table p3_2 {
        key = {
            ig_md.s_1.l16_3: ternary;
            ig_md.s_1.l16_4: ternary;
        }
        actions = {
            set_p3_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
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
    action set_x4_4(bit<32> in01, bit<32> in02, bit<32> in03) {
        ig_md.linear1.l32_1 = in01 + ig_md.linear1.l32_1;
        ig_md.linear1.l32_2 = in02 + ig_md.linear1.l32_2;
        ig_md.linear1.l32_3 = in03 + ig_md.linear1.l32_3;
    }
    action set_x4_5(bit<32> in01, bit<32> in02, bit<32> in03) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
    }
    action set_x4_6(bit<32> in01, bit<32> in02, bit<32> in03) {
        ig_md.linear1.l32_1 = in01 + ig_md.linear1.l32_1;
        ig_md.linear1.l32_2 = in02 + ig_md.linear1.l32_2;
        ig_md.linear1.l32_3 = in03 + ig_md.linear1.l32_3;
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
            ig_md.temp.l4_9: exact;
        }
        actions = {
            set_x4_3;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table x4_4 {
        key = {
            ig_md.temp.l4_10: exact;
            ig_md.temp.l4_11: exact;
            ig_md.temp.l4_12: exact;
        }
        actions = {
            set_x4_4;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table x4_5 {
        key = {
            ig_md.temp.l4_13: exact;
            ig_md.temp.l4_14: exact;
            ig_md.temp.l4_15: exact;
        }
        actions = {
            set_x4_5;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    table x4_6 {
        key = {
            ig_md.temp.l4_16: exact;
        }
        actions = {
            set_x4_6;
            noaction;
        }
        size = 16;
        default_action = noaction();
        idle_timeout = false;
    }
    apply
    {
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
        ig_md.s_1.l16_3 = ig_md.s_1.l16_3 + ig_md.s_3.l16_3;
        ig_md.s_1.l16_4 = ig_md.s_1.l16_4 + ig_md.s_3.l16_4;
        ig_md.s_2.l16_1 = ig_md.s_2.l16_1 + ig_md.s_4.l16_1;
        ig_md.s_2.l16_2 = ig_md.s_2.l16_2 + ig_md.s_4.l16_2;
        ig_md.s_2.l16_3 = ig_md.s_2.l16_3 + ig_md.s_4.l16_3;
        ig_md.s_2.l16_4 = ig_md.s_2.l16_4 + ig_md.s_4.l16_4;

        ig_md.s_1.l16_1 = ig_md.s_1.l16_1 + ig_md.s_2.l16_1;
        ig_md.s_1.l16_2 = ig_md.s_1.l16_2 + ig_md.s_2.l16_2;
        ig_md.s_1.l16_3 = ig_md.s_1.l16_3 + ig_md.s_2.l16_3;
        ig_md.s_1.l16_4 = ig_md.s_1.l16_4 + ig_md.s_2.l16_4;

        p3_1.apply();
        p3_2.apply();

        x4_1.apply();
        x4_2.apply();
        x4_3.apply();
        x4_4.apply();
        x4_5.apply();
        x4_6.apply();

        hdr.output.linear1.l32_1 = ig_md.linear1.l32_1 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = ig_md.linear1.l32_2 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = ig_md.linear1.l32_3 + hdr.output.linear1.l32_3;

        hdr.output.setValid();

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