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

    action set_emb_pkl(bit<16> in01, bit<16> in02) {
        hdr.feature.x_8.l16_1 = in01;
        hdr.feature.x_8.l16_2 = in02;
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
        size = 1501;
        default_action = noaction();
    }

    action set_emb_ipd(bit<16> in01, bit<16> in02) {
        hdr.feature.x_8.l16_1 = in01 + hdr.feature.x_8.l16_1;
        hdr.feature.x_8.l16_2 = in02 + hdr.feature.x_8.l16_2;
    }

    @stage (1)
    table emb_ipd {
        key = {
            hdr.feature.ipd: exact;
        }
        actions = {
            set_emb_ipd;
            noaction;
        }
        size = 2561;
        default_action = noaction();
    }

// ****cal cnn first


    action set_x1_1_2(bit<4> in01) {
        ig_md.temp.l4_1 = in01;
    }
    
    action set_x1_3_4(bit<4> in01) {
        ig_md.temp.l4_2 = in01;
    }
    
    action set_x2_1_2(bit<4> in01) {
        ig_md.temp.l4_3 = in01;
    }
    
    action set_x2_3_4(bit<4> in01) {
        ig_md.temp.l4_4 = in01;
    }
    
    action set_x3_1_2(bit<4> in01) {
        ig_md.temp.l4_5 = in01;
    }
    
    action set_x3_3_4(bit<4> in01) {
        ig_md.temp.l4_6 = in01;
    }
    
    action set_x4_1_2(bit<4> in01) {
        ig_md.temp.l4_7 = in01;
    }
    
    action set_x4_3_4(bit<4> in01) {
        ig_md.temp.l4_8 = in01;
    }
    
    action set_x5_1_2(bit<4> in01) {
        ig_md.temp.l4_9 = in01;
    }
    
    action set_x5_3_4(bit<4> in01) {
        ig_md.temp.l4_10 = in01;
    }
    
    action set_x6_1_2(bit<4> in01) {
        ig_md.temp.l4_11 = in01;
    }
    
    action set_x6_3_4(bit<4> in01) {
        ig_md.temp.l4_12 = in01;
    }
    
    action set_x7_1_2(bit<4> in01) {
        ig_md.temp.l4_13 = in01;
    }
    
    action set_x7_3_4(bit<4> in01) {
        ig_md.temp.l4_14 = in01;
    }
    
    action set_x8_1_2(bit<4> in01) {
        ig_md.temp.l4_15 = in01;
    }
    
    action set_x8_3_4(bit<4> in01) {
        ig_md.temp.l4_16 = in01;
    }
    
    @stage (0)
    table x1_1_2 {
        key = {
            hdr.feature.x_1.l16_1: ternary;
        }
        actions = {
            set_x1_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (0)
    table x1_3_4 {
        key = {
            hdr.feature.x_1.l16_2: ternary;
        }
        actions = {
            set_x1_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (0)
    table x2_1_2 {
        key = {
            hdr.feature.x_2.l16_1: ternary;
        }
        actions = {
            set_x2_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (0)
    table x2_3_4 {
        key = {
            hdr.feature.x_2.l16_2: ternary;
        }
        actions = {
            set_x2_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (0)
    table x3_1_2 {
        key = {
            hdr.feature.x_3.l16_1: ternary;
        }
        actions = {
            set_x3_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (0)
    table x3_3_4 {
        key = {
            hdr.feature.x_3.l16_2: ternary;
        }
        actions = {
            set_x3_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (0)
    table x4_1_2 {
        key = {
            hdr.feature.x_4.l16_1: ternary;
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
            hdr.feature.x_4.l16_2: ternary;
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
    table x5_1_2 {
        key = {
            hdr.feature.x_5.l16_1: ternary;
        }
        actions = {
            set_x5_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (1)
    table x5_3_4 {
        key = {
            hdr.feature.x_5.l16_2: ternary;
        }
        actions = {
            set_x5_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (1)
    table x6_1_2 {
        key = {
            hdr.feature.x_6.l16_1: ternary;
        }
        actions = {
            set_x6_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (1)
    table x6_3_4 {
        key = {
            hdr.feature.x_6.l16_2: ternary;
        }
        actions = {
            set_x6_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (1)
    table x7_1_2 {
        key = {
            hdr.feature.x_7.l16_1: ternary;
        }
        actions = {
            set_x7_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (1)
    table x7_3_4 {
        key = {
            hdr.feature.x_7.l16_2: ternary;
        }
        actions = {
            set_x7_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (1)
    table x8_1_2 {
        key = {
            hdr.feature.x_8.l16_1: ternary;
        }
        actions = {
            set_x8_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    @stage (1)
    table x8_3_4 {
        key = {
            hdr.feature.x_8.l16_2: ternary;
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

    action set_x1_1_4(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04, bit<16> in05, bit<16> in06) {
        hdr.output.s1.l16_1 = in01;
        hdr.output.s1.l16_2 = in02;
        hdr.output.s1.l16_3 = in03;
        hdr.output.s1.l16_4 = in04;
        hdr.output.s1.l16_5 = in05;
        hdr.output.s1.l16_6 = in06;
    }
    action set_x2_1_4(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04, bit<16> in05, bit<16> in06) {
        ig_md.s1.l16_1 = in01;
        ig_md.s1.l16_2 = in02;
        ig_md.s1.l16_3 = in03;
        ig_md.s1.l16_4 = in04;
        ig_md.s1.l16_5 = in05;
        ig_md.s1.l16_6 = in06;
    }
    action set_x3_1_4(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04, bit<16> in05, bit<16> in06) {
        ig_md.s2.l16_1 = in01;
        ig_md.s2.l16_2 = in02;
        ig_md.s2.l16_3 = in03;
        ig_md.s2.l16_4 = in04;
        ig_md.s2.l16_5 = in05;
        ig_md.s2.l16_6 = in06;
    }
    action set_x4_1_4(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04, bit<16> in05, bit<16> in06) {
        ig_md.s3.l16_1 = in01;
        ig_md.s3.l16_2 = in02;
        ig_md.s3.l16_3 = in03;
        ig_md.s3.l16_4 = in04;
        ig_md.s3.l16_5 = in05;
        ig_md.s3.l16_6 = in06;
    }
    action set_x5_1_4(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04, bit<16> in05, bit<16> in06) {
        hdr.output.s1.l16_1 = in01 + hdr.output.s1.l16_1;
        hdr.output.s1.l16_2 = in02 + hdr.output.s1.l16_2;
        hdr.output.s1.l16_3 = in03 + hdr.output.s1.l16_3;
        hdr.output.s1.l16_4 = in04 + hdr.output.s1.l16_4;
        hdr.output.s1.l16_5 = in05 + hdr.output.s1.l16_5;
        hdr.output.s1.l16_6 = in06 + hdr.output.s1.l16_6;
    }
    action set_x6_1_4(bit<16> in01, bit<16> in02, bit<16> in03, bit<16> in04, bit<16> in05, bit<16> in06) {
        ig_md.s1.l16_1 = in01 + ig_md.s1.l16_1;
        ig_md.s1.l16_2 = in02 + ig_md.s1.l16_2;
        ig_md.s1.l16_3 = in03 + ig_md.s1.l16_3;
        ig_md.s1.l16_4 = in04 + ig_md.s1.l16_4;
        ig_md.s1.l16_5 = in05 + ig_md.s1.l16_5;
        ig_md.s1.l16_6 = in06 + ig_md.s1.l16_6;
    }
    @stage (1)
    table x1_1_4 {
        key = {
            ig_md.temp.l4_1: exact;
            ig_md.temp.l4_2: exact;
            ig_md.temp.l4_3: exact;
        }
        actions = {
            set_x1_1_4;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (1)
    table x2_1_4 {
        key = {
            ig_md.temp.l4_4: exact;
            ig_md.temp.l4_5: exact;
            ig_md.temp.l4_6: exact;
        }
        actions = {
            set_x2_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (2)
    table x3_1_4 {
        key = {
            ig_md.temp.l4_7: exact;
            ig_md.temp.l4_8: exact;
            ig_md.temp.l4_9: exact;
        }
        actions = {
            set_x3_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (2)
    table x4_1_4 {
        key = {
            ig_md.temp.l4_10: exact;
            ig_md.temp.l4_11: exact;
            ig_md.temp.l4_12: exact;
        }
        actions = {
            set_x4_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (3)
    table x5_1_4 {
        key = {
            ig_md.temp.l4_13: exact;
            ig_md.temp.l4_14: exact;
            ig_md.temp.l4_15: exact;
        }
        actions = {
            set_x5_1_4;
            noaction;
        }
        size = 256;
        default_action = noaction();
        idle_timeout = false;
    }
    
    @stage (17)
    table x6_1_4 {
        key = {
            ig_md.temp.l4_16: exact;
        }
        actions = {
            set_x6_1_4;
            noaction;
        }
        size = 16;
        default_action = noaction();
        idle_timeout = false;
    }
// ************************    

    apply
    {
// *****获取嵌入层******
        emb_pkl.apply();
        emb_ipd.apply();
// *****cal cnn
        x1_1_2.apply();
        x1_3_4.apply();
        x2_1_2.apply();
        x2_3_4.apply();
        x3_1_2.apply();
        x3_3_4.apply();
        x4_1_2.apply();
        x4_3_4.apply();
        x5_1_2.apply();
        x5_3_4.apply();
        x6_1_2.apply();
        x6_3_4.apply();
        x7_1_2.apply();
        x7_3_4.apply();
        x8_1_2.apply();
        x8_3_4.apply();

        x1_1_4.apply();
        x2_1_4.apply();
        x3_1_4.apply();
        x4_1_4.apply();
        x5_1_4.apply();
        x6_1_4.apply();

        hdr.output.s1.l16_1 = ig_md.s2.l16_1 + hdr.output.s1.l16_1;
        hdr.output.s1.l16_2 = ig_md.s2.l16_2 + hdr.output.s1.l16_2;
        hdr.output.s1.l16_3 = ig_md.s2.l16_3 + hdr.output.s1.l16_3;
        hdr.output.s1.l16_4 = ig_md.s2.l16_4 + hdr.output.s1.l16_4;
        hdr.output.s1.l16_5 = ig_md.s2.l16_5 + hdr.output.s1.l16_5;
        hdr.output.s1.l16_6 = ig_md.s2.l16_6 + hdr.output.s1.l16_6;
        ig_md.s1.l16_1 = ig_md.s3.l16_1 + ig_md.s1.l16_1;
        ig_md.s1.l16_2 = ig_md.s3.l16_2 + ig_md.s1.l16_2;
        ig_md.s1.l16_3 = ig_md.s3.l16_3 + ig_md.s1.l16_3;
        ig_md.s1.l16_4 = ig_md.s3.l16_4 + ig_md.s1.l16_4;
        ig_md.s1.l16_5 = ig_md.s3.l16_5 + ig_md.s1.l16_5;
        ig_md.s1.l16_6 = ig_md.s3.l16_6 + ig_md.s1.l16_6;

        hdr.output.s1.l16_1 = ig_md.s1.l16_1 + hdr.output.s1.l16_1;
        hdr.output.s1.l16_2 = ig_md.s1.l16_2 + hdr.output.s1.l16_2;
        hdr.output.s1.l16_3 = ig_md.s1.l16_3 + hdr.output.s1.l16_3;
        hdr.output.s1.l16_4 = ig_md.s1.l16_4 + hdr.output.s1.l16_4;
        hdr.output.s1.l16_5 = ig_md.s1.l16_5 + hdr.output.s1.l16_5;
        hdr.output.s1.l16_6 = ig_md.s1.l16_6 + hdr.output.s1.l16_6;

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