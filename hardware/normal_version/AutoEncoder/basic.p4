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
    action set_emb_pkl(bit<8> in01, bit<8> in02, bit<8> in03, bit<8> in04) {
        hdr.feature.x_8.l8_1 = in01;
        hdr.feature.x_8.l8_2 = in02;
        hdr.feature.x_8.l8_3 = in03;
        hdr.feature.x_8.l8_4 = in04;
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

    action set_emb_ipd(bit<8> in01, bit<8> in02, bit<8> in03, bit<8> in04) {
        hdr.feature.x_8.l8_1 = in01 + hdr.feature.x_8.l8_1;
        hdr.feature.x_8.l8_2 = in02 + hdr.feature.x_8.l8_2;
        hdr.feature.x_8.l8_3 = in03 + hdr.feature.x_8.l8_3;
        hdr.feature.x_8.l8_4 = in04 + hdr.feature.x_8.l8_4;
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
        size = 4096;
        default_action = noaction();
    }

    action set_b1_1_2(bit<4> in01) {
        ig_md.temp.l4_1 = in01;
    }
    action set_b1_3_4(bit<4> in01) {
        ig_md.temp.l4_2 = in01;
    }
    action set_b2_1_2(bit<4> in01) {
        ig_md.temp.l4_3 = in01;
    }
    action set_b2_3_4(bit<4> in01) {
        ig_md.temp.l4_4 = in01;
    }
    action set_b3_1_2(bit<4> in01) {
        ig_md.temp.l4_5 = in01;
    }
    action set_b3_3_4(bit<4> in01) {
        ig_md.temp.l4_6 = in01;
    }
    action set_b4_1_2(bit<4> in01) {
        ig_md.temp.l4_7 = in01;
    }
    action set_b4_3_4(bit<4> in01) {
        ig_md.temp.l4_8 = in01;
    }
    action set_b5_1_2(bit<4> in01) {
        ig_md.temp.l4_9 = in01;
    }
    action set_b5_3_4(bit<4> in01) {
        ig_md.temp.l4_10 = in01;
    }
    action set_b6_1_2(bit<4> in01) {
        ig_md.temp.l4_11 = in01;
    }
    action set_b6_3_4(bit<4> in01) {
        ig_md.temp.l4_12 = in01;
    }
    action set_b7_1_2(bit<4> in01) {
        ig_md.temp.l4_13 = in01;
    }
    action set_b7_3_4(bit<4> in01) {
        ig_md.temp.l4_14 = in01;
    }
    action set_b8_1_2(bit<4> in01) {
        ig_md.temp.l4_15 = in01;
    }
    action set_b8_3_4(bit<4> in01) {
        ig_md.temp.l4_16 = in01;
    }
    // @stage ()
    table b1_1_2 {
        key = {
            hdr.feature.x_1.l8_1: ternary;
            hdr.feature.x_1.l8_2: ternary;
        }
        actions = {
            set_b1_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b1_3_4 {
        key = {
            hdr.feature.x_1.l8_3: ternary;
            hdr.feature.x_1.l8_4: ternary;
        }
        actions = {
            set_b1_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    // @stage ()
    table b2_1_2 {
        key = {
            hdr.feature.x_2.l8_1: ternary;
            hdr.feature.x_2.l8_2: ternary;
        }
        actions = {
            set_b2_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b2_3_4 {
        key = {
            hdr.feature.x_2.l8_3: ternary;
            hdr.feature.x_2.l8_4: ternary;
        }
        actions = {
            set_b2_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    // @stage ()
    table b3_1_2 {
        key = {
            hdr.feature.x_3.l8_1: ternary;
            hdr.feature.x_3.l8_2: ternary;
        }
        actions = {
            set_b3_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b3_3_4 {
        key = {
            hdr.feature.x_3.l8_3: ternary;
            hdr.feature.x_3.l8_4: ternary;
        }
        actions = {
            set_b3_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    // @stage ()
    table b4_1_2 {
        key = {
            hdr.feature.x_4.l8_1: ternary;
            hdr.feature.x_4.l8_2: ternary;
        }
        actions = {
            set_b4_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b4_3_4 {
        key = {
            hdr.feature.x_4.l8_3: ternary;
            hdr.feature.x_4.l8_4: ternary;
        }
        actions = {
            set_b4_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    // @stage ()
    table b5_1_2 {
        key = {
            hdr.feature.x_5.l8_1: ternary;
            hdr.feature.x_5.l8_2: ternary;
        }
        actions = {
            set_b5_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b5_3_4 {
        key = {
            hdr.feature.x_5.l8_3: ternary;
            hdr.feature.x_5.l8_4: ternary;
        }
        actions = {
            set_b5_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    // @stage ()
    table b6_1_2 {
        key = {
            hdr.feature.x_6.l8_1: ternary;
            hdr.feature.x_6.l8_2: ternary;
        }
        actions = {
            set_b6_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b6_3_4 {
        key = {
            hdr.feature.x_6.l8_3: ternary;
            hdr.feature.x_6.l8_4: ternary;
        }
        actions = {
            set_b6_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    // @stage ()
    table b7_1_2 {
        key = {
            hdr.feature.x_7.l8_1: ternary;
            hdr.feature.x_7.l8_2: ternary;
        }
        actions = {
            set_b7_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b7_3_4 {
        key = {
            hdr.feature.x_7.l8_3: ternary;
            hdr.feature.x_7.l8_4: ternary;
        }
        actions = {
            set_b7_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    // @stage ()
    table b8_1_2 {
        key = {
            hdr.feature.x_8.l8_1: ternary;
            hdr.feature.x_8.l8_2: ternary;
        }
        actions = {
            set_b8_1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b8_3_4 {
        key = {
            hdr.feature.x_8.l8_3: ternary;
            hdr.feature.x_8.l8_4: ternary;
        }
        actions = {
            set_b8_3_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
// *****cal cnn second

    action set_b2_1(bit<8> in01, bit<8> in02, bit<8> in03, bit<8> in04, bit<8> in05, bit<8> in06) {
        ig_md.b1.l8_1 = in01;
        ig_md.b1.l8_2 = in02;
        ig_md.b1.l8_3 = in03;
        ig_md.b1.l8_4 = in04;
        ig_md.b1.l8_5 = in05;
        ig_md.b1.l8_6 = in06;
    }
    action set_b2_2(bit<8> in01, bit<8> in02, bit<8> in03, bit<8> in04, bit<8> in05, bit<8> in06) {
        ig_md.b2.l8_1 = in01;
        ig_md.b2.l8_2 = in02;
        ig_md.b2.l8_3 = in03;
        ig_md.b2.l8_4 = in04;
        ig_md.b2.l8_5 = in05;
        ig_md.b2.l8_6 = in06;
    }
    action set_b2_3(bit<8> in01, bit<8> in02, bit<8> in03, bit<8> in04, bit<8> in05, bit<8> in06) {
        ig_md.b3.l8_1 = in01;
        ig_md.b3.l8_2 = in02;
        ig_md.b3.l8_3 = in03;
        ig_md.b3.l8_4 = in04;
        ig_md.b3.l8_5 = in05;
        ig_md.b3.l8_6 = in06;
    }
    action set_b2_4(bit<8> in01, bit<8> in02, bit<8> in03, bit<8> in04, bit<8> in05, bit<8> in06) {
        ig_md.b1.l8_1 = in01 + ig_md.b1.l8_1;
        ig_md.b1.l8_2 = in02 + ig_md.b1.l8_2;
        ig_md.b1.l8_3 = in03 + ig_md.b1.l8_3;
        ig_md.b1.l8_4 = in04 + ig_md.b1.l8_4;
        ig_md.b1.l8_5 = in05 + ig_md.b1.l8_5;
        ig_md.b1.l8_6 = in06 + ig_md.b1.l8_6;
    }
    action set_b2_5(bit<8> in01, bit<8> in02, bit<8> in03, bit<8> in04, bit<8> in05, bit<8> in06) {
        ig_md.b2.l8_1 = in01 + ig_md.b2.l8_1;
        ig_md.b2.l8_2 = in02 + ig_md.b2.l8_2;
        ig_md.b2.l8_3 = in03 + ig_md.b2.l8_3;
        ig_md.b2.l8_4 = in04 + ig_md.b2.l8_4;
        ig_md.b2.l8_5 = in05 + ig_md.b2.l8_5;
        ig_md.b2.l8_6 = in06 + ig_md.b2.l8_6;
    }
    action set_b2_6(bit<8> in01, bit<8> in02, bit<8> in03, bit<8> in04, bit<8> in05, bit<8> in06) {
        ig_md.b3.l8_1 = in01 + ig_md.b3.l8_1;
        ig_md.b3.l8_2 = in02 + ig_md.b3.l8_2;
        ig_md.b3.l8_3 = in03 + ig_md.b3.l8_3;
        ig_md.b3.l8_4 = in04 + ig_md.b3.l8_4;
        ig_md.b3.l8_5 = in05 + ig_md.b1.l8_5;
        ig_md.b3.l8_6 = in06 + ig_md.b1.l8_6;
    }
    //@stage (1)
    table b2_1 {
        key = {
            ig_md.temp.l4_1: exact;
            ig_md.temp.l4_2: exact;
            ig_md.temp.l4_3: exact;
        }
        actions = {
            set_b2_1;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    //@stage (1)
    table b2_2 {
        key = {
            ig_md.temp.l4_4: exact;
            ig_md.temp.l4_5: exact;
            ig_md.temp.l4_6: exact;
        }
        actions = {
            set_b2_2;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    //@stage (1)
    table b2_3 {
        key = {
            ig_md.temp.l4_7: exact;
            ig_md.temp.l4_8: exact;
            ig_md.temp.l4_9: exact;
        }
        actions = {
            set_b2_3;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    //@stage (1)
    table b2_4 {
        key = {
            ig_md.temp.l4_10: exact;
            ig_md.temp.l4_11: exact;
            ig_md.temp.l4_12: exact;
        }
        actions = {
            set_b2_4;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    //@stage (1)
    table b2_5 {
        key = {
            ig_md.temp.l4_13: exact;
            ig_md.temp.l4_14: exact;
            ig_md.temp.l4_15: exact;
        }
        actions = {
            set_b2_5;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
    //@stage (1)
    table b2_6 {
        key = {
            ig_md.temp.l4_16: exact;
        }
        actions = {
            set_b2_6;
            noaction;
        }
        size = 16;
        default_action = noaction();
        idle_timeout = false;
    }


    action set_b3_1(bit<4> in01) {
        ig_md.temp.l4_1 = in01;
    }
    action set_b3_2(bit<4> in01) {
        ig_md.temp.l4_2 = in01;
    }
    action set_b3_3(bit<4> in01) {
        ig_md.temp.l4_3 = in01;
    }
    table b3_1 {
        key = {
            ig_md.b1.l8_1: ternary;
            ig_md.b1.l8_2: ternary;
        }
        actions = {
            set_b3_1;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b3_2 {
        key = {
            ig_md.b1.l8_3: ternary;
            ig_md.b1.l8_4: ternary;
        }
        actions = {
            set_b3_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }
    table b3_3 {
        key = {
            ig_md.b1.l8_5: ternary;
            ig_md.b1.l8_6: ternary;
        }
        actions = {
            set_b3_3;
            noaction;
        }
        size = 1024;
        default_action = noaction();
        idle_timeout = false;
    }


action set_b4_1(bit<8> in01, bit<8> in02, bit<8> in03, bit<8> in04, bit<8> in05, bit<8> in06, bit<8> in07, bit<8> in08, bit<8> in09, bit<8> in10, bit<8> in11, bit<8> in12, bit<8> in13, bit<8> in14, bit<8> in15, bit<8> in16, bit<8> in17, bit<8> in18, bit<8> in19, bit<8> in20, bit<8> in21, bit<8> in22, bit<8> in23, bit<8> in24, bit<8> in25, bit<8> in26, bit<8> in27, bit<8> in28, bit<8> in29, bit<8> in30, bit<8> in31, bit<8> in32) {
    ig_md.d1.l8_1 = in01;
    ig_md.d1.l8_2 = in02;
    ig_md.d1.l8_3 = in03;
    ig_md.d1.l8_4 = in04;
    ig_md.d1.l8_5 = in05;
    ig_md.d1.l8_6 = in06;
    ig_md.d1.l8_7 = in07;
    ig_md.d1.l8_8 = in08;
    ig_md.d1.l8_9 = in09;
    ig_md.d1.l8_10 = in10;
    ig_md.d1.l8_11 = in11;
    ig_md.d1.l8_12 = in12;
    ig_md.d1.l8_13 = in13;
    ig_md.d1.l8_14 = in14;
    ig_md.d1.l8_15 = in15;
    ig_md.d1.l8_16 = in16;
    ig_md.d1.l8_17 = in17;
    ig_md.d1.l8_18 = in18;
    ig_md.d1.l8_19 = in19;
    ig_md.d1.l8_20 = in20;
    ig_md.d1.l8_21 = in21;
    ig_md.d1.l8_22 = in22;
    ig_md.d1.l8_23 = in23;
    ig_md.d1.l8_24 = in24;
    ig_md.d1.l8_25 = in25;
    ig_md.d1.l8_26 = in26;
    ig_md.d1.l8_27 = in27;
    ig_md.d1.l8_28 = in28;
    ig_md.d1.l8_29 = in29;
    ig_md.d1.l8_30 = in30;
    ig_md.d1.l8_31 = in31;
    ig_md.d1.l8_32 = in32;
}

    @stage(15)
    table b4_1 {
        key = {
            ig_md.temp.l4_1: exact;
            ig_md.temp.l4_2: exact;
            ig_md.temp.l4_3: exact;
        }
        actions = {
            set_b4_1;
            noaction;
        }
        size = 4096;
        default_action = noaction();
        idle_timeout = false;
    }
// ************************    

    apply
    {
// *****获取嵌入层******
        emb_pkl.apply();
        emb_ipd.apply();

        b1_1_2.apply();
        b1_3_4.apply();
        b2_1_2.apply();
        b2_3_4.apply();
        b3_1_2.apply();
        b3_3_4.apply();
        b4_1_2.apply();
        b4_3_4.apply();
        b5_1_2.apply();
        b5_3_4.apply();
        b6_1_2.apply();
        b6_3_4.apply();
        b7_1_2.apply();
        b7_3_4.apply();
        b8_1_2.apply();
        b8_3_4.apply();

        b2_1.apply();
        b2_2.apply();
        b2_3.apply();
        b2_4.apply();
        b2_5.apply();
        b2_6.apply();

        ig_md.b1.l8_1 = ig_md.b1.l8_1 + ig_md.b2.l8_1;
        ig_md.b1.l8_2 = ig_md.b1.l8_2 + ig_md.b2.l8_2;
        ig_md.b1.l8_3 = ig_md.b1.l8_3 + ig_md.b2.l8_3;
        ig_md.b1.l8_4 = ig_md.b1.l8_4 + ig_md.b2.l8_4;
        ig_md.b1.l8_5 = ig_md.b1.l8_5 + ig_md.b2.l8_5;
        ig_md.b1.l8_6 = ig_md.b1.l8_6 + ig_md.b2.l8_6;

        ig_md.b1.l8_1 = ig_md.b1.l8_1 + ig_md.b3.l8_1;
        ig_md.b1.l8_2 = ig_md.b1.l8_2 + ig_md.b3.l8_2;
        ig_md.b1.l8_3 = ig_md.b1.l8_3 + ig_md.b3.l8_3;
        ig_md.b1.l8_4 = ig_md.b1.l8_4 + ig_md.b3.l8_4;
        ig_md.b1.l8_5 = ig_md.b1.l8_5 + ig_md.b3.l8_5;
        ig_md.b1.l8_6 = ig_md.b1.l8_6 + ig_md.b3.l8_6;

        b3_1.apply();
        b3_2.apply();
        b3_3.apply();

        b4_1.apply();

ig_md.d1.l8_1 = hdr.feature.x_1.l8_1 - ig_md.d1.l8_1; if(ig_md.d1.l8_1 & 0x80 == 0x80){ hdr.storage.s_1.l16_1[7:0] = 0 - ig_md.d1.l8_1;} else { hdr.storage.s_1.l16_1[7:0] = ig_md.d1.l8_1;}
ig_md.d1.l8_2 = hdr.feature.x_1.l8_2 - ig_md.d1.l8_2; if(ig_md.d1.l8_2 & 0x80 == 0x80){ hdr.storage.s_1.l16_2[7:0] = 0 - ig_md.d1.l8_2;} else { hdr.storage.s_1.l16_2[7:0] = ig_md.d1.l8_2;}
ig_md.d1.l8_3 = hdr.feature.x_1.l8_3 - ig_md.d1.l8_3; if(ig_md.d1.l8_3 & 0x80 == 0x80){ hdr.storage.s_1.l16_3[7:0] = 0 - ig_md.d1.l8_3;} else { hdr.storage.s_1.l16_3[7:0] = ig_md.d1.l8_3;}
ig_md.d1.l8_4 = hdr.feature.x_1.l8_4 - ig_md.d1.l8_4; if(ig_md.d1.l8_4 & 0x80 == 0x80){ hdr.storage.s_1.l16_4[7:0] = 0 - ig_md.d1.l8_4;} else { hdr.storage.s_1.l16_4[7:0] = ig_md.d1.l8_4;}
ig_md.d1.l8_5 = hdr.feature.x_2.l8_1 - ig_md.d1.l8_5; if(ig_md.d1.l8_5 & 0x80 == 0x80){ hdr.storage.s_1.l16_5[7:0] = 0 - ig_md.d1.l8_5;} else { hdr.storage.s_1.l16_5[7:0] = ig_md.d1.l8_5;}
ig_md.d1.l8_6 = hdr.feature.x_2.l8_2 - ig_md.d1.l8_6; if(ig_md.d1.l8_6 & 0x80 == 0x80){ hdr.storage.s_1.l16_6[7:0] = 0 - ig_md.d1.l8_6;} else { hdr.storage.s_1.l16_6[7:0] = ig_md.d1.l8_6;}
ig_md.d1.l8_7 = hdr.feature.x_2.l8_3 - ig_md.d1.l8_7; if(ig_md.d1.l8_7 & 0x80 == 0x80){ hdr.storage.s_1.l16_7[7:0] = 0 - ig_md.d1.l8_7;} else { hdr.storage.s_1.l16_7[7:0] = ig_md.d1.l8_7;}
ig_md.d1.l8_8 = hdr.feature.x_2.l8_4 - ig_md.d1.l8_8; if(ig_md.d1.l8_8 & 0x80 == 0x80){ hdr.storage.s_1.l16_8[7:0] = 0 - ig_md.d1.l8_8;} else { hdr.storage.s_1.l16_8[7:0] = ig_md.d1.l8_8;}
ig_md.d1.l8_9 = hdr.feature.x_3.l8_1 - ig_md.d1.l8_9; if(ig_md.d1.l8_9 & 0x80 == 0x80){ hdr.storage.s_1.l16_9[7:0] = 0 - ig_md.d1.l8_9;} else { hdr.storage.s_1.l16_9[7:0] = ig_md.d1.l8_9;}
ig_md.d1.l8_10 = hdr.feature.x_3.l8_2 - ig_md.d1.l8_10; if(ig_md.d1.l8_10 & 0x80 == 0x80){ hdr.storage.s_1.l16_10[7:0] = 0 - ig_md.d1.l8_10;} else { hdr.storage.s_1.l16_10[7:0] = ig_md.d1.l8_10;}
ig_md.d1.l8_11 = hdr.feature.x_3.l8_3 - ig_md.d1.l8_11; if(ig_md.d1.l8_11 & 0x80 == 0x80){ hdr.storage.s_1.l16_11[7:0] = 0 - ig_md.d1.l8_11;} else { hdr.storage.s_1.l16_11[7:0] = ig_md.d1.l8_11;}
ig_md.d1.l8_12 = hdr.feature.x_3.l8_4 - ig_md.d1.l8_12; if(ig_md.d1.l8_12 & 0x80 == 0x80){ hdr.storage.s_1.l16_12[7:0] = 0 - ig_md.d1.l8_12;} else { hdr.storage.s_1.l16_12[7:0] = ig_md.d1.l8_12;}
ig_md.d1.l8_13 = hdr.feature.x_4.l8_1 - ig_md.d1.l8_13; if(ig_md.d1.l8_13 & 0x80 == 0x80){ hdr.storage.s_1.l16_13[7:0] = 0 - ig_md.d1.l8_13;} else { hdr.storage.s_1.l16_13[7:0] = ig_md.d1.l8_13;}
ig_md.d1.l8_14 = hdr.feature.x_4.l8_2 - ig_md.d1.l8_14; if(ig_md.d1.l8_14 & 0x80 == 0x80){ hdr.storage.s_1.l16_14[7:0] = 0 - ig_md.d1.l8_14;} else { hdr.storage.s_1.l16_14[7:0] = ig_md.d1.l8_14;}
ig_md.d1.l8_15 = hdr.feature.x_4.l8_3 - ig_md.d1.l8_15; if(ig_md.d1.l8_15 & 0x80 == 0x80){ hdr.storage.s_1.l16_15[7:0] = 0 - ig_md.d1.l8_15;} else { hdr.storage.s_1.l16_15[7:0] = ig_md.d1.l8_15;}
ig_md.d1.l8_16 = hdr.feature.x_4.l8_4 - ig_md.d1.l8_16; if(ig_md.d1.l8_16 & 0x80 == 0x80){ hdr.storage.s_1.l16_16[7:0] = 0 - ig_md.d1.l8_16;} else { hdr.storage.s_1.l16_16[7:0] = ig_md.d1.l8_16;}
ig_md.d1.l8_17 = hdr.feature.x_5.l8_1 - ig_md.d1.l8_17; if(ig_md.d1.l8_17 & 0x80 == 0x80){ hdr.storage.s_1.l16_17[7:0] = 0 - ig_md.d1.l8_17;} else { hdr.storage.s_1.l16_17[7:0] = ig_md.d1.l8_17;}
ig_md.d1.l8_18 = hdr.feature.x_5.l8_2 - ig_md.d1.l8_18; if(ig_md.d1.l8_18 & 0x80 == 0x80){ hdr.storage.s_1.l16_18[7:0] = 0 - ig_md.d1.l8_18;} else { hdr.storage.s_1.l16_18[7:0] = ig_md.d1.l8_18;}
ig_md.d1.l8_19 = hdr.feature.x_5.l8_3 - ig_md.d1.l8_19; if(ig_md.d1.l8_19 & 0x80 == 0x80){ hdr.storage.s_1.l16_19[7:0] = 0 - ig_md.d1.l8_19;} else { hdr.storage.s_1.l16_19[7:0] = ig_md.d1.l8_19;}
ig_md.d1.l8_20 = hdr.feature.x_5.l8_4 - ig_md.d1.l8_20; if(ig_md.d1.l8_20 & 0x80 == 0x80){ hdr.storage.s_1.l16_20[7:0] = 0 - ig_md.d1.l8_20;} else { hdr.storage.s_1.l16_20[7:0] = ig_md.d1.l8_20;}
ig_md.d1.l8_21 = hdr.feature.x_6.l8_1 - ig_md.d1.l8_21; if(ig_md.d1.l8_21 & 0x80 == 0x80){ hdr.storage.s_1.l16_21[7:0] = 0 - ig_md.d1.l8_21;} else { hdr.storage.s_1.l16_21[7:0] = ig_md.d1.l8_21;}
ig_md.d1.l8_22 = hdr.feature.x_6.l8_2 - ig_md.d1.l8_22; if(ig_md.d1.l8_22 & 0x80 == 0x80){ hdr.storage.s_1.l16_22[7:0] = 0 - ig_md.d1.l8_22;} else { hdr.storage.s_1.l16_22[7:0] = ig_md.d1.l8_22;}
ig_md.d1.l8_23 = hdr.feature.x_6.l8_3 - ig_md.d1.l8_23; if(ig_md.d1.l8_23 & 0x80 == 0x80){ hdr.storage.s_1.l16_23[7:0] = 0 - ig_md.d1.l8_23;} else { hdr.storage.s_1.l16_23[7:0] = ig_md.d1.l8_23;}
ig_md.d1.l8_24 = hdr.feature.x_6.l8_4 - ig_md.d1.l8_24; if(ig_md.d1.l8_24 & 0x80 == 0x80){ hdr.storage.s_1.l16_24[7:0] = 0 - ig_md.d1.l8_24;} else { hdr.storage.s_1.l16_24[7:0] = ig_md.d1.l8_24;}
ig_md.d1.l8_25 = hdr.feature.x_7.l8_1 - ig_md.d1.l8_25; if(ig_md.d1.l8_25 & 0x80 == 0x80){ hdr.storage.s_1.l16_25[7:0] = 0 - ig_md.d1.l8_25;} else { hdr.storage.s_1.l16_25[7:0] = ig_md.d1.l8_25;}
ig_md.d1.l8_26 = hdr.feature.x_7.l8_2 - ig_md.d1.l8_26; if(ig_md.d1.l8_26 & 0x80 == 0x80){ hdr.storage.s_1.l16_26[7:0] = 0 - ig_md.d1.l8_26;} else { hdr.storage.s_1.l16_26[7:0] = ig_md.d1.l8_26;}
ig_md.d1.l8_27 = hdr.feature.x_7.l8_3 - ig_md.d1.l8_27; if(ig_md.d1.l8_27 & 0x80 == 0x80){ hdr.storage.s_1.l16_27[7:0] = 0 - ig_md.d1.l8_27;} else { hdr.storage.s_1.l16_27[7:0] = ig_md.d1.l8_27;}
ig_md.d1.l8_28 = hdr.feature.x_7.l8_4 - ig_md.d1.l8_28; if(ig_md.d1.l8_28 & 0x80 == 0x80){ hdr.storage.s_1.l16_28[7:0] = 0 - ig_md.d1.l8_28;} else { hdr.storage.s_1.l16_28[7:0] = ig_md.d1.l8_28;}
ig_md.d1.l8_29 = hdr.feature.x_8.l8_1 - ig_md.d1.l8_29; if(ig_md.d1.l8_29 & 0x80 == 0x80){ hdr.storage.s_1.l16_29[7:0] = 0 - ig_md.d1.l8_29;} else { hdr.storage.s_1.l16_29[7:0] = ig_md.d1.l8_29;}
ig_md.d1.l8_30 = hdr.feature.x_8.l8_2 - ig_md.d1.l8_30; if(ig_md.d1.l8_30 & 0x80 == 0x80){ hdr.storage.s_1.l16_30[7:0] = 0 - ig_md.d1.l8_30;} else { hdr.storage.s_1.l16_30[7:0] = ig_md.d1.l8_30;}
ig_md.d1.l8_31 = hdr.feature.x_8.l8_3 - ig_md.d1.l8_31; if(ig_md.d1.l8_31 & 0x80 == 0x80){ hdr.storage.s_1.l16_31[7:0] = 0 - ig_md.d1.l8_31;} else { hdr.storage.s_1.l16_31[7:0] = ig_md.d1.l8_31;}
ig_md.d1.l8_32 = hdr.feature.x_8.l8_4 - ig_md.d1.l8_32; if(ig_md.d1.l8_32 & 0x80 == 0x80){ hdr.storage.s_1.l16_32[7:0] = 0 - ig_md.d1.l8_32;} else { hdr.storage.s_1.l16_32[7:0] = ig_md.d1.l8_32;}

hdr.storage.s_1.l16_1 = hdr.storage.s_1.l16_1 + hdr.storage.s_1.l16_17;
hdr.storage.s_1.l16_2 = hdr.storage.s_1.l16_2 + hdr.storage.s_1.l16_18;
hdr.storage.s_1.l16_3 = hdr.storage.s_1.l16_3 + hdr.storage.s_1.l16_19;
hdr.storage.s_1.l16_4 = hdr.storage.s_1.l16_4 + hdr.storage.s_1.l16_20;
hdr.storage.s_1.l16_5 = hdr.storage.s_1.l16_5 + hdr.storage.s_1.l16_21;
hdr.storage.s_1.l16_6 = hdr.storage.s_1.l16_6 + hdr.storage.s_1.l16_22;
hdr.storage.s_1.l16_7 = hdr.storage.s_1.l16_7 + hdr.storage.s_1.l16_23;
hdr.storage.s_1.l16_8 = hdr.storage.s_1.l16_8 + hdr.storage.s_1.l16_24;
hdr.storage.s_1.l16_9 = hdr.storage.s_1.l16_9 + hdr.storage.s_1.l16_25;
hdr.storage.s_1.l16_10 = hdr.storage.s_1.l16_10 + hdr.storage.s_1.l16_26;
hdr.storage.s_1.l16_11 = hdr.storage.s_1.l16_11 + hdr.storage.s_1.l16_27;
hdr.storage.s_1.l16_12 = hdr.storage.s_1.l16_12 + hdr.storage.s_1.l16_28;
hdr.storage.s_1.l16_13 = hdr.storage.s_1.l16_13 + hdr.storage.s_1.l16_29;
hdr.storage.s_1.l16_14 = hdr.storage.s_1.l16_14 + hdr.storage.s_1.l16_30;
hdr.storage.s_1.l16_15 = hdr.storage.s_1.l16_15 + hdr.storage.s_1.l16_31;
hdr.storage.s_1.l16_16 = hdr.storage.s_1.l16_16 + hdr.storage.s_1.l16_32;
hdr.storage.s_1.l16_1 = hdr.storage.s_1.l16_1 + hdr.storage.s_1.l16_9;
hdr.storage.s_1.l16_2 = hdr.storage.s_1.l16_2 + hdr.storage.s_1.l16_10;
hdr.storage.s_1.l16_3 = hdr.storage.s_1.l16_3 + hdr.storage.s_1.l16_11;
hdr.storage.s_1.l16_4 = hdr.storage.s_1.l16_4 + hdr.storage.s_1.l16_12;
hdr.storage.s_1.l16_5 = hdr.storage.s_1.l16_5 + hdr.storage.s_1.l16_13;
hdr.storage.s_1.l16_6 = hdr.storage.s_1.l16_6 + hdr.storage.s_1.l16_14;
hdr.storage.s_1.l16_7 = hdr.storage.s_1.l16_7 + hdr.storage.s_1.l16_15;
hdr.storage.s_1.l16_8 = hdr.storage.s_1.l16_8 + hdr.storage.s_1.l16_16;
hdr.storage.s_1.l16_1 = hdr.storage.s_1.l16_1 + hdr.storage.s_1.l16_5;
hdr.storage.s_1.l16_2 = hdr.storage.s_1.l16_2 + hdr.storage.s_1.l16_6;
hdr.storage.s_1.l16_3 = hdr.storage.s_1.l16_3 + hdr.storage.s_1.l16_7;
hdr.storage.s_1.l16_4 = hdr.storage.s_1.l16_4 + hdr.storage.s_1.l16_8;
hdr.storage.s_1.l16_1 = hdr.storage.s_1.l16_1 + hdr.storage.s_1.l16_3;
hdr.storage.s_1.l16_2 = hdr.storage.s_1.l16_2 + hdr.storage.s_1.l16_4;
hdr.storage.s_1.l16_1 = hdr.storage.s_1.l16_1 + hdr.storage.s_1.l16_2;

        

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
    apply{
        hdr.output.l16_1 = hdr.storage.s_1.l16_1;
        hdr.output.setValid();
    }
}
Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         SwitchEgressParser(),
         SwitchEgress(),
         SwitchEgressDeparser()) pipe;

Switch(pipe) main;