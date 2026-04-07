/*
 * expert_shaders.metal — Dequant+matvec for MoE expert FFN.
 * Verified correct against GGML's reference dequantize_row_q3_K/q4_K/q5_K.
 * One thread per output row (scalar). TODO: SIMD tiling.
 */

#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;
constant uint TYPE_Q3K = 11;
constant uint TYPE_Q4K = 12;
constant uint TYPE_Q5K = 13;
constant uint BLOCK_Q3K = 110;
constant uint BLOCK_Q4K = 144;
constant uint BLOCK_Q5K = 176;

// ── Q3_K ───────────────────────────────────────────────────────────

inline float q3k_block_dot(device const uint8_t * bp, device const float * x, uint xo) {
    device const uint8_t * hm = bp;
    device const uint8_t * q  = bp + 32;
    device const uint8_t * sr = bp + 96;
    float d_all = float(as_type<half>(*(device const ushort *)(bp + 108)));

    uint a0 = uint(sr[0])|(uint(sr[1])<<8)|(uint(sr[2])<<16)|(uint(sr[3])<<24);
    uint a1 = uint(sr[4])|(uint(sr[5])<<8)|(uint(sr[6])<<16)|(uint(sr[7])<<24);
    uint tmp = uint(sr[8])|(uint(sr[9])<<8)|(uint(sr[10])<<16)|(uint(sr[11])<<24);
    uint k1=0x03030303u, k2=0x0f0f0f0fu;
    uint r0=(a0&k2)|(((tmp>>0)&k1)<<4), r1=(a1&k2)|(((tmp>>2)&k1)<<4);
    uint r2=((a0>>4)&k2)|(((tmp>>4)&k1)<<4), r3=((a1>>4)&k2)|(((tmp>>6)&k1)<<4);
    int8_t sc[16];
    sc[0]=int8_t(r0&0xFF);sc[1]=int8_t((r0>>8)&0xFF);sc[2]=int8_t((r0>>16)&0xFF);sc[3]=int8_t((r0>>24)&0xFF);
    sc[4]=int8_t(r1&0xFF);sc[5]=int8_t((r1>>8)&0xFF);sc[6]=int8_t((r1>>16)&0xFF);sc[7]=int8_t((r1>>24)&0xFF);
    sc[8]=int8_t(r2&0xFF);sc[9]=int8_t((r2>>8)&0xFF);sc[10]=int8_t((r2>>16)&0xFF);sc[11]=int8_t((r2>>24)&0xFF);
    sc[12]=int8_t(r3&0xFF);sc[13]=int8_t((r3>>8)&0xFF);sc[14]=int8_t((r3>>16)&0xFF);sc[15]=int8_t((r3>>24)&0xFF);

    float sum=0; int is=0; uint8_t m=1;
    for (uint n=0; n<QK_K; n+=128) {
        int shift=0;
        for (uint j=0; j<4; j++) {
            float dl=d_all*float(sc[is++]-32);
            for (uint l=0;l<16;l++) {
                int qv=int((q[l]>>shift)&3)-((hm[l]&m)?0:4);
                sum+=dl*float(qv)*x[xo+n+j*32+l];
            }
            dl=d_all*float(sc[is++]-32);
            for (uint l=0;l<16;l++) {
                int qv=int((q[l+16]>>shift)&3)-((hm[l+16]&m)?0:4);
                sum+=dl*float(qv)*x[xo+n+j*32+16+l];
            }
            shift+=2; m<<=1;
        }
        q+=32;
    }
    return sum;
}

// ── Q4_K / Q5_K shared helper ──────────────────────────────────────

inline void get_scale_min_k4(int j, device const uint8_t * q, thread uint8_t &d, thread uint8_t &m) {
    if (j<4) { d=q[j]&63; m=q[j+4]&63; }
    else { d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); m=(q[j+4]>>4)|((q[j]>>6)<<4); }
}

// ── Q4_K ───────────────────────────────────────────────────────────

inline float q4k_block_dot(device const uint8_t * bp, device const float * x, uint xo) {
    float d=float(as_type<half>(*(device const ushort*)(bp)));
    float mn=float(as_type<half>(*(device const ushort*)(bp+2)));
    device const uint8_t*sc=bp+4, *ql=bp+16;
    float sum=0; int is=0;
    for (uint j=0;j<QK_K;j+=64) {
        uint8_t s1,m1,s2,m2;
        get_scale_min_k4(is,sc,s1,m1); get_scale_min_k4(is+1,sc,s2,m2);
        float d1=d*float(s1),m1f=mn*float(m1),d2=d*float(s2),m2f=mn*float(m2);
        for(uint l=0;l<32;l++) sum+=(d1*float(ql[l]&0xF)-m1f)*x[xo+j+l];
        for(uint l=0;l<32;l++) sum+=(d2*float(ql[l]>>4)-m2f)*x[xo+j+32+l];
        ql+=32; is+=2;
    }
    return sum;
}

// ── Q5_K ───────────────────────────────────────────────────────────

inline float q5k_block_dot(device const uint8_t * bp, device const float * x, uint xo) {
    float d=float(as_type<half>(*(device const ushort*)(bp)));
    float mn=float(as_type<half>(*(device const ushort*)(bp+2)));
    device const uint8_t*sc=bp+4, *ql=bp+16, *qh=bp+144;
    float sum=0; int is=0; uint8_t u1=1,u2=2;
    for (uint j=0;j<QK_K;j+=64) {
        uint8_t s1,m1,s2,m2;
        get_scale_min_k4(is,sc,s1,m1); get_scale_min_k4(is+1,sc,s2,m2);
        float d1=d*float(s1),m1f=mn*float(m1),d2=d*float(s2),m2f=mn*float(m2);
        for(uint l=0;l<32;l++) sum+=(d1*float((ql[l]&0xF)+((qh[l]&u1)?16:0))-m1f)*x[xo+j+l];
        for(uint l=0;l<32;l++) sum+=(d2*float((ql[l]>>4)+((qh[l]&u2)?16:0))-m2f)*x[xo+j+32+l];
        ql+=32; is+=2; u1<<=2; u2<<=2;
    }
    return sum;
}

// ── Dispatch ───────────────────────────────────────────────────────

inline uint blk_bytes(uint t) {
    switch(t){case TYPE_Q3K:return BLOCK_Q3K;case TYPE_Q4K:return BLOCK_Q4K;case TYPE_Q5K:return BLOCK_Q5K;default:return 0;}
}

// One thread per output row
kernel void dequant_matvec(
    device const uint8_t * W [[buffer(0)]],
    device const float   * x [[buffer(1)]],
    device float         * out [[buffer(2)]],
    constant uint        * params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint out_dim=params[0], in_dim=params[1], type_id=params[2];
    if (tid>=out_dim) return;
    uint bs=blk_bytes(type_id), nb=in_dim/QK_K, rb=nb*bs;
    device const uint8_t*row=W+tid*rb;
    float sum=0;
    for (uint b=0;b<nb;b++) {
        device const uint8_t*blk=row+b*bs;
        switch(type_id){
            case TYPE_Q3K:sum+=q3k_block_dot(blk,x,b*QK_K);break;
            case TYPE_Q4K:sum+=q4k_block_dot(blk,x,b*QK_K);break;
            case TYPE_Q5K:sum+=q5k_block_dot(blk,x,b*QK_K);break;
            default:break;
        }
    }
    out[tid]=sum;
}

kernel void swiglu_fused(
    device float*gate[[buffer(0)]],device const float*up[[buffer(1)]],
    constant uint*params[[buffer(2)]],uint tid[[thread_position_in_grid]]
) {
    if(tid>=params[0])return;
    float g=gate[tid]; gate[tid]=(g/(1.0f+exp(-g)))*up[tid];
}

kernel void weighted_add(
    device float*out[[buffer(0)]],device const float*src[[buffer(1)]],
    constant float*weight[[buffer(2)]],constant uint*params[[buffer(3)]],
    uint tid[[thread_position_in_grid]]
) {
    if(tid>=params[0])return;
    out[tid]+=weight[0]*src[tid];
}
