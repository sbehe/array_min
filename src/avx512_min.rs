use core::arch::x86_64::*;
use core::arch::asm;

#[repr(align(64))]
struct AlignedArray([u16; 32]);

static INDEX_TABLE: AlignedArray = AlignedArray([
    0,1,2,3,4,5,6,7,
    8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,
    24,25,26,27,28,29,30,31
]);

pub fn scalar_min(arr: &[u16], start_index: usize, end_index: usize) -> (u16, usize) {
    let mut min_val = u16::MAX;
    let mut min_idx = 0;
    for i in start_index..=end_index {
        if arr[i] < min_val {
            min_val = arr[i];
            min_idx = i;
        }
    }
    (min_val, min_idx)
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn minindex_u16(
    arr: &[u16; 256],
    start: u16,
    end: u16,
) -> (u16, u16) {
    debug_assert!(start <= end);
    debug_assert!(end < 256);

    let val: u32;

    unsafe { asm!(
        "movzx eax, {start:l}",
        "movzx ecx, {end:l}",
        
        // ====================== DETECT ACTIVE BLOCKS ======================
        "mov     rdx, 0xE0C0A08060402000",
        "vmovq   xmm0, rdx",
        "vpmovzxbd ymm0, xmm0",
        
        "mov     rdx, 0xFFDFBF9F7F5F3F1F",
        "vmovq   xmm1, rdx",
        "vpmovzxbd ymm1, xmm1",
        
        "vpbroadcastd ymm2, ecx",
        "vpbroadcastd ymm3, eax",
        
        "vpcmpd k1, ymm0, ymm2, 2",
        "vpcmpd k2, ymm1, ymm3, 5",
        "kandb k3, k1, k2",
        
        // ====================== MASK GENERATION ======================
        "vpbroadcastb xmm4, eax",
        "vpbroadcastb xmm5, ecx",

        "mov     rdx, 0xE0C0A08060402000",
        "vmovq   xmm6, rdx",

        "vpsubusb xmm7, xmm4, xmm6",
        "vpsubusb xmm8, xmm5, xmm6",

        "vpcmpeqd ymm10, ymm10, ymm10",
        "vpmovzxbd ymm11, xmm7",
        "vpmovzxbd ymm12, xmm8",

        "vpsllvd  ymm13, ymm10, ymm11",
        "vpsllvd  ymm14, ymm10, ymm12",
        "vpslld   ymm14, ymm14, 1",
        "vpandn   ymm15, ymm14, ymm13",
        
        "vmovdqa32 ymm15{{k3}}{{z}}, ymm15",

        // ====================== PRESET TO 0xFFFFFFFF (max u32) ======================
        "vpternlogd zmm0, zmm0, zmm0, 0xFF",
        "vmovdqa64 zmm1, zmm0",
        "vmovdqa64 zmm2, zmm0",
        "vmovdqa64 zmm3, zmm0",
        "vmovdqa64 zmm4, zmm0",
        "vmovdqa64 zmm5, zmm0",
        "vmovdqa64 zmm6, zmm0",
        "vmovdqa64 zmm7, zmm0",

        // ====================== INDEX GENERATION (Base + Increment) ======================
        // Load [0..31] indices as u16
        "vmovdqu16 ymm16, [rip + {index}]",
        "vpmovzxwd zmm16, ymm16",            // Extend to 32-bit: [0,1,2,...,31]
        
        "mov r8d, 32",
        "vpbroadcastd zmm17, r8d",           // zmm17 = [32, 32, 32, ...]
        
        // Generate indices for each block by adding 32 repeatedly
        "vmovdqa64 zmm24, zmm16",            // Block 0: [0..31]
        "vpaddd zmm25, zmm24, zmm17",        // Block 1: [32..63]
        "vpaddd zmm26, zmm25, zmm17",        // Block 2: [64..95]
        "vpaddd zmm27, zmm26, zmm17",        // Block 3: [96..127]
        "vpaddd zmm28, zmm27, zmm17",        // Block 4: [128..159]
        "vpaddd zmm29, zmm28, zmm17",        // Block 5: [160..191]
        "vpaddd zmm30, zmm29, zmm17",        // Block 6: [192..223]
        "vpaddd zmm31, zmm30, zmm17",        // Block 7: [224..255]

        // ====================== LOAD AND COMBINE VALUE+INDEX ======================
        // Each block: load 32 u16 values, extend to u32, shift left 16, OR with indices
        
        // Block 0
        "vpextrd r8d, xmm15, 0",
        "kmovd k4, r8d",
        "vpmovzxwd zmm0{{k4}}{{z}}, [{ptr}]",
        "vpslld zmm0, zmm0, 16",
        "vpord zmm0, zmm0, zmm24",
        
        // Block 1
        "vpextrd r8d, xmm15, 1",
        "kmovd k4, r8d",
        "vpmovzxwd zmm1{{k4}}{{z}}, [{ptr} + 64]",
        "vpslld zmm1, zmm1, 16",
        "vpord zmm1, zmm1, zmm25",
        
        // Block 2
        "vpextrd r8d, xmm15, 2",
        "kmovd k4, r8d",
        "vpmovzxwd zmm2{{k4}}{{z}}, [{ptr} + 128]",
        "vpslld zmm2, zmm2, 16",
        "vpord zmm2, zmm2, zmm26",
        
        // Block 3
        "vpextrd r8d, xmm15, 3",
        "kmovd k4, r8d",
        "vpmovzxwd zmm3{{k4}}{{z}}, [{ptr} + 192]",
        "vpslld zmm3, zmm3, 16",
        "vpord zmm3, zmm3, zmm27",
        
        // Block 4
        "vextracti128 xmm14, ymm15, 1",
        "vpextrd r8d, xmm14, 0",
        "kmovd k4, r8d",
        "vpmovzxwd zmm4{{k4}}{{z}}, [{ptr} + 256]",
        "vpslld zmm4, zmm4, 16",
        "vpord zmm4, zmm4, zmm28",
        
        // Block 5
        "vpextrd r8d, xmm14, 1",
        "kmovd k4, r8d",
        "vpmovzxwd zmm5{{k4}}{{z}}, [{ptr} + 320]",
        "vpslld zmm5, zmm5, 16",
        "vpord zmm5, zmm5, zmm29",
        
        // Block 6
        "vpextrd r8d, xmm14, 2",
        "kmovd k4, r8d",
        "vpmovzxwd zmm6{{k4}}{{z}}, [{ptr} + 384]",
        "vpslld zmm6, zmm6, 16",
        "vpord zmm6, zmm6, zmm30",
        
        // Block 7
        "vpextrd r8d, xmm14, 3",
        "kmovd k4, r8d",
        "vpmovzxwd zmm7{{k4}}{{z}}, [{ptr} + 448]",
        "vpslld zmm7, zmm7, 16",
        "vpord zmm7, zmm7, zmm31",

        // ====================== FOLD BLOCKS (Using vpminud) ======================
        "vpminud zmm0, zmm0, zmm1",
        "vpminud zmm2, zmm2, zmm3",
        "vpminud zmm4, zmm4, zmm5",
        "vpminud zmm6, zmm6, zmm7",
        
        "vpminud zmm0, zmm0, zmm2",
        "vpminud zmm4, zmm4, zmm6",
        
        "vpminud zmm0, zmm0, zmm4",

        // ====================== HORIZONTAL REDUCE (Independent shuffles) ======================
        // zmm0 has 16 elements (each is u32 with packed value+index)
        
        // Reduce 16 -> 8: compare upper 256 bits with lower 256 bits
        "vextracti64x4 ymm16, zmm0, 1",
        "vpminud ymm0, ymm0, ymm16",
        
        // Reduce 8 -> 4: compare upper 128 bits with lower 128 bits
        "vextracti32x4 xmm16, ymm0, 1",
        "vpminud xmm0, xmm0, xmm16",
        
        // Reduce 4 -> 2: shuffle and compare
        "vpshufd xmm16, xmm0, 0x4E",         // [2,3,0,1]
        "vpminud xmm0, xmm0, xmm16",
        
        // Reduce 2 -> 1: shuffle and compare
        "vpshufd xmm16, xmm0, 0xB1",         // [1,0,3,2]
        "vpminud xmm0, xmm0, xmm16",
        
        // Extract final result
        "vmovd {result:e}, xmm0",
        
        ptr = in(reg) arr.as_ptr(),
        start = in(reg) start as u32,
        end = in(reg) end as u32,
        index = sym INDEX_TABLE,
        result = out(reg) val,

        out("rax") _, out("rcx") _, out("rdx") _, out("r8") _,
        out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _, out("xmm8") _,
        out("ymm10") _, out("ymm11") _, out("ymm12") _, out("ymm13") _, out("xmm14") _, out("ymm15") _,
        out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
        out("zmm16") _, out("zmm17") _,
        out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
        out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
        out("k1") _, out("k2") _, out("k3") _, out("k4") _,
        options(nostack),
    ); }

    let value = (val >> 16) as u16;
    let index = (val & 0xFFFF) as u16;
    
    (value, index)
}