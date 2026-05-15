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
    let idx: u32;

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

        // ====================== PRESET TO 0xFFFF ======================
        "vpternlogd zmm16, zmm16, zmm16, 0xFF",
        "vmovdqa64 zmm17, zmm16",
        "vmovdqa64 zmm18, zmm16",
        "vmovdqa64 zmm19, zmm16",
        "vmovdqa64 zmm20, zmm16",
        "vmovdqa64 zmm21, zmm16",
        "vmovdqa64 zmm22, zmm16",
        "vmovdqa64 zmm23, zmm16",

        // ====================== MASKED LOADS ======================
        "vpextrd eax, xmm15, 0",
        "kmovd k4, eax",
        "vmovdqu16 zmm16{{k4}}, [{ptr}]",

        "vpextrd eax, xmm15, 1",
        "kmovd k4, eax",
        "vmovdqu16 zmm17{{k4}}, [{ptr} + 64]",

        "vpextrd eax, xmm15, 2",
        "kmovd k4, eax",
        "vmovdqu16 zmm18{{k4}}, [{ptr} + 128]",

        "vpextrd eax, xmm15, 3",
        "kmovd k4, eax",
        "vmovdqu16 zmm19{{k4}}, [{ptr} + 192]",

        "vextracti128 xmm14, ymm15, 1",
        
        "vpextrd eax, xmm14, 0",
        "kmovd k4, eax",
        "vmovdqu16 zmm20{{k4}}, [{ptr} + 256]",

        "vpextrd eax, xmm14, 1",
        "kmovd k4, eax",
        "vmovdqu16 zmm21{{k4}}, [{ptr} + 320]",

        "vpextrd eax, xmm14, 2",
        "kmovd k4, eax",
        "vmovdqu16 zmm22{{k4}}, [{ptr} + 384]",

        "vpextrd eax, xmm14, 3",
        "kmovd k4, eax",
        "vmovdqu16 zmm23{{k4}}, [{ptr} + 448]",

        // ====================== INDEX GENERATION ======================
        "vmovdqu16 zmm24, [rip + {index}]",
        "vmovdqa64 zmm25, zmm24",
        "vmovdqa64 zmm26, zmm24",
        "vmovdqa64 zmm27, zmm24",
        "vmovdqa64 zmm28, zmm24",
        "vmovdqa64 zmm29, zmm24",
        "vmovdqa64 zmm30, zmm24",
        "vmovdqa64 zmm31, zmm24",

        "vpaddw zmm25, zmm25, [rip + {offset32}]",
        "vpaddw zmm26, zmm26, [rip + {offset64}]",
        "vpaddw zmm27, zmm27, [rip + {offset96}]",
        "vpaddw zmm28, zmm28, [rip + {offset128}]",
        "vpaddw zmm29, zmm29, [rip + {offset160}]",
        "vpaddw zmm30, zmm30, [rip + {offset192}]",
        "vpaddw zmm31, zmm31, [rip + {offset224}]",

        // ====================== FOLD BLOCKS 0-3 ======================
        "vpcmpuw k5, zmm17, zmm16, 1",
        "vmovdqu16 zmm16{{k5}}, zmm17",
        "vmovdqu16 zmm24{{k5}}, zmm25",

        "vpcmpuw k5, zmm19, zmm18, 1",
        "vmovdqu16 zmm18{{k5}}, zmm19",
        "vmovdqu16 zmm26{{k5}}, zmm27",

        "vpcmpuw k5, zmm18, zmm16, 1",
        "vmovdqu16 zmm16{{k5}}, zmm18",
        "vmovdqu16 zmm24{{k5}}, zmm26",

        // ====================== FOLD BLOCKS 4-7 ======================
        "vpcmpuw k5, zmm21, zmm20, 1",
        "vmovdqu16 zmm20{{k5}}, zmm21",
        "vmovdqu16 zmm28{{k5}}, zmm29",

        "vpcmpuw k5, zmm23, zmm22, 1",
        "vmovdqu16 zmm22{{k5}}, zmm23",
        "vmovdqu16 zmm30{{k5}}, zmm31",

        "vpcmpuw k5, zmm22, zmm20, 1",
        "vmovdqu16 zmm20{{k5}}, zmm22",
        "vmovdqu16 zmm28{{k5}}, zmm30",

        // ====================== FOLD HALVES ======================
        "vpcmpuw k5, zmm20, zmm16, 1",
        "vmovdqu16 zmm16{{k5}}, zmm20",
        "vmovdqu16 zmm24{{k5}}, zmm28",

        // ====================== HORIZONTAL REDUCE ======================
        "vshufi64x2 zmm0, zmm16, zmm16, 0x4E",
        "vshufi64x2 zmm1, zmm24, zmm24, 0x4E",
        "vpcmpuw k5, zmm0, zmm16, 1",
        "vmovdqu16 zmm16{{k5}}, zmm0",
        "vmovdqu16 zmm24{{k5}}, zmm1",

        "vshufi64x2 zmm0, zmm16, zmm16, 0xB1",
        "vshufi64x2 zmm1, zmm24, zmm24, 0xB1",
        "vpcmpuw k5, zmm0, zmm16, 1",
        "vmovdqu16 zmm16{{k5}}, zmm0",
        "vmovdqu16 zmm24{{k5}}, zmm1",

        "vpshufd zmm0, zmm16, 0x4E",
        "vpshufd zmm1, zmm24, 0x4E",
        "vpcmpuw k5, zmm0, zmm16, 1",
        "vmovdqu16 zmm16{{k5}}, zmm0",
        "vmovdqu16 zmm24{{k5}}, zmm1",

        "vpshuflw zmm0, zmm16, 0x4E",
        "vpshuflw zmm1, zmm24, 0x4E",
        "vpcmpuw k5, zmm0, zmm16, 1",
        "vmovdqu16 zmm16{{k5}}, zmm0",
        "vmovdqu16 zmm24{{k5}}, zmm1",

        "vpshuflw zmm0, zmm16, 0xB1",
        "vpshuflw zmm1, zmm24, 0xB1",
        "vpcmpuw k5, zmm0, zmm16, 1",
        "vmovdqu16 zmm16{{k5}}, zmm0",
        "vmovdqu16 zmm24{{k5}}, zmm1",

        "vpextrw {val:e}, xmm16, 0",
        "vpextrw {idx:e}, xmm24, 0",

        ptr = in(reg) arr.as_ptr(),
        start = in(reg) start as u32,
        end = in(reg) end as u32,
        index = sym INDEX_TABLE,
        offset32 = sym OFFSET_32,
        offset64 = sym OFFSET_64,
        offset96 = sym OFFSET_96,
        offset128 = sym OFFSET_128,
        offset160 = sym OFFSET_160,
        offset192 = sym OFFSET_192,
        offset224 = sym OFFSET_224,

        val = out(reg) val,
        idx = out(reg) idx,

        out("rax") _, out("rcx") _, out("rdx") _,
        out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
        out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _, out("xmm8") _,
        out("ymm10") _, out("ymm11") _, out("ymm12") _, out("ymm13") _, out("xmm14") _, out("ymm15") _,
        out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
        out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
        out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
        out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
        out("k1") _, out("k2") _, out("k3") _, out("k4") _, out("k5") _,
        options(nostack),
    ); }

    (val as u16, idx as u16)
}

#[repr(align(64))]
struct OffsetArray([u16; 32]);

static OFFSET_32: OffsetArray = OffsetArray([32; 32]);
static OFFSET_64: OffsetArray = OffsetArray([64; 32]);
static OFFSET_96: OffsetArray = OffsetArray([96; 32]);
static OFFSET_128: OffsetArray = OffsetArray([128; 32]);
static OFFSET_160: OffsetArray = OffsetArray([160; 32]);
static OFFSET_192: OffsetArray = OffsetArray([192; 32]);
static OFFSET_224: OffsetArray = OffsetArray([224; 32]);












#[cfg(test)]
mod tests {
    use super::*;
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn professor_mask_gen_complete(start: u8, end: u8) -> [u32; 8] {
    let mut masks = [0u32; 8];
    
    unsafe { asm!(
        "movzx eax, {start:l}",
        "movzx ecx, {end:l}",
        
        // ====================== DETECT ACTIVE BLOCKS FIRST ======================
        "mov     rdx, 0xE0C0A08060402000",
        "vmovq   xmm0, rdx",
        "vpmovzxbd ymm0, xmm0",              // ymm0 = [0, 32, 64, 96, 128, 160, 192, 224]
        
        "mov     rdx, 0xFFDFBF9F7F5F3F1F",
        "vmovq   xmm1, rdx",
        "vpmovzxbd ymm1, xmm1",              // ymm1 = [31, 63, 95, 127, 159, 191, 223, 255]
        
        "vpbroadcastd ymm2, ecx",            // broadcast end
        "vpbroadcastd ymm3, eax",            // broadcast start
        
        "vpcmpd k1, ymm0, ymm2, 2",          // k1: block_start <= end
        "vpcmpd k2, ymm1, ymm3, 5",          // k2: block_end >= start
        "kandb k3, k1, k2",                  // REMOVED knotb
        
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
        
        // Apply the mask: k3=1 means keep, k3=0 means zero
        "vmovdqa32 ymm15{{k3}}{{z}}, ymm15",

        "vmovdqu [{out}], ymm15",

        start = in(reg) start as u32,
        end = in(reg) end as u32,
        out = in(reg) masks.as_mut_ptr(),
        out("rax") _, out("rcx") _, out("rdx") _,
        out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
        out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _, out("xmm8") _,
        out("ymm10") _, out("ymm11") _, out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
        out("k1") _, out("k2") _, out("k3") _,
        options(nostack),
    ); }
    
    masks
}
#[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn professor_mask_gen_fixed(start: u8, end: u8) -> [u32; 8] {
        let mut masks = [0u32; 8];
        
        unsafe { asm!(
            // Work with bytes first
            "vpbroadcastb xmm4, {start:e}",     // xmm4 = [start; 16]
            "vpbroadcastb xmm5, {end:e}",       // xmm5 = [end; 16]

            "mov     rax, 0xE0C0A08060402000",
            "vmovq   xmm6, rax",                 // xmm6 = [0, 32, 64, 96, 128, 160, 192, 224] as bytes

            // Subtract while still bytes
            "vpsubusb xmm7, xmm4, xmm6",        // xmm7 = start - [0, 32, 64, ...] (saturating)
            "vpsubusb xmm8, xmm5, xmm6",        // xmm8 = end - [0, 32, 64, ...] (saturating)

            // NOW extend to dwords for the shift operations
            "vpcmpeqd ymm10, ymm10, ymm10",     // All 1s
            "vpmovzxbd ymm11, xmm7",            // Zero-extend start offsets to dwords
            "vpmovzxbd ymm12, xmm8",            // Zero-extend end offsets to dwords

            "vpsllvd  ymm13, ymm10, ymm11",     // 0xFFFFFFFF << start_offset
            "vpsllvd  ymm14, ymm10, ymm12",     // 0xFFFFFFFF << end_offset
            "vpslld   ymm14, ymm14, 1",         // 0xFFFFFFFF << (end_offset + 1)
            "vpandn   ymm15, ymm14, ymm13",     // (~ymm14) & ymm13

            "vmovdqu [{out}], ymm15",

            start = in(reg) start as u32,
            end = in(reg) end as u32,
            out = in(reg) masks.as_mut_ptr(),
            out("rax") _,
            out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _, out("xmm8") _,
            out("ymm10") _, out("ymm11") _, out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
            options(nostack),
        ); }
        
        masks
    }
    // Professor's mask generation - EXACT copy of his code
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn professor_mask_gen(start: u8, end: u8) -> [u32; 8] {
        let mut masks = [0u32; 8];
        
        unsafe { asm!(
            "vpbroadcastb xmm4, {start:e}",
            "vpbroadcastb xmm5, {end:e}",

            "mov     rax, 0xE0C0A08060402000",
            "vmovq   xmm6, rax",
            "vpmovzxbd ymm6, xmm6",

            "vpsubusb ymm7, ymm4, ymm6",
            "vpsubusb ymm8, ymm5, ymm6",

            "vpcmpeqd ymm10, ymm10, ymm10",
            "vpmovzxbd ymm11, xmm7",
            "vpmovzxbd ymm12, xmm8",

            "vpsllvd  ymm13, ymm10, ymm11",
            "vpsllvd  ymm14, ymm10, ymm12",
            "vpslld   ymm14, ymm14, 1",
            "vpandn   ymm15, ymm14, ymm13",

            "vmovdqu [{out}], ymm15",

            start = in(reg) start as u32,
            end = in(reg) end as u32,
            out = in(reg) masks.as_mut_ptr(),
            out("rax") _,
            out("xmm4") _, out("xmm5") _, out("xmm6") _,
            out("ymm7") _, out("ymm8") _, out("ymm10") _,
            out("ymm11") _, out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
            options(nostack),
        ); }
        
        masks
    }

    // Helper to compute expected mask for a block
    fn expected_mask(start: u8, end: u8, block: usize) -> u32 {
        let block_start = (block * 32) as u8;
        let block_end = block_start + 31;
        
        // Block completely outside range
        if end < block_start || start > block_end {
            return 0;
        }
        
        // Compute first and last bit positions within this block
        let first_bit = if start > block_start {
            (start - block_start) as u32
        } else {
            0
        };
        
        let last_bit = if end < block_end {
            (end - block_start) as u32
        } else {
            31
        };
        
        // Generate mask with bits [first_bit, last_bit] set
        let mut mask = 0u32;
        for bit in first_bit..=last_bit {
            mask |= 1u32 << bit;
        }
        mask
    }

    #[test]
    fn test_professor_masks_single_block() {
        unsafe {
            // Test 1: First block only, full
            let masks = professor_mask_gen(0, 31);
            println!("\nTest 1: start=0, end=31");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(0, 31, 0),
                expected_mask(0, 31, 1),
                expected_mask(0, 31, 2),
                expected_mask(0, 31, 3),
                expected_mask(0, 31, 4),
                expected_mask(0, 31, 5),
                expected_mask(0, 31, 6),
                expected_mask(0, 31, 7),
            ]);
            
            // Test 2: First block only, partial
            let masks = professor_mask_gen(10, 20);
            println!("\nTest 2: start=10, end=20");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(10, 20, 0),
                expected_mask(10, 20, 1),
                expected_mask(10, 20, 2),
                expected_mask(10, 20, 3),
                expected_mask(10, 20, 4),
                expected_mask(10, 20, 5),
                expected_mask(10, 20, 6),
                expected_mask(10, 20, 7),
            ]);
        }
    }

    #[test]
    fn test_professor_masks_spanning_blocks() {
        unsafe {
            // Test 3: Spanning blocks 0-1
            let masks = professor_mask_gen(20, 50);
            println!("\nTest 3: start=20, end=50");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(20, 50, 0),
                expected_mask(20, 50, 1),
                expected_mask(20, 50, 2),
                expected_mask(20, 50, 3),
                expected_mask(20, 50, 4),
                expected_mask(20, 50, 5),
                expected_mask(20, 50, 6),
                expected_mask(20, 50, 7),
            ]);
            
            // Test 4: Spanning many blocks
            let masks = professor_mask_gen(42, 150);
            println!("\nTest 4: start=42, end=150");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(42, 150, 0),
                expected_mask(42, 150, 1),
                expected_mask(42, 150, 2),
                expected_mask(42, 150, 3),
                expected_mask(42, 150, 4),
                expected_mask(42, 150, 5),
                expected_mask(42, 150, 6),
                expected_mask(42, 150, 7),
            ]);
        }
    }

    #[test]
    fn test_professor_masks_middle_blocks() {
        unsafe {
            // Test 5: Middle block only
            let masks = professor_mask_gen(64, 95);
            println!("\nTest 5: start=64, end=95");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(64, 95, 0),
                expected_mask(64, 95, 1),
                expected_mask(64, 95, 2),
                expected_mask(64, 95, 3),
                expected_mask(64, 95, 4),
                expected_mask(64, 95, 5),
                expected_mask(64, 95, 6),
                expected_mask(64, 95, 7),
            ]);
            
            // Test 6: Last block only
            let masks = professor_mask_gen(224, 255);
            println!("\nTest 6: start=224, end=255");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(224, 255, 0),
                expected_mask(224, 255, 1),
                expected_mask(224, 255, 2),
                expected_mask(224, 255, 3),
                expected_mask(224, 255, 4),
                expected_mask(224, 255, 5),
                expected_mask(224, 255, 6),
                expected_mask(224, 255, 7),
            ]);
        }
    }

    #[test]
    fn test_professor_masks_edge_cases() {
        unsafe {
            // Test 7: Single element
            let masks = professor_mask_gen(50, 50);
            println!("\nTest 7: start=50, end=50 (single element)");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(50, 50, 0),
                expected_mask(50, 50, 1),
                expected_mask(50, 50, 2),
                expected_mask(50, 50, 3),
                expected_mask(50, 50, 4),
                expected_mask(50, 50, 5),
                expected_mask(50, 50, 6),
                expected_mask(50, 50, 7),
            ]);
            
            // Test 8: Entire array
            let masks = professor_mask_gen(0, 255);
            println!("\nTest 8: start=0, end=255 (entire array)");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(0, 255, 0),
                expected_mask(0, 255, 1),
                expected_mask(0, 255, 2),
                expected_mask(0, 255, 3),
                expected_mask(0, 255, 4),
                expected_mask(0, 255, 5),
                expected_mask(0, 255, 6),
                expected_mask(0, 255, 7),
            ]);
        }
    }
    #[test]
    fn test_professor_masks_fixed() {
        unsafe {
            println!("\n=== FIXED VERSION ===");
            
            let masks = professor_mask_gen_fixed(0, 31);
            println!("\nTest 1: start=0, end=31");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(0, 31, 0),
                expected_mask(0, 31, 1),
                expected_mask(0, 31, 2),
                expected_mask(0, 31, 3),
                expected_mask(0, 31, 4),
                expected_mask(0, 31, 5),
                expected_mask(0, 31, 6),
                expected_mask(0, 31, 7),
            ]);
            
            let masks = professor_mask_gen_fixed(42, 150);
            println!("\nTest 4: start=42, end=150");
            println!("Generated: {:08X?}", masks);
            println!("Expected:  {:08X?}", [
                expected_mask(42, 150, 0),
                expected_mask(42, 150, 1),
                expected_mask(42, 150, 2),
                expected_mask(42, 150, 3),
                expected_mask(42, 150, 4),
                expected_mask(42, 150, 5),
                expected_mask(42, 150, 6),
                expected_mask(42, 150, 7),
            ]);
        }
    }
    #[test]
    fn test_professor_masks_complete() {
        unsafe {
            println!("\n=== COMPLETE FIXED VERSION ===");
            
            // Test all cases
            let test_cases = [
                (0, 31, "First block only"),
                (10, 20, "Partial first block"),
                (20, 50, "Spanning blocks 0-1"),
                (42, 150, "Spanning blocks 1-4"),
                (64, 95, "Middle block only"),
                (224, 255, "Last block only"),
                (50, 50, "Single element"),
                (0, 255, "Entire array"),
            ];
            
            for &(start, end, desc) in &test_cases {
                let masks = professor_mask_gen_complete(start, end);
                let expected = [
                    expected_mask(start, end, 0),
                    expected_mask(start, end, 1),
                    expected_mask(start, end, 2),
                    expected_mask(start, end, 3),
                    expected_mask(start, end, 4),
                    expected_mask(start, end, 5),
                    expected_mask(start, end, 6),
                    expected_mask(start, end, 7),
                ];
                
                println!("\n{}: start={}, end={}", desc, start, end);
                println!("Generated: {:08X?}", masks);
                println!("Expected:  {:08X?}", expected);
                
                assert_eq!(masks, expected, "Mismatch for {}", desc);
            }
            
            println!("\n✓ All tests passed!");
        }
    }
}


