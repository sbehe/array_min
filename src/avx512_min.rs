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

static INF_VEC: AlignedArray = AlignedArray([0xFFFF; 32]);


#[inline(always)]
pub fn compute_start(i: usize) -> (usize, __mmask32) {
    let block = i >> 5;
    let offset = i & 31;
    let mask = (!0u32 << offset) as __mmask32;
    (block, mask)
}

#[inline(always)]
pub fn compute_end(j: usize) -> (usize, __mmask32) {
    let block = j >> 5;
    let offset = j & 31;
    let mask = ((1u64 << (offset + 1)) - 1) as u32 as __mmask32;
    (block, mask)
}

pub fn scalar_min<const M: usize>(arr: &[u16; M], start_index: usize, end_index: usize) -> (u16, usize) {
    #[cfg(feature = "trace_avx_search")]
    println!("Using scalar min for range {}..{}", start_index, end_index);
    assert!(!arr.is_empty()); 
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
pub unsafe fn minindex_u16<const N: usize>(
    arr: &[u16; N],
    start: usize,
    end: usize,
) -> (u16, usize) {

    assert!(N % 32 == 0);
    assert!(start <= end);
    assert!(end < N);

    //println!("computing start mask");
    let (start_block, start_mask) = compute_start(start);
    //println!("computing end mask");
    let (end_block, end_mask) = compute_end(end);

    let blocks = end_block - start_block + 1;

    let ptr = unsafe { arr.as_ptr().add(start_block * 32) };

    let (val, idx) = {
        
        // no start mask and no end mask
        if ((start % 32) == 0) && ((end % 32) == 31) {
            
            unsafe { kernel_many_unmasked(ptr, blocks) }
        } else {
            //println!("at least 1 mask present");
            if blocks == 1 {
                unsafe { kernel_1_masked(ptr, start_mask & end_mask) }
            } else {
                unsafe { kernel_many(ptr, blocks, start_mask, end_mask) }
            }
        }
    };

    (val, idx + start_block * 32)
}


#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn fold_1_block(
    ptr: *const u16,
    base_idx: u32,
    zmm4: &mut __m512i,
    zmm5: &mut __m512i,
) {
    unsafe { asm!(
        "vmovdqu16 zmm6, [{ptr}]",

        "vmovd xmm0, {base:e}",
        "vpbroadcastw zmm7, xmm0",
        "vmovdqu16 zmm8, [rip + {index}]",
        "vpaddw zmm7, zmm8, zmm7",

        "vpcmpuw k1, zmm6, zmm4, 1",
        "vmovdqu16 zmm4{{k1}}, zmm6",
        "vmovdqu16 zmm5{{k1}}, zmm7",

        ptr   = in(reg) ptr,
        base  = in(reg) base_idx,
        index = sym INDEX_TABLE,

        inout("zmm4") *zmm4,
        inout("zmm5") *zmm5,
        out("xmm0") _,
        out("zmm6") _, out("zmm7") _, out("zmm8") _,
        out("k1") _,
        options(nostack),
    ); }
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn fold_4_blocks(
    ptr: *const u16,
    base_idx: u32,
    zmm4: &mut __m512i,
    zmm5: &mut __m512i,
) {
    unsafe { asm!(
        "vmovdqu16 zmm6,  [{ptr}]",
        "vmovdqu16 zmm8,  [{ptr} + 64]",
        "vmovdqu16 zmm9,  [{ptr} + 128]",
        "vmovdqu16 zmm10, [{ptr} + 192]",

        "vmovdqu16 zmm11, [rip + {index}]",

        "vmovd xmm0, {base:e}",
        "vpbroadcastw zmm2, xmm0",
        "vpaddw zmm7, zmm11, zmm2",

        "add {base:e}, 32",
        "vmovd xmm0, {base:e}",
        "vpbroadcastw zmm2, xmm0",
        "vpaddw zmm13, zmm11, zmm2",

        "add {base:e}, 32",
        "vmovd xmm0, {base:e}",
        "vpbroadcastw zmm2, xmm0",
        "vpaddw zmm14, zmm11, zmm2",

        "add {base:e}, 32",
        "vmovd xmm0, {base:e}",
        "vpbroadcastw zmm2, xmm0",
        "vpaddw zmm15, zmm11, zmm2",

        "vpcmpuw k1, zmm8,  zmm6,  1",
        "vmovdqu16 zmm6{{k1}},  zmm8",
        "vmovdqu16 zmm7{{k1}},  zmm13",

        "vpcmpuw k1, zmm10, zmm9,  1",
        "vmovdqu16 zmm9{{k1}},  zmm10",
        "vmovdqu16 zmm14{{k1}}, zmm15",

        "vpcmpuw k1, zmm9,  zmm6,  1",
        "vmovdqu16 zmm6{{k1}},  zmm9",
        "vmovdqu16 zmm7{{k1}},  zmm14",

        "vpcmpuw k1, zmm6, zmm4, 1",
        "vmovdqu16 zmm4{{k1}}, zmm6",
        "vmovdqu16 zmm5{{k1}}, zmm7",

        ptr   = in(reg) ptr,
        base  = inout(reg) base_idx => _,
        index = sym INDEX_TABLE,

        inout("zmm4") *zmm4,
        inout("zmm5") *zmm5,
        out("xmm0") _,
        out("zmm2") _,
        out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _,
        out("zmm10") _, out("zmm11") _,
        out("zmm13") _, out("zmm14") _, out("zmm15") _,
        out("k1") _,
        options(nostack),
    ); }
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn fold_1_block_masked(
    ptr: *const u16,
    mask: u32,
    base_idx: u32,
    zmm4: &mut __m512i,
    zmm5: &mut __m512i,
) {
    unsafe { asm!(
        "kmovd k2, {mask:e}",
        "vmovdqu16 zmm6{{k2}}{{z}}, [{ptr}]",
        "vmovdqu16 zmm8, [rip + {inf}]",
        "vmovdqu16 zmm8{{k2}}, zmm6",

        "vmovd xmm0, {base:e}",
        "vpbroadcastw zmm7, xmm0",
        "vmovdqu16 zmm9, [rip + {index}]",
        "vpaddw zmm7, zmm9, zmm7",

        "vpcmpuw k1, zmm8, zmm4, 1",
        "vmovdqu16 zmm4{{k1}}, zmm8",
        "vmovdqu16 zmm5{{k1}}, zmm7",

        ptr   = in(reg) ptr,
        mask  = in(reg) mask,
        base  = in(reg) base_idx,
        index = sym INDEX_TABLE,
        inf   = sym INF_VEC,

        inout("zmm4") *zmm4,
        inout("zmm5") *zmm5,
        out("xmm0") _,
        out("zmm6") _, out("zmm7") _, out("zmm8") _, out("zmm9") _,
        out("k1") _, out("k2") _,
        options(nostack),
    ); }
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn horizontal_reduce(zmm4: __m512i, zmm5: __m512i) -> (u16, usize) {
    let val: u32;
    let idx: u32;
    unsafe {asm!(
        "vshufi64x2 zmm2, zmm4, zmm4, 0x4E",
        "vshufi64x2 zmm3, zmm5, zmm5, 0x4E",
        "vpcmpuw k1, zmm2, zmm4, 1",
        "vmovdqu16 zmm4{{k1}}, zmm2",
        "vmovdqu16 zmm5{{k1}}, zmm3",

        "vshufi64x2 zmm2, zmm4, zmm4, 0xB1",
        "vshufi64x2 zmm3, zmm5, zmm5, 0xB1",
        "vpcmpuw k1, zmm2, zmm4, 1",
        "vmovdqu16 zmm4{{k1}}, zmm2",
        "vmovdqu16 zmm5{{k1}}, zmm3",

        "vpshufd zmm2, zmm4, 0x4E",
        "vpshufd zmm3, zmm5, 0x4E",
        "vpcmpuw k1, zmm2, zmm4, 1",
        "vmovdqu16 zmm4{{k1}}, zmm2",
        "vmovdqu16 zmm5{{k1}}, zmm3",

        "vpshuflw zmm2, zmm4, 0x4E",
        "vpshuflw zmm3, zmm5, 0x4E",
        "vpcmpuw k1, zmm2, zmm4, 1",
        "vmovdqu16 zmm4{{k1}}, zmm2",
        "vmovdqu16 zmm5{{k1}}, zmm3",

        "vpshuflw zmm2, zmm4, 0xB1",
        
        "vpshuflw zmm3, zmm5, 0xB1",
        "vpcmpuw k1, zmm2, zmm4, 1",
        "vmovdqu16 zmm4{{k1}}, zmm2",
        "vmovdqu16 zmm5{{k1}}, zmm3",

        "vpextrw {val:e}, xmm4, 0",
        "vpextrw {idx:e}, xmm5, 0",

        inout("zmm4") zmm4 => _,
        inout("zmm5") zmm5 => _,
        out("zmm2") _, out("zmm3") _,
        val = out(reg) val,
        idx = out(reg) idx,
        out("k1") _,
        options(nostack),
    ); }
    (val as u16, idx as usize)
}
// ---- public kernels ------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
pub unsafe fn kernel_many_unmasked(ptr: *const u16, num_blocks: usize) -> (u16, usize) {
    debug_assert!(num_blocks >= 1);

    let (mut zmm4, mut zmm5): (__m512i, __m512i);
    unsafe { asm!(
        "vmovdqu16 zmm4, [{ptr}]",
        "vmovdqu16 zmm5, [rip + {index}]",
        ptr   = in(reg) ptr,
        index = sym INDEX_TABLE,
        out("zmm4") zmm4,
        out("zmm5") zmm5,
        options(nostack),
    ); }

    let mut block = 1usize;
    let remaining = num_blocks - 1;

    let rem = remaining % 4;
    for _ in 0..rem {
        unsafe { fold_1_block(ptr.add(block * 32), (block * 32) as u32, &mut zmm4, &mut zmm5);}
        block += 1;
    }

    while block < num_blocks {
        unsafe { fold_4_blocks(ptr.add(block * 32), (block * 32) as u32, &mut zmm4, &mut zmm5);}
        block += 4;
    }

    unsafe { horizontal_reduce(zmm4, zmm5)}
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn kernel_many(
    ptr: *const u16,       // points to start_block (already offset)
    num_blocks: usize,     // total blocks including start and end
    start_mask: u32,
    end_mask: u32,
) -> (u16, usize) {        // returned index is relative to ptr
    debug_assert!(num_blocks >= 2);

    let (mut zmm4, mut zmm5): (__m512i, __m512i);
    unsafe { asm!(
        "vmovdqu16 zmm4, [rip + {inf}]",
        "vmovdqu16 zmm5, [rip + {index}]",
        inf   = sym INF_VEC,
        index = sym INDEX_TABLE,
        out("zmm4") zmm4,
        out("zmm5") zmm5,
        options(nostack),
    ); }

    // start block: relative block 0, base_idx = 0
    unsafe { fold_1_block_masked(ptr, start_mask, 0, &mut zmm4, &mut zmm5);}

    // end block: relative block (num_blocks-1)
    let end_rel = num_blocks - 1;
    unsafe { fold_1_block_masked(
        ptr.add(end_rel * 32),
        end_mask,
        (end_rel * 32) as u32,
        &mut zmm4,
        &mut zmm5,
    );}

    // middle blocks: 1 .. num_blocks-1 (exclusive), all unmasked
    let mut block = 1usize;
    let mid_count = end_rel - 1;   // = num_blocks - 2

    let rem = mid_count % 4;
    for _ in 0..rem {
        unsafe { fold_1_block(ptr.add(block * 32), (block * 32) as u32, &mut zmm4, &mut zmm5);}
        block += 1;
    }
    while block < end_rel {
        unsafe { fold_4_blocks(ptr.add(block * 32), (block * 32) as u32, &mut zmm4, &mut zmm5);}
        block += 4;
    }

    unsafe { horizontal_reduce(zmm4, zmm5)}
}

#[target_feature(enable="avx512f,avx512bw")]
pub unsafe fn kernel_1_masked(ptr: *const u16, mask: u32) -> (u16, usize) {
    let (zmm4, zmm5): (__m512i, __m512i);

    unsafe { asm!(
        "kmovd k1, {mask:e}",
        "vmovdqu16 zmm2{{k1}}{{z}}, [{ptr}]",
        "vmovdqu16 zmm4, [rip + {inf}]",
        "vmovdqu16 zmm4{{k1}}, zmm2",
        "vmovdqu16 zmm5, [rip + {index}]",

        ptr   = in(reg) ptr,
        mask  = in(reg) mask,
        index = sym INDEX_TABLE,
        inf   = sym INF_VEC,

        out("zmm2") _,
        out("k1") _,
        out("zmm4") zmm4,
        out("zmm5") zmm5,

        options(nostack),
    );}

    // Debug: print the first 32 u16 lanes of zmm4 and zmm5
    // let vals: [u16; 32] = std::mem::transmute(zmm4);
    // let idxs: [u16; 32] = std::mem::transmute(zmm5);
    // println!("zmm4 (vals): {:?}", &vals[..32]);
    // println!("zmm5 (idxs): {:?}", &idxs[..32]);
    unsafe {horizontal_reduce(zmm4, zmm5)}
}