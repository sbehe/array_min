use core::arch::x86_64::*;
use core::arch::asm;

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
    let mask = ((1u32 << (offset + 1)) - 1) as __mmask32;
    (block, mask)
}

unsafe fn dump_u16_512(label: &str, v: __m512i) {
    let mut arr = [0u16; 32];
    unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, v) };
    println!("{}: {:?}", label, arr);
}

#[inline(always)]
unsafe fn dump_u16_256(label: &str, v: __m256i) {
    let mut arr = [0u16; 16];
    unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut _, v) };
    println!("{}: {:?}", label, arr);
}

#[inline(always)]
unsafe fn dump_u16_128(label: &str, v: __m128i) {
    let mut arr = [0u16; 8];
    unsafe { _mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    println!("{}: {:?}", label, arr);
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn horizontal_reduce_min_u16(
    vmin: __m512i,
    vidx: __m512i,
) -> (u16, usize) {

    #[cfg(feature = "trace_avx_search")]
    {println!("===== START REDUCTION =====");
    dump_u16_512("Initial vmin", vmin);
    dump_u16_512("Initial vidx", vidx);
    }
    // =========================
    // 512 → 256
    // =========================

    let vmin_hi = _mm512_extracti64x4_epi64(vmin, 1);
    let vidx_hi = _mm512_extracti64x4_epi64(vidx, 1);

    let vmin_lo = _mm512_castsi512_si256(vmin);
    let vidx_lo = _mm512_castsi512_si256(vidx);


    #[cfg(feature = "trace_avx_search")]
    {
    dump_u16_256("vmin_lo (512→256)", vmin_lo);
    dump_u16_256("vmin_hi (512→256)", vmin_hi);
    }

    let mask = unsafe { _mm256_cmp_epu16_mask(vmin_hi, vmin_lo, _MM_CMPINT_LT) };
    
    #[cfg(feature = "trace_avx_search")]
    println!("Mask 512→256: {:016b}", mask);

    let vmin256 = unsafe { _mm256_mask_mov_epi16(vmin_lo, mask, vmin_hi) };
    let vidx256 = unsafe { _mm256_mask_mov_epi16(vidx_lo, mask, vidx_hi) };


    #[cfg(feature = "trace_avx_search")]
    {dump_u16_256("vmin after 512→256", vmin256);
    dump_u16_256("vidx after 512→256", vidx256);
    }
    // =========================
    // 256 → 128
    // =========================

    let vmin_hi = _mm256_extracti128_si256(vmin256, 1);
    let vidx_hi = _mm256_extracti128_si256(vidx256, 1);

    let vmin_lo = _mm256_castsi256_si128(vmin256);
    let vidx_lo = _mm256_castsi256_si128(vidx256);


    #[cfg(feature = "trace_avx_search")]
    {dump_u16_128("vmin_lo (256→128)", vmin_lo);
    dump_u16_128("vmin_hi (256→128)", vmin_hi);
    }
    let mask = unsafe { _mm_cmp_epu16_mask(vmin_hi, vmin_lo, _MM_CMPINT_LT) };
    #[cfg(feature = "trace_avx_search")]
    println!("Mask 256→128: {:08b}", mask);

    let mut vmin128 = unsafe { _mm_mask_mov_epi16(vmin_lo, mask, vmin_hi) };
    let mut vidx128 = unsafe { _mm_mask_mov_epi16(vidx_lo, mask, vidx_hi) };

    #[cfg(feature = "trace_avx_search")]
    {dump_u16_128("vmin after 256→128", vmin128);
    dump_u16_128("vidx after 256→128", vidx128);
    }

    // =========================
    // 128-bit lane reduction
    // =========================

    #[cfg(feature = "trace_avx_search")]
    {println!("--- 8 → 4 reduction ---");
    dump_u16_128("before", vmin128);
    }
    let tval = _mm_srli_si128(vmin128, 8); // shift by 4 u16
    let tidx = _mm_srli_si128(vidx128, 8);


    #[cfg(feature = "trace_avx_search")]
    dump_u16_128("shifted vals", tval);

    let mask = unsafe { _mm_cmplt_epu16_mask(tval, vmin128) };
    
    #[cfg(feature = "trace_avx_search")]
    println!("mask: {:08b}", mask);

    vmin128 = unsafe { _mm_mask_mov_epi16(vmin128, mask, tval) };
    vidx128 = unsafe { _mm_mask_mov_epi16(vidx128, mask, tidx) };


    #[cfg(feature = "trace_avx_search")]
    {dump_u16_128("after 8→4 vmin", vmin128);
    dump_u16_128("after 8→4 vidx", vidx128);
    }

    #[cfg(feature = "trace_avx_search")]
    {println!("--- 4 → 2 reduction ---");
    dump_u16_128("before", vmin128);
    }
    let tval = _mm_srli_si128(vmin128, 4); // shift by 2 u16
    let tidx = _mm_srli_si128(vidx128, 4);


    #[cfg(feature = "trace_avx_search")]
    dump_u16_128("shifted vals", tval);

    let mask = unsafe { _mm_cmplt_epu16_mask(tval, vmin128) };
    
    #[cfg(feature = "trace_avx_search")]
    println!("mask: {:08b}", mask);

    vmin128 = unsafe { _mm_mask_mov_epi16(vmin128, mask, tval) };
    vidx128 = unsafe { _mm_mask_mov_epi16(vidx128, mask, tidx) };


    #[cfg(feature = "trace_avx_search")]
    {dump_u16_128("after 4→2 vmin", vmin128);
    dump_u16_128("after 4→2 vidx", vidx128);
    }


    #[cfg(feature = "trace_avx_search")]
    {println!("--- 2 → 1 reduction ---");
    dump_u16_128("before", vmin128);
    }
    let tval = _mm_srli_si128(vmin128, 2); // shift by 1 u16
    let tidx = _mm_srli_si128(vidx128, 2);


    #[cfg(feature = "trace_avx_search")]
    dump_u16_128("shifted vals", tval);

    let mask = unsafe { _mm_cmplt_epu16_mask(tval, vmin128) };
    
    #[cfg(feature = "trace_avx_search")]
    println!("mask: {:08b}", mask);

    vmin128 = unsafe { _mm_mask_mov_epi16(vmin128, mask, tval) };
    vidx128 = unsafe { _mm_mask_mov_epi16(vidx128, mask, tidx) };

    
    #[cfg(feature = "trace_avx_search")]
    {dump_u16_128("after 2→1 vmin", vmin128);
    dump_u16_128("after 2→1 vidx", vidx128);
    }

    let min_val = _mm_extract_epi16(vmin128, 0) as u16;
    let min_idx = _mm_extract_epi16(vidx128, 0) as usize;

    
    #[cfg(feature = "trace_avx_search")]
    {println!("FINAL min_val: {}", min_val);
    println!("FINAL min_idx: {}", min_idx);
    println!("===== END REDUCTION =====");
    }
    (min_val, min_idx)
}

pub fn scalar_min<const M: usize>(arr: &[u16; M], start_index: usize, end_index: usize) -> Option<(u16, usize)> {
    #[cfg(feature = "trace_avx_search")]
    println!("Using scalar min for range {}..{}", start_index, end_index);
    if arr.is_empty() {
        return None;
    }
    let mut min_val = u16::MAX;
    let mut min_idx = 0;
    for i in start_index..=end_index {
        if arr[i] < min_val {
            min_val = arr[i];
            min_idx = i;
        }
    }
    Some((min_val, min_idx))
}


#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn minindex_u16_specialized<const N: usize>(
    arr: &[u16; N],
    start: usize,
    end: usize,
) -> (u16, usize) {

    assert!(N % 32 == 0);
    assert!(start <= end);
    assert!(end < N);

    let (start_block, start_mask) = compute_start(start);
    let (end_block, end_mask) = compute_end(end);

    let blocks = end_block - start_block + 1;

    let ptr = unsafe { arr.as_ptr().add(start_block * 32) };

    let (val, idx) = match blocks {

        1 => {
            let mask = start_mask & end_mask;
            unsafe { kernel_1(ptr, mask) }
        }

        _ => {
            panic!("Not yet implemented: blocks = {}", blocks);
        }
    };

    (val, idx + start_block * 32)
}

#[repr(align(64))]
struct AlignedArray([u16; 32]);

static INDEX_TABLE: AlignedArray = AlignedArray([
    0,1,2,3,4,5,6,7,
    8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,
    24,25,26,27,28,29,30,31
]);

static INF_VEC: AlignedArray = AlignedArray([0xFFFF; 32]);


#[cfg(feature = "trace_avx_search")]
fn dump(label: &str, v: &[u16;32]) {
    println!("{}: {:?}", label, v);
}

#[cfg(not(feature = "trace_avx_search"))]
fn dump(_: &str, _: &[u16;32]) {}

#[target_feature(enable="avx512f,avx512bw")]
pub unsafe fn kernel_1(ptr: *const u16, mask: u32) -> (u16, usize) {

    let mut stage0 = [0u16;32];
    let mut stage1 = [0u16;32];
    let mut stage2 = [0u16;32];
    let mut stage3 = [0u16;32];
    let mut stage4 = [0u16;32];
    let mut stage5 = [0u16;32];
    let mut stage6 = [0u16;32];
    let mut stage7 = [0u16;32];
    let mut stage8 = [0u16;32];
    let mut stage9 = [0u16;32];
    let mut stage10 = [0u16;32];
    let mut stage11 = [0u16;32];
    let mut stage12 = [0u16;32];
    let mut stage13 = [0u16;32];
    let mut stage14 = [0u16;32];
    let mut stage15 = [0u16;32];
    let mut stage16 = [0u16;32];
    let mut stage17 = [0u16;32];

    let val: u32;
    let idx: u32;

    unsafe { asm!(
        // load mask
        "kmovd k1, {mask:e}",

        // masked load (zero masked lanes)
        "vmovdqu16 zmm2{{k1}}{{z}}, [{ptr}]",

        // load INF vector
        "vmovdqu16 zmm0, [rip + {inf}]",

        // set masked lanes = 0xFFFF
        "vmovdqu16 zmm0{{k1}}, zmm2",

        // load index vector
        "vmovdqu16 zmm1, [rip + {index}]",


        // init working vectors
        "vmovdqa64 zmm4, zmm0",
        "vmovdqa64 zmm5, zmm1",

        // "vmovdqu16 [{dbg0}], zmm4",
        // // initial index
        // "vmovdqu16 [{dbg1}], zmm5",

        // reduction step 1
        "vshufi64x2 zmm6, zmm4, zmm4, 0x4E",
        "vshufi64x2 zmm7, zmm5, zmm5, 0x4E",

        // "vmovdqu16 [{dbg2}], zmm6",
        // "vmovdqu16 [{dbg3}], zmm7",

        "vpcmpuw k3, zmm6, zmm4, 1",

        "vmovdqu16 zmm4{{k3}}, zmm6",
        "vmovdqu16 zmm5{{k3}}, zmm7",

        // dump index vector
        // "vmovdqu16 [{dbg4}], zmm4",
        // "vmovdqu16 [{dbg5}], zmm5",

        // reduction step 2
        "vshufi64x2 zmm6, zmm4, zmm4, 0xB1",
        "vshufi64x2 zmm7, zmm5, zmm5, 0xB1",

        //"vmovdqu16 [{dbg6}], zmm6",
        //"vmovdqu16 [{dbg7}], zmm7",

        "vpcmpuw k3, zmm6, zmm4, 1",

        "vmovdqu16 zmm4{{k3}}, zmm6",
        "vmovdqu16 zmm5{{k3}}, zmm7",

         "vmovdqu16 [{dbg8}], zmm4",

        // dump index vector
        // "vmovdqu16 [{dbg9}], zmm5",

        // reduction step 3
        "vpshufd zmm6, zmm4, 0x4E",
        "vpshufd zmm7, zmm5, 0x4E",

        "vmovdqu16 [{dbg10}], zmm6",
        "vmovdqu16 [{dbg11}], zmm7",

        "vpcmpuw k3, zmm6, zmm4, 1",

        "vmovdqu16 zmm4{{k3}}, zmm6",
        "vmovdqu16 zmm5{{k3}}, zmm7",

        "vmovdqu16 [{dbg12}], zmm4",

        // dump index vector
        "vmovdqu16 [{dbg13}], zmm5",

        // reduction step 4
        "vpshuflw zmm6, zmm4, 0x4E",
        "vpshuflw zmm7, zmm5, 0x4E",

        "vmovdqu16 [{dbg14}], zmm6",
        "vmovdqu16 [{dbg15}], zmm7",

        "vpcmpuw k3, zmm6, zmm4, 1",

        "vmovdqu16 zmm4{{k3}}, zmm6",
        "vmovdqu16 zmm5{{k3}}, zmm7",

        "vmovdqu16 [{dbg16}], zmm4",

        // dump index vector
        "vmovdqu16 [{dbg17}], zmm5",

        // reduction step 5
        "vpshuflw zmm6, zmm4, 0xB1",
        "vpshuflw zmm7, zmm5, 0xB1",

        "vmovdqu16 [{dbg14}], zmm6",
        "vmovdqu16 [{dbg15}], zmm7",

        "vpcmpuw k3, zmm6, zmm4, 1",

        "vmovdqu16 zmm4{{k3}}, zmm6",
        "vmovdqu16 zmm5{{k3}}, zmm7",

        "vmovdqu16 [{dbg16}], zmm4",

        // dump index vector
        "vmovdqu16 [{dbg17}], zmm5",

        // extract result
        "vpextrw {val:e}, xmm4, 0",
        "vpextrw {idx:e}, xmm5, 0",

        ptr = in(reg) ptr,
        mask = in(reg) mask,
        index = sym INDEX_TABLE,
        inf = sym INF_VEC,

        // dbg0 = in(reg) stage0.as_mut_ptr(),
        // dbg1 = in(reg) stage1.as_mut_ptr(),
        // dbg2 = in(reg) stage2.as_mut_ptr(),
        // dbg3 = in(reg) stage3.as_mut_ptr(),
        // dbg4 = in(reg) stage4.as_mut_ptr(),
        // dbg5 = in(reg) stage5.as_mut_ptr(),
        // dbg6 = in(reg) stage6.as_mut_ptr(),
        // dbg7 = in(reg) stage7.as_mut_ptr(),
         dbg8 = in(reg) stage8.as_mut_ptr(),
        // dbg9 = in(reg) stage9.as_mut_ptr(),
        dbg10 = in(reg) stage10.as_mut_ptr(),
        dbg11 = in(reg) stage11.as_mut_ptr(),
        dbg12 = in(reg) stage12.as_mut_ptr(),
        dbg13 = in(reg) stage13.as_mut_ptr(),
        dbg14 = in(reg) stage14.as_mut_ptr(),
        dbg15 = in(reg) stage15.as_mut_ptr(),
        dbg16 = in(reg) stage16.as_mut_ptr(),
        dbg17 = in(reg) stage17.as_mut_ptr(),

        val = out(reg) val,
        idx = out(reg) idx,

        out("zmm0") _, out("zmm1") _, out("zmm2") _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("k1") _, out("k2") _, out("k3") _,

        options(nostack)
    ); }

    //dump("initial values \t \t \t      ", &stage0);
    //dump("initial index", &stage1);
    //dump("value vector after shuffle.  - step 1", &stage2);
    //dump("index vector after shuffle", &stage3);
    //dump("value vector after reduction - step 1", &stage4);
    //dump("index vector after reduction 1", &stage5);
    //dump("value vector after shuffle.  - step 2", &stage6);
    //dump("index vector after shuffle", &stage6);
    dump("value vector after reduction - step 2", &stage8);
    //dump("index vector after reduction 2", &stage9);
    dump("value vector after shuffle.  - step 3", &stage10);
    //dump("index vector after shuffle", &stage11);
    dump("value vector after reduction - step 3", &stage12);
    //dump("index vector after reduction 3", &stage13);
    dump("value vector after shuffle.  - step 4", &stage14);
    //dump("index vector after shuffle", &stage15);
    dump("value vector after reduction - step 4", &stage16);
    //dump("index vector after reduction 4", &stage17);

    (val as u16, idx as usize)
}