use core::arch::x86_64::*;
#[allow(unsafe_code)]
/// Reduce a 512-bit vector (16 i32s) to scalar min
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn arraymin_reduce_u16(v: __m512i) -> u16 {
    let mut tmp = [0u16; 32];
    unsafe { _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v) };
    tmp.into_iter().min().unwrap()
}

/// Find minimum value in arbitrary sized array
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn arraymin_u16(array: *const u16, size: usize) -> u16 {
    if size == 0 {
        return u16::MAX;
    }

    let mut i = 0;

    let mut vmin = _mm512_set1_epi16(unsafe { *array } as i16);

    while i + 32 <= size {
        let v = unsafe { _mm512_loadu_si512(array.add(i) as *const _) };
        vmin = _mm512_min_epu16(vmin, v);
        i += 32;
    }

    let mut min_val = unsafe { arraymin_reduce_u16(vmin) };

    while i < size {
        min_val = min_val.min(unsafe { *array.add(i) });
        i += 1;
    }

    min_val
}

/// Optimized version for exactly 128 elements
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn arraymin128_u16(array: *const u16) -> u16 {
    let v0 = unsafe { _mm512_loadu_si512(array.add(0) as *const _) };
    let v1 = unsafe { _mm512_loadu_si512(array.add(32) as *const _) };
    let v2 = unsafe { _mm512_loadu_si512(array.add(64) as *const _) };
    let v3 = unsafe { _mm512_loadu_si512(array.add(96) as *const _) };

    let m0 = _mm512_min_epu16(v0, v1);
    let m1 = _mm512_min_epu16(v2, v3);
    let m = _mm512_min_epu16(m0, m1);

    unsafe { arraymin_reduce_u16(m) }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn horizontal_reduce_min_u16_v1(
    mut vmin: __m512i,
    mut vidx: __m512i,
) -> (usize, u16) {
    // 512 → 256
    let tval = _mm512_shuffle_i64x2(vmin, vmin, 0x4E);
    let tidx = _mm512_shuffle_i64x2(vidx, vidx, 0x4E);
    let mask = _mm512_cmp_epu16_mask(tval, vmin, _MM_CMPINT_LT);
    vmin = _mm512_mask_mov_epi16(vmin, mask, tval);
    vidx = _mm512_mask_mov_epi16(vidx, mask, tidx);

    // 256 → 128
    let tval = _mm512_shuffle_i64x2(vmin, vmin, 0xB1);
    let tidx = _mm512_shuffle_i64x2(vidx, vidx, 0xB1);
    let mask = _mm512_cmp_epu16_mask(tval, vmin, _MM_CMPINT_LT);
    vmin = _mm512_mask_mov_epi16(vmin, mask, tval);
    vidx = _mm512_mask_mov_epi16(vidx, mask, tidx);

    // within lanes
    let tval = _mm512_shuffle_epi32(vmin, 0x4E);
    let tidx = _mm512_shuffle_epi32(vidx, 0x4E);
    let mask = _mm512_cmp_epu16_mask(tval, vmin, _MM_CMPINT_LT);
    vmin = _mm512_mask_mov_epi16(vmin, mask, tval);
    vidx = _mm512_mask_mov_epi16(vidx, mask, tidx);

    let tval = _mm512_shuffle_epi32(vmin, 0xB1);
    let tidx = _mm512_shuffle_epi32(vidx, 0xB1);
    let mask = _mm512_cmp_epu16_mask(tval, vmin, _MM_CMPINT_LT);
    vmin = _mm512_mask_mov_epi16(vmin, mask, tval);
    vidx = _mm512_mask_mov_epi16(vidx, mask, tidx);

    let min_val = _mm_extract_epi16(_mm512_castsi512_si128(vmin), 0) as u16;
    let min_index = _mm_extract_epi16(_mm512_castsi512_si128(vidx), 0) as usize;

    (min_index, min_val)
}

unsafe fn dump_u16_512(label: &str, v: __m512i) {
    let mut arr = [0u16; 32];
    _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, v);
    println!("{}: {:?}", label, arr);
}

#[inline(always)]
unsafe fn dump_u16_256(label: &str, v: __m256i) {
    let mut arr = [0u16; 16];
    _mm256_storeu_si256(arr.as_mut_ptr() as *mut _, v);
    println!("{}: {:?}", label, arr);
}

#[inline(always)]
unsafe fn dump_u16_128(label: &str, v: __m128i) {
    let mut arr = [0u16; 8];
    _mm_storeu_si128(arr.as_mut_ptr() as *mut _, v);
    println!("{}: {:?}", label, arr);
}

#[cfg(feature = "trace")]
macro_rules! trace_println {
    ($($arg:tt)*) => {
        println!($($arg)*);
    };
}

#[cfg(not(feature = "trace"))]
macro_rules! trace_println {
    ($($arg:tt)*) => {};
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn horizontal_reduce_min_u16(
    mut vmin: __m512i,
    mut vidx: __m512i,
) -> (usize, u16) {


    #[cfg(feature = "trace")]
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


    #[cfg(feature = "trace")]
    {
    dump_u16_256("vmin_lo (512→256)", vmin_lo);
    dump_u16_256("vmin_hi (512→256)", vmin_hi);
    }

    let mask = _mm256_cmp_epu16_mask(vmin_hi, vmin_lo, _MM_CMPINT_LT);
    
    #[cfg(feature = "trace")]
    println!("Mask 512→256: {:016b}", mask);

    let mut vmin256 = _mm256_mask_mov_epi16(vmin_lo, mask, vmin_hi);
    let mut vidx256 = _mm256_mask_mov_epi16(vidx_lo, mask, vidx_hi);


    #[cfg(feature = "trace")]
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


    #[cfg(feature = "trace")]
    {dump_u16_128("vmin_lo (256→128)", vmin_lo);
    dump_u16_128("vmin_hi (256→128)", vmin_hi);
    }
    let mask = _mm_cmp_epu16_mask(vmin_hi, vmin_lo, _MM_CMPINT_LT);
    println!("Mask 256→128: {:08b}", mask);

    let mut vmin128 = _mm_mask_mov_epi16(vmin_lo, mask, vmin_hi);
    let mut vidx128 = _mm_mask_mov_epi16(vidx_lo, mask, vidx_hi);

    #[cfg(feature = "trace")]
    {dump_u16_128("vmin after 256→128", vmin128);
    dump_u16_128("vidx after 256→128", vidx128);
    }

    // =========================
    // 128-bit lane reduction
    // =========================

    #[cfg(feature = "trace")]
    {println!("--- 8 → 4 reduction ---");
    dump_u16_128("before", vmin128);
    }
    let tval = _mm_srli_si128(vmin128, 8); // shift by 4 u16
    let tidx = _mm_srli_si128(vidx128, 8);


    #[cfg(feature = "trace")]
    dump_u16_128("shifted vals", tval);

    let mask = _mm_cmplt_epu16_mask(tval, vmin128);
    
    #[cfg(feature = "trace")]
    println!("mask: {:08b}", mask);

    vmin128 = _mm_mask_mov_epi16(vmin128, mask, tval);
    vidx128 = _mm_mask_mov_epi16(vidx128, mask, tidx);


    #[cfg(feature = "trace")]
    {dump_u16_128("after 8→4 vmin", vmin128);
    dump_u16_128("after 8→4 vidx", vidx128);
    }

    #[cfg(feature = "trace")]
    {println!("--- 4 → 2 reduction ---");
    dump_u16_128("before", vmin128);
    }
    let tval = _mm_srli_si128(vmin128, 4); // shift by 2 u16
    let tidx = _mm_srli_si128(vidx128, 4);


    #[cfg(feature = "trace")]
    dump_u16_128("shifted vals", tval);

    let mask = _mm_cmplt_epu16_mask(tval, vmin128);
    
    #[cfg(feature = "trace")]
    println!("mask: {:08b}", mask);

    vmin128 = _mm_mask_mov_epi16(vmin128, mask, tval);
    vidx128 = _mm_mask_mov_epi16(vidx128, mask, tidx);


    #[cfg(feature = "trace")]
    {dump_u16_128("after 4→2 vmin", vmin128);
    dump_u16_128("after 4→2 vidx", vidx128);
    }


    #[cfg(feature = "trace")]
    {println!("--- 2 → 1 reduction ---");
    dump_u16_128("before", vmin128);
    }
    let tval = _mm_srli_si128(vmin128, 2); // shift by 1 u16
    let tidx = _mm_srli_si128(vidx128, 2);


    #[cfg(feature = "trace")]
    dump_u16_128("shifted vals", tval);

    let mask = _mm_cmplt_epu16_mask(tval, vmin128);
    
    #[cfg(feature = "trace")]
    println!("mask: {:08b}", mask);

    vmin128 = _mm_mask_mov_epi16(vmin128, mask, tval);
    vidx128 = _mm_mask_mov_epi16(vidx128, mask, tidx);

    
    #[cfg(feature = "trace")]
    {dump_u16_128("after 2→1 vmin", vmin128);
    dump_u16_128("after 2→1 vidx", vidx128);
    }

    let min_val = _mm_extract_epi16(vmin128, 0) as u16;
    let min_idx = _mm_extract_epi16(vidx128, 0) as usize;

    
    #[cfg(feature = "trace")]
    {println!("FINAL min_val: {}", min_val);
    println!("FINAL min_idx: {}", min_idx);
    println!("===== END REDUCTION =====");
    }
    (min_idx, min_val)
}

#[target_feature(enable = "avx512f,avx512bw")]
pub fn compute_start_mask(i: usize) 
    -> (usize, __mmask32)
{
    let start_block = i / 32;
    let start_offset = i % 32;
    let start_mask = (!0u32 << start_offset) as __mmask32;
    (start_block, start_mask)
}


#[target_feature(enable = "avx512f,avx512bw")]
pub fn compute_end_mask(j: usize) -> (usize, __mmask32) {
    let end_block   = j / 32;
    let end_offset   = j % 32;
    let end_mask = ((1u64 << (end_offset + 1)) - 1) as __mmask32;

    (end_block, end_mask)
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn minindex_u16_flexible(
    array: *const u16,
    start_block: usize,      // block index for start_mask
    start_mask: __mmask32,     // mask for first block (or !0)
    end_block: usize,         // block index for end_mask
    end_mask: __mmask32       // mask for last block (or !0)
) -> (usize, u16) {
    #[cfg(feature = "trace")]
    println!("masks are {:032b} {:032b}", start_mask, end_mask);
    let mut vmin = _mm512_set1_epi16(u16::MAX as i16);
    let mut vidx = _mm512_setzero_si512();

    let offsets: [i16; 32] = core::array::from_fn(|i| i as i16);
    let voff = unsafe { _mm512_loadu_si512(offsets.as_ptr() as *const _) };

    #[cfg(feature = "trace")]
    {
        println!("--- Number of Blocks {} ---", end_block - start_block + 1);
        dump_u16_512("Current vmin before", vmin);
        dump_u16_512("Current vidx before", vidx);
    }

    for block in start_block..end_block + 1 {
        let i = block * 32;
        let ptr = unsafe { array.add(i) };

        let mask = if block == start_block && block == end_block {
            #[cfg(feature = "trace")]
            println!("Using combined mask {:032b}", start_mask & end_mask);
            start_mask & end_mask
        } else if block == start_block {
            #[cfg(feature = "trace")]
            println!("Using start mask {:032b}", start_mask);
            start_mask
        } else if block == end_block {
            #[cfg(feature = "trace")]
            println!("Using end mask {:032b}", end_mask);
            end_mask
        } else {
            u32::MAX as __mmask32
        };

        let v = unsafe { _mm512_maskz_loadu_epi16(mask, ptr as *const _) };

        let base = _mm512_set1_epi16(i as i16);
        let idx = _mm512_add_epi16(base, voff);

        let cmp_mask = _mm512_mask_cmp_epu16_mask(
            mask,
            v,
            vmin,
            _MM_CMPINT_LT,
        );

        vmin = _mm512_mask_mov_epi16(vmin, cmp_mask, v);
        vidx = _mm512_mask_mov_epi16(vidx, cmp_mask, idx);
        #[cfg(feature = "trace")]
        {
            print!("After processing block {}: ", block);
            dump_u16_512("Updated vmin", vmin);
            dump_u16_512("Updated vidx", vidx);
        }
    }

    unsafe { horizontal_reduce_min_u16(vmin, vidx) }
}


#[target_feature(enable = "avx512f,avx512bw")]


#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn minindex_u16(array: *const u16, size: usize) -> (i32, u16) {
    if size == 0 {
        return (-1, u16::MAX);
    }

    // running minima
    let mut vmin = _mm512_set1_epi16(unsafe { *array } as i16);

    // running indices (i16!)
    let mut vidx = _mm512_set1_epi16(0);

    // offsets 0..31 (i16)
    let offsets: [i16; 32] = core::array::from_fn(|i| i as i16);
    let voff = unsafe { _mm512_loadu_si512(offsets.as_ptr() as *const _) };

    let mut i = 0;

    while i + 32 <= size {
        let ptr = unsafe { array.add(i) };

        let v = unsafe { _mm512_loadu_si512(ptr as *const _) };

        let base = _mm512_set1_epi16(i as i16);
        let idx = _mm512_add_epi16(base, voff);

        let mask = _mm512_cmp_epu16_mask(v, vmin, _MM_CMPINT_LT);  // Note: epu16 for unsigned

        vmin = _mm512_mask_mov_epi16(vmin, mask, v);
        vidx = _mm512_mask_mov_epi16(vidx, mask, idx);

        i += 32;
    }

    // horizontal reduction (scalar)
    let mut vals = [0u16; 32];
    let mut inds = [0i16; 32];

    unsafe { _mm512_storeu_si512(vals.as_mut_ptr() as *mut _, vmin) };
    unsafe { _mm512_storeu_si512(inds.as_mut_ptr() as *mut _, vidx) };

    let mut min_val = vals[0];
    let mut min_index = inds[0] as i32;

    for j in 1..32 {
        if vals[j] < min_val {
            min_val = vals[j];
            min_index = inds[j] as i32;
        }
    }

    // scalar tail
    while i < size {
        let val = unsafe { *array.add(i) };
        if val < min_val {
            min_val = val;
            min_index = i as i32;
        }
        i += 1;
    }

    (min_index, min_val)
}

pub fn find_min(arr: &[u16], start_index: usize, end_index: usize) -> Option<(u16, usize)> {
    if arr.is_empty() {
        return None;
    }

    if std::is_x86_feature_detected!("avx512f")
                        && std::is_x86_feature_detected!("avx512bw") {

        let (start_block, start_mask) = unsafe { compute_start_mask(start_index) };
        let (end_block, end_mask) = unsafe { compute_end_mask(end_index) };
        let (index, val) = unsafe { minindex_u16_flexible(arr.as_ptr(),
                                        start_block, start_mask,
                                        end_block, end_mask)};
        Some((val,index))
    } else {
        scalar_min(arr, start_index, end_index)
    }
}

pub fn scalar_min(arr: &[u16], start_index: usize, end_index: usize) -> Option<(u16, usize)> {
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