use array_min::avx512_min::*;

fn find_min(arr: &[u16], start_index: usize, end_index: usize) -> Option<(u16, usize)> {
    if arr.is_empty() {
        return None;
    }
    let (start_block, start_mask) = unsafe { compute_start_mask(start_index) };
    let (end_block, end_mask) = unsafe { compute_end_mask(end_index) };
    let (index, val) = unsafe { minindex_u16_flexible(arr.as_ptr(),
                                    start_block, start_mask,
                                    end_block, end_mask)};
    Some((val,index))
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
fn main() {
    let a:[u16; 64] = [05, 13, 09, 44, 22, 08, 33, 07, 34, 15, 
                       13, 12, 11, 10, 20, 19, 18, 17, 16, 31,
                       20, 29, 28, 27, 26, 25, 24, 23, 22, 21, 
                       30, 39, 38, 37, 36, 04, 45, 44, 43, 42,
                       41, 50, 14, 49, 48, 47, 46, 55, 54, 53, 
                       52, 51, 60, 59, 58, 57, 56, 65, 64, 63, 
                       62, 61, 60, 6];
    for start_index in 0..a.len() {
        for end_index in start_index..a.len() {
                let (scalar_val, scalar_idx) = scalar_min(&a, start_index, end_index).unwrap();
                let (avx_val, avx_idx) = find_min(&a, start_index, end_index).unwrap();
                assert_eq!(scalar_val, avx_val);
                assert_eq!(scalar_idx, avx_idx);
                println!("Min of a[{}..={}] = {} at index {}", start_index, end_index, avx_val, avx_idx);
            }
    }
}