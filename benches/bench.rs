// bench_min.rs  —  SIMD vs Scalar array minimum finding benchmark
// Compile with: RUSTFLAGS="-C target-cpu=native" cargo run --release --bin bench_min

use array_min::avx512_min::*;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::hint::black_box;
use std::fs;
use std::io::{BufWriter, Write};

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const ARRAY_SIZES: &[usize] = &[32, 64, 128, 256, 512, 1024];
const WARMUP_ITERS: usize = 2;
const MEASURE_ITERS: usize = 8;
const SEED: u64 = 41;

// ═══════════════════════════════════════════════════════════════════════════
// RDTSC UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rdtsc_start() -> u64 {
    use core::arch::x86_64::{__cpuid, _rdtsc};

    unsafe { __cpuid(0) }; // serialize
    unsafe { _rdtsc() }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rdtsc_end() -> u64 {
    use core::arch::x86_64::{__cpuid, __rdtscp};

    let mut aux = 0;
    let t = unsafe { __rdtscp(&mut aux) }; // serialize + read
    unsafe { __cpuid(0) }; // serialize
    t
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn rdtsc_start() -> u64 { 0 }
#[cfg(not(target_arch = "x86_64"))]
unsafe fn rdtsc_end() -> u64 { 0 }

// ═══════════════════════════════════════════════════════════════════════════
// MEASUREMENT HELPERS
// ═══════════════════════════════════════════════════════════════════════════

fn median_u64(samples: &mut Vec<u64>) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK RESULT
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct BenchResult {
    start: usize,
    end: usize,
    range_len: usize,
    scalar_cycles: u64,
    vector_cycles: u64,
    scalar_val: u16,
    vector_val: u16,
    correct: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// MEASUREMENT FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Measure a single (start, end) range pair
fn measure_range<const N: usize>(
    array: &[u16; N],
    start: usize,
    end: usize,
) -> BenchResult {
    // Warm up
    for _ in 0..WARMUP_ITERS {
        let _ = scalar_min(array, start, end);
        let _ = unsafe { minindex_u16(array, start as u16, end as u16) };
    }

    // Measure scalar
    let mut scalar_cycles_samples = Vec::with_capacity(MEASURE_ITERS);
    let mut scalar_val = 0u16;

    for _ in 0..MEASURE_ITERS {
        let cyc0 = unsafe { rdtsc_start() };
        let (v, _) = scalar_min(black_box(array), black_box(start), black_box(end));
        let cyc1 = unsafe { rdtsc_end() };

        scalar_val = black_box(v);
        scalar_cycles_samples.push(cyc1.saturating_sub(cyc0));
    }

    let scalar_cycles = median_u64(&mut scalar_cycles_samples);

    // Measure vector
    let mut vector_cycles_samples = Vec::with_capacity(MEASURE_ITERS);
    let mut vector_val = 0u16;

    for _ in 0..MEASURE_ITERS {
        let cyc0 = unsafe { rdtsc_start() };
        let (v, _) = unsafe { minindex_u16(black_box(array), black_box(start as u16), black_box(end as u16)) };
        let cyc1 = unsafe { rdtsc_end() };

        vector_val = black_box(v);
        vector_cycles_samples.push(cyc1.saturating_sub(cyc0));
    }

    let vector_cycles = median_u64(&mut vector_cycles_samples);

    BenchResult {
        start,
        end,
        range_len: end - start + 1,
        scalar_cycles,
        vector_cycles,
        scalar_val,
        vector_val,
        correct: scalar_val == vector_val,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK RUNNER
// ═══════════════════════════════════════════════════════════════════════════

fn benchmark_size<const N: usize>(size: usize) -> std::io::Result<()> {
    println!("=== array size {} ===", size);

    // Create shuffled array
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut array: [u16; N] = std::array::from_fn(|i| i as u16);
    array.shuffle(&mut rng);

    // Setup output file
    fs::create_dir_all("bench_results")?;
    let path = format!("bench_results/bench_{}.csv", size);
    let file = fs::File::create(&path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(
        writer,
        "array_size,start,end,range_len,scalar_cycles,vector_cycles,scalar_val,vector_val,correct"
    )?;

    let sample_rate = if size > 128 { 4 } else if size > 64 { 2 } else { 1 };
    let mut count = 0usize;
    let mut mismatches = 0usize;

    // Sample pairs
    for start in (0..size).step_by(sample_rate) {
        for end in (start..size).step_by(sample_rate) {
            eprintln!("  Testing ({}, {})...", start, end);

            let result = measure_range(&array, start, end);

            if !result.correct {
                mismatches += 1;
            }

            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{}",
                size,
                result.start,
                result.end,
                result.range_len,
                result.scalar_cycles,
                result.vector_cycles,
                result.scalar_val,
                result.vector_val,
                result.correct as u8,
            )?;

            count += 1;
        }
    }

    writer.flush()?;
    println!("  → {} written (mismatches: {})\n", path, mismatches);

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    println!("Benchmark: scalar_min vs minindex_u16");
    println!("Warmup iters : {}", WARMUP_ITERS);
    println!("Measure iters: {}", MEASURE_ITERS);
    println!("Array sizes  : {:?}\n", ARRAY_SIZES);

    for &size in ARRAY_SIZES {
        let result = match size {
            32 => benchmark_size::<32>(size),
            64 => benchmark_size::<64>(size),
            128 => benchmark_size::<128>(size),
            256 => benchmark_size::<256>(size),
            512 => benchmark_size::<512>(size),
            _ => {
                eprintln!("Unsupported size: {}", size);
                continue;
            }
        };

        if let Err(e) = result {
            eprintln!("Error benchmarking size {}: {}", size, e);
        }
    }

    println!("All done. CSVs in ./bench_results/");
}