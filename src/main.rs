use candle_core::Device;
use paste::paste;
use std::time::Instant;
use tabled::{Table, Tabled};

mod binary;
mod layout;
mod macros;
mod unary;

use binary::*;
use layout::*;
use unary::*;

#[must_use]
#[derive(Debug, Clone, Tabled)]
pub struct BenchResults {
    duration_secs: f64,
    n: usize,
}

#[must_use]
#[derive(Debug, Clone)]
pub struct AllBenchResults {
    candle: BenchResults,
    torch: BenchResults,
}

#[derive(Debug, Clone, Tabled)]
pub struct TableRow {
    test: &'static str,
    device: String,
    candle_time_per_pass: String,
    torch_time_per_pass: String,
    n: usize,
    result: String,
}

pub(crate) trait CandleToTorchDevice {
    fn to_torch_device(&self) -> String;
}

impl CandleToTorchDevice for Device {
    fn to_torch_device(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(_) => "cuda".to_string(),
            Device::Metal(_) => "mps".to_string(),
        }
    }
}

fn benchmark_thing(f: impl Fn(), n: usize) -> BenchResults {
    let now = Instant::now();
    for _ in 0..n {
        f();
    }
    let duration = Instant::now().duration_since(now);
    BenchResults {
        duration_secs: duration.as_secs_f64(),
        n,
    }
}

fn main() {
    pyo3::prepare_freethreaded_python();
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap()
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap()
    } else {
        Device::Cpu
    };

    let mut results = Vec::new();

    let n = if device.is_cpu() { 1000 } else { 100_000 };
    println!("===== BENCHMARKING WITH DEVICE {device:?} =====\n");
    println!("Benching same-shape binary: add");
    run_a_bench!(add, n, device, results);
    println!("Benching same-shape binary: matmul");
    run_a_bench!(matmul, n, device, results);

    println!("Benching unary: relu");
    run_a_bench!(relu, n, device, results);
    println!("Benching unary: gelu");
    run_a_bench!(gelu, n, device, results);
    println!("Benching unary: silu");
    run_a_bench!(silu, n, device, results);
    println!("Benching unary: softmax (last dim)");
    run_a_bench!(softmax, n, device, results);

    println!("Benching layout: reshape (256, 4, 1024, 8) -> (4096, 2048)");
    run_a_bench!(reshape, n, device, results);
    println!("Benching layout: transpose (swap last 2)");
    run_a_bench!(transpose, n, device, results);
    println!("Benching layout: narrow (256, 1024, 4) -> (2, 512, 4)");
    run_a_bench!(narrow, n, device, results);

    let table = Table::new(results).to_string();
    println!("\n\nBenchmark results\n{table}");
}
