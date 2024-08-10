use candle_core::Device;
#[cfg(feature = "cuda")]
use cublaslt::setup_cublas_lt_wrapper;
use paste::paste;
use std::time::Instant;
use tabled::{settings::Style, Table, Tabled};

mod binary;
#[cfg(feature = "cuda")]
mod cublaslt;
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

pub fn baseline_test(n: usize) -> AllBenchResults {
    let py_res = pyo3::Python::with_gil(|py| {
        let locals = pyo3::types::PyDict::new_bound(py);
        let _ = py
            .run_bound("a=1\nb=2", None, Some(&locals))
            .expect("Expected setup to work!");

        crate::benchmark_thing(
            || {
                // Test a+b because this is similar to the code we're running
                let _ = py
                    .run_bound("a+b", None, Some(&locals))
                    .expect("Bench failed.");
            },
            n,
        )
    });

    crate::AllBenchResults {
        candle: BenchResults {
            duration_secs: 0.0,
            n: 0,
        },
        torch: py_res,
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
    #[cfg(feature = "cuda")]
    setup_cublas_lt_wrapper();

    let mut results = Vec::new();

    let n = if device.is_cpu() { 1000 } else { 100_000 };

    let baseline_test = &baseline_test(n).torch.duration_secs;
    println!("===== BENCHMARKING WITH DEVICE {device:?} =====\n");
    println!("Benching same-shape binary: add");
    run_a_bench!(add, n, device, results, baseline_test);
    println!("Benching same-shape binary: matmul");
    run_a_bench!(matmul, n, device, results, baseline_test);

    #[cfg(feature = "cuda")]
    {
        println!("Benching same-shape binary: cuBLASlt matmul");
        run_a_bench!(cublaslt_matmul, n, device, results, baseline_test);
    }

    println!("Benching unary: relu");
    run_a_bench!(relu, n, device, results, baseline_test);
    println!("Benching unary: gelu");
    run_a_bench!(gelu, n, device, results, baseline_test);
    println!("Benching unary: silu");
    run_a_bench!(silu, n, device, results, baseline_test);
    println!("Benching unary: softmax (last dim)");
    run_a_bench!(softmax, n, device, results, baseline_test);

    println!("Benching layout: reshape (256, 4, 1024, 8) -> (4096, 2048)");
    run_a_bench!(reshape, n, device, results, baseline_test);
    println!("Benching layout: transpose (swap last 2)");
    run_a_bench!(transpose, n, device, results, baseline_test);
    println!("Benching layout: narrow (256, 1024, 4) -> (2, 512, 4)");
    run_a_bench!(narrow, n, device, results, baseline_test);

    let table = Table::new(results).with(Style::markdown()).to_string();
    println!("\n\nBenchmark results\n{table}");
}
