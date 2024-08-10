use candle_core::Tensor;

use crate::benchmark_same_shape_binary;

#[inline(never)]
fn do_matmul(a: &Tensor, b: &Tensor) {
    let _ = a.matmul(b).unwrap();
}

#[cfg(feature = "cuda")]
#[inline(never)]
fn do_cublaslt_matmul(a: &Tensor, b: &Tensor) {
    use crate::cublaslt::CUBLASLT_HANDLE;

    let cublaslt = CUBLASLT_HANDLE.lock().unwrap().unwrap();
    let _ = cublaslt.matmul(a, b, None, None, None, None, None).unwrap();
}

#[inline(never)]
fn do_add(a: &Tensor, b: &Tensor) {
    let _ = (a + b).unwrap();
}

benchmark_same_shape_binary!((1024, 1024), add, do_add, "a + b");
benchmark_same_shape_binary!((1024, 1024), matmul, do_matmul, "a.matmul(b)");
#[cfg(feature = "cuda")]
benchmark_same_shape_binary!(
    (1024, 1024),
    cublaslt_matmul,
    do_cublaslt_matmul,
    "a.matmul(b)"
);
