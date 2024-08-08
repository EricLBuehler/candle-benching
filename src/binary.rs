use candle_core::Tensor;

use crate::benchmark_same_shape_binary;

#[inline(never)]
fn do_matmul(a: &Tensor, b: &Tensor) {
    let _ = a.matmul(b);
}

#[inline(never)]
fn do_add(a: &Tensor, b: &Tensor) {
    let _ = a + b;
}

benchmark_same_shape_binary!((1024, 1024), add, do_add, "a + b");
benchmark_same_shape_binary!((1024, 1024), matmul, do_matmul, "a.matmul(b)");
