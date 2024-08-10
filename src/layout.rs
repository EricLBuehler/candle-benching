use candle_core::Tensor;

use crate::benchmark_same_shape_layout;

#[inline(never)]
fn do_reshape(a: &Tensor) {
    let _ = a.reshape((4096, 2048)).unwrap();
}

#[inline(never)]
fn do_transpose(a: &Tensor) {
    let _ = a.t().unwrap();
}

#[inline(never)]
fn do_narrow(a: &Tensor) {
    let _ = a.narrow(1, 0, 512).unwrap();
}

benchmark_same_shape_layout!(
    (256, 4, 1024, 8),
    reshape,
    do_reshape,
    "torch.reshape(a, (4096, 2048))"
);
benchmark_same_shape_layout!(
    (256, 4, 1024, 8),
    transpose,
    do_transpose,
    "a.transpose(1,2)"
);
benchmark_same_shape_layout!(
    (256, 1024, 4),
    narrow,
    do_narrow,
    "a.narrow(dim=1, start=0, length=512)"
);
