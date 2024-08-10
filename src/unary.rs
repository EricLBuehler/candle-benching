use candle_core::Tensor;
use candle_nn::Module;

use crate::benchmark_same_shape_unary;

#[inline(never)]
fn do_relu(a: &Tensor) {
    let _ = candle_nn::Activation::Relu.forward(a).unwrap();
}

#[inline(never)]
fn do_gelu(a: &Tensor) {
    let _ = candle_nn::Activation::Gelu.forward(a).unwrap();
}

#[inline(never)]
fn do_silu(a: &Tensor) {
    let _ = candle_nn::Activation::Silu.forward(a).unwrap();
}

#[inline(never)]
fn do_softmax(a: &Tensor) {
    let _ = candle_nn::ops::softmax_last_dim(a).unwrap();
}

benchmark_same_shape_unary!((1024, 1024), relu, do_relu, "torch.relu(a)");
benchmark_same_shape_unary!((1024, 1024), gelu, do_gelu, "torch.nn.GELU(a)");
benchmark_same_shape_unary!((1024, 1024), silu, do_silu, "torch.nn.SiLU(a)");
benchmark_same_shape_unary!(
    (1024, 1024),
    softmax,
    do_softmax,
    "torch.softmax(a, dim=-1)"
);
