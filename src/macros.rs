#[macro_export]
macro_rules! benchmark_same_shape_binary {
    ($size:expr, $rust_name:ident, $rust_code:tt, $py_code:expr) => {
        paste::paste! {
            pub fn [<benchmark_ $rust_name>](device: &candle_core::Device, n: usize) -> $crate::AllBenchResults {
                let a = Tensor::zeros($size, candle_core::DType::F32, &device).unwrap();
                let b = Tensor::zeros($size, candle_core::DType::F32, &device).unwrap();

                let candle_res = $crate::benchmark_thing(
                    || {
                        let _ = std::hint::black_box($rust_code(&a, &b));
                    },
                    n,
                );

                let py_res = pyo3::Python::with_gil(|py| {
                    use pyo3::types::PyDictMethods;
                    use pyo3::ToPyObject;
                    use $crate::CandleToTorchDevice;

                    let locals = pyo3::types::PyDict::new_bound(py);
                    locals
                        .set_item("size".to_object(py), $size.to_object(py))
                        .expect("Failed to set size in locals");
                    locals
                        .set_item("device".to_object(py), device.to_torch_device().to_object(py))
                        .expect("Failed to set size in locals");

                    let _ = py
                        .run_bound(
                            include_str!("../python_src/init_same_shape.py"),
                            None,
                            Some(&locals),
                        )
                        .expect("Expected setup to work!");

                    $crate::benchmark_thing(
                        || {
                            let _ = py
                                .run_bound($py_code, None, Some(&locals))
                                .expect("Bench failed.");
                        },
                        n,
                    )
                });

                $crate::AllBenchResults {
                    candle: candle_res,
                    torch: py_res,
                }
            }
        }
    };
}

#[macro_export]
macro_rules! benchmark_same_shape_unary {
    ($size:expr, $rust_name:ident, $rust_code:tt, $py_code:expr) => {
        paste::paste! {
            pub fn [<benchmark_ $rust_name>](device: &candle_core::Device, n: usize) -> $crate::AllBenchResults {
                let a = Tensor::zeros($size, candle_core::DType::F32, &device).unwrap();

                let candle_res = $crate::benchmark_thing(
                    || {
                        let _ = std::hint::black_box($rust_code(&a));
                    },
                    n,
                );

                let py_res = pyo3::Python::with_gil(|py| {
                    use pyo3::types::PyDictMethods;
                    use pyo3::ToPyObject;
                    use $crate::CandleToTorchDevice;

                    let locals = pyo3::types::PyDict::new_bound(py);
                    locals
                        .set_item("size".to_object(py), $size.to_object(py))
                        .expect("Failed to set size in locals");
                    locals
                        .set_item("device".to_object(py), device.to_torch_device().to_object(py))
                        .expect("Failed to set size in locals");

                    let _ = py
                        .run_bound(
                            include_str!("../python_src/init_one_shape.py"),
                            None,
                            Some(&locals),
                        )
                        .expect("Expected setup to work!");

                    $crate::benchmark_thing(
                        || {
                            let _ = py
                                .run_bound($py_code, None, Some(&locals))
                                .expect("Bench failed.");
                        },
                        n,
                    )
                });

                $crate::AllBenchResults {
                    candle: candle_res,
                    torch: py_res,
                }
            }
        }
    };
}

#[macro_export]
macro_rules! benchmark_same_shape_layout {
    ($size:expr, $rust_name:ident, $rust_code:tt, $py_code:expr) => {
        paste::paste! {
            pub fn [<benchmark_ $rust_name>](device: &candle_core::Device, n: usize) -> $crate::AllBenchResults {
                let a = Tensor::zeros($size, candle_core::DType::F32, &device).unwrap();

                let candle_res = $crate::benchmark_thing(
                    || {
                        let _ = std::hint::black_box($rust_code(&a));
                    },
                    n,
                );

                let py_res = pyo3::Python::with_gil(|py| {
                    use pyo3::types::PyDictMethods;
                    use pyo3::ToPyObject;
                    use $crate::CandleToTorchDevice;

                    let locals = pyo3::types::PyDict::new_bound(py);
                    locals
                        .set_item("size".to_object(py), $size.to_object(py))
                        .expect("Failed to set size in locals");
                    locals
                        .set_item("device".to_object(py), device.to_torch_device().to_object(py))
                        .expect("Failed to set size in locals");

                    let _ = py
                        .run_bound(
                            include_str!("../python_src/init_one_shape.py"),
                            None,
                            Some(&locals),
                        )
                        .expect("Expected setup to work!");

                    $crate::benchmark_thing(
                        || {
                            let _ = py
                                .run_bound($py_code, None, Some(&locals))
                                .expect("Bench failed.");
                        },
                        n,
                    )
                });

                $crate::AllBenchResults {
                    candle: candle_res,
                    torch: py_res,
                }
            }
        }
    };
}

#[macro_export]
macro_rules! run_a_bench {
    ($name:ident, $n:expr, $device:ident, $results:ident) => {
        paste! {
            let AllBenchResults {
                candle: [<$name _candle>],
                torch: [<$name _torch>],
            } = [<benchmark_ $name>](&$device, $n);

            let candle = [<$name _candle>].duration_secs / $n as f64;
            let torch = [<$name _torch>].duration_secs / $n as f64;
            let result = if candle <= torch {
                format!("✅ Candle faster than Torch by {:.3}%", (1. - candle / torch ) * 100.)
            } else {
                format!("❌ Candle slower than Torch by {:.3}%", (1. - torch / candle ) * 100.)
            };

            $results.push(TableRow {
                test: stringify!($name),
                device: $device.to_torch_device(),
                candle_time_per_pass: format!("{:.3}µs", candle * 1000000.),
                torch_time_per_pass: format!("{:.3}µs", torch * 1000000.),
                n:  [<$name _torch>].n,
                result
            });
        }
    };
}
