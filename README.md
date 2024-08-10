# candle-benching

Benchmarking for Candle.

Just clone and then run one of:
- `cargo run --release`
- `cargo run --release --features cuda`
- `cargo run --release --features metal`

## **CUDA**: RTX 4070
| test            | device | candle_time_per_pass | torch_time_per_pass | n      | result                                  |
|-----------------|--------|----------------------|---------------------|--------|-----------------------------------------|
| add             | cuda   | 19.496µs             | 17.935µs            | 100000 | ❌ Candle slower than Torch by 1.087x   |
| matmul          | cuda   | 259.859µs            | 234.579µs           | 100000 | ❌ Candle slower than Torch by 1.108x   |
| cublaslt_matmul | cuda   | 136.350µs            | 254.747µs           | 100000 | ✅ Candle faster than Torch by 1.868x   |
| relu            | cuda   | 17.804µs             | 22.853µs            | 100000 | ✅ Candle faster than Torch by 1.284x   |
| gelu            | cuda   | 21.429µs             | 10.031µs            | 100000 | ❌ Candle slower than Torch by 2.136x   |
| silu            | cuda   | 20.767µs             | 10.429µs            | 100000 | ❌ Candle slower than Torch by 1.991x   |
| softmax         | cuda   | 22.955µs             | 27.366µs            | 100000 | ✅ Candle faster than Torch by 1.192x   |
| reshape         | cuda   | 0.123µs              | 9.104µs             | 100000 | ✅ Candle faster than Torch by 73.951x  |
| transpose       | cuda   | 0.094µs              | 5.810µs             | 100000 | ✅ Candle faster than Torch by 61.876x  |
| narrow          | cuda   | 0.113µs              | 11.722µs            | 100000 | ✅ Candle faster than Torch by 104.156x |

## **CPU**: Intel Core Ultra 9 185H
| test      | device | candle_time_per_pass | torch_time_per_pass | n    | result                                  |
|-----------|--------|----------------------|---------------------|------|-----------------------------------------|
| add       | cpu    | 133.909µs            | 51.276µs            | 1000 | ❌ Candle slower than Torch by 2.612x   |
| matmul    | cpu    | 4464.338µs           | 4658.558µs          | 1000 | ✅ Candle faster than Torch by 1.044x   |
| relu      | cpu    | 175.867µs            | 50.372µs            | 1000 | ❌ Candle slower than Torch by 3.491x   |
| gelu      | cpu    | 2374.970µs           | 10.533µs            | 1000 | ❌ Candle slower than Torch by 225.487x |
| silu      | cpu    | 2115.940µs           | 9.030µs             | 1000 | ❌ Candle slower than Torch by 234.315x |
| softmax   | cpu    | 1655.149µs           | 266.818µs           | 1000 | ❌ Candle slower than Torch by 6.203x   |
| reshape   | cpu    | 0.074µs              | 13.229µs            | 1000 | ✅ Candle faster than Torch by 179.150x |
| transpose | cpu    | 0.120µs              | 8.362µs             | 1000 | ✅ Candle faster than Torch by 69.404x  |
| narrow    | cpu    | 0.064µs              | 12.545µs            | 1000 | ✅ Candle faster than Torch by 197.377x |

## **CPU and MKL**: Intel Core Ultra 9 185H
| test      | device | candle_time_per_pass | torch_time_per_pass | n    | result                                  |
|-----------|--------|----------------------|---------------------|------|-----------------------------------------|
| add       | cpu    | 34.672µs             | 49.539µs            | 1000 | ✅ Candle faster than Torch by 1.429x   |
| matmul    | cpu    | 4718.913µs           | 4705.201µs          | 1000 | ❌ Candle slower than Torch by 1.003x   |
| relu      | cpu    | 257.947µs            | 65.408µs            | 1000 | ❌ Candle slower than Torch by 3.944x   |
| gelu      | cpu    | 2405.888µs           | 12.113µs            | 1000 | ❌ Candle slower than Torch by 198.618x |
| silu      | cpu    | 523.669µs            | 9.747µs             | 1000 | ❌ Candle slower than Torch by 53.725x  |
| softmax   | cpu    | 1667.239µs           | 272.704µs           | 1000 | ❌ Candle slower than Torch by 6.114x   |
| reshape   | cpu    | 0.132µs              | 18.616µs            | 1000 | ✅ Candle faster than Torch by 141.064x |
| transpose | cpu    | 0.171µs              | 7.215µs             | 1000 | ✅ Candle faster than Torch by 42.137x  |
| narrow    | cpu    | 0.107µs              | 13.318µs            | 1000 | ✅ Candle faster than Torch by 124.090x |
