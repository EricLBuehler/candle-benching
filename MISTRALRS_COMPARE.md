# Comparison with mistral.rs

**Hardware**

```bash
$ nvidia-smi
Mon Aug 19 17:03:11 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A10                     On  | 00000000:07:00.0 Off |                    0 |
|  0%   32C    P8              15W / 150W |      4MiB / 23028MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

**Prompt used**

`<s> [INST] What is graphene? [/INST]`

**Model**
- TheBloke/Mistral-7B-Instruct-v0.1-GGUF
- mistral-7b-instruct-v0.1.Q4_K_S.gguf

**Candle commit**
```
$ git rev-parse HEAD
b47c0bc475cc955805d9379b4444aebf3e9a516d
```

## ToC

- [Results](#results)
    - In the typical mistral.rs use case as in examples
    - PagedAttention enabled
- [Results without PagedAttention](#results-wo-mistralrs-pagedattention)
    - PagedAttention disabled


## Mistral.rs optimizations
- Use of cuBLASlt gives +80% PP speeds
- Fused RoPE, Fused RmsNorm
- (when not using PagedAttention in mistral.rs) Fusing softmax scale and attention mask application with cuBLASlt

### Results
Tests conducted with no warmup runs, in order of run id.

**For mistral.rs:**

```
cargo run --release --features cuda -- --port 1234 --log output.txt gguf -t mistralai/Mistral-7B-Instruct-v0.1 -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF -f mistral-7b-instruct-v0.1.Q4_K_S.gguf
```

**For candle:**

```
cargo run --release --features cuda --example quantized -- --which 7b-mistral-instruct --prompt "<s> [INST] What is graphene? [/INST]"
```

|Run id|Mistral.rs PP T/s|Candle PP T/s|
| --- | --- | --- |
| 1 |91.46|100.71|
| 2 |182.93|101.21|
| 3 |185.16|99.66|
| 4 |182.93|98.45|
| 5 |182.93|99.16|


|Run id|Mistral.rs TG T/s|Mistral.rs TG T/s|
| --- | --- | --- |
| 1 |86.19|75.56|
| 2 |89.75|75.45|
| 3 |88.94|75.57|
| 4 |88.24|75.72|
| 5 |89.14|75.74|

### Results w/o mistral.rs PagedAttention
Tests conducted with no warmup runs, in order of run id.

**For mistral.rs:**

```
cargo run --release --features cuda -- --port 1234 --log output.txt --no-paged-attn gguf -t mistralai/Mistral-7B-Instruct-v0.1 -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF -f mistral-7b-instruct-v0.1.Q4_K_S.gguf
```

**For candle:**

```
cargo run --release --features cuda --example quantized -- --which 7b-mistral-instruct --prompt "<s> [INST] What is graphene? [/INST]"
```

|Run id|Mistral.rs PP T/s|Candle PP T/s|
| --- | --- | --- |
| 1 |93.75|101.88|
| 2 |180.72|101.33|
| 3 |176.47|101.61|
| 4 |180.72|100.77|
| 5 |178.57|98.85|


|Run id|Mistral.rs TG T/s|Mistral.rs TG T/s|
| --- | --- | --- |
| 1 |76.23|76.12|
| 2 |77.87|75.89|
| 3 |77.63|76.00|
| 4 |77.72|75.93|
| 5 |78.41|75.99|