# ADTC 2026 — Gemma 3 270M Financial Intelligence

> **On-device M-Pesa financial intelligence for resource-constrained environments.**

This notebook builds a compact financial intelligence pipeline around **Gemma 3 270M IT**, fine-tuned with **LoRA** to transform Kenyan M-Pesa SMS messages into strict, structured transaction JSON.

The resulting structured records are converted into a local transaction ledger and analyzed with deterministic Python logic for cash flow, spending patterns, balances, and an illustrative affordability heuristic. The trained model is then merged, exported to GGUF, quantized to **Q4_K_M**, and prepared for local inference through `llama.cpp`.

---

## Overview

The central design principle is:

```text
                    Incoming M-Pesa SMS
                            │
                            ▼
                 Gemma 3 270M + LoRA
                            │
                            ▼
                  Strict structured JSON
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
       Transaction Ledger          Deterministic Analytics
             │                             │
             │                    ┌────────┴────────┐
             │                    ▼                 ▼
             │              Cash Flow       Spending Analysis
             │                                      │
             └──────────────────┬───────────────────┘
                                ▼
                   Local Financial Intelligence
                                │
                                ▼
                    GGUF → Q4_K_M → llama.cpp
```

The LLM is deliberately responsible for **language understanding and structured extraction**. Financial arithmetic and bookkeeping are handled outside the LLM using deterministic Python code. This makes calculations auditable and avoids relying on a 270M-parameter model for multi-step financial arithmetic.

---

## What the Model Does

The fine-tuned model extracts five canonical fields from M-Pesa SMS messages:

```json
{
  "entity": "Ann Mueni",
  "amount": 20000.0,
  "balance": 159583.0,
  "date": "2026-03-05",
  "type": "income"
}
```

### Five-field schema

| Field | Description |
|---|---|
| `entity` | Person, merchant, bank, business, or organization involved |
| `amount` | Transaction amount in KES |
| `balance` | Observed M-Pesa balance after the transaction |
| `date` | Transaction date in `YYYY-MM-DD` format |
| `type` | `income` or `expense` |

The extraction prompt explicitly teaches the model to avoid returning transaction IDs, phone numbers, account numbers, merchant IDs, fees, categories, or additional JSON fields.

---

## Why This Architecture?

A small language model should not be responsible for everything.

Instead:

### Gemma handles

- Understanding varied M-Pesa SMS language
- Identifying the transaction entity
- Extracting amounts
- Extracting the observed balance
- Identifying transaction dates
- Classifying transactions as income or expense
- Producing a consistent JSON representation

### Python handles

- Ledger construction
- Balance tracking
- Cash-flow calculations
- Weekly/monthly spending totals
- Spending by entity
- Spending by category
- Savings-rate calculations
- Financial summaries
- Loan-affordability heuristic
- Validation of extracted records

This separation is a core design decision of the project.

---

# Dataset

The notebook uses a cleaned M-Pesa extraction dataset containing **200 records** with:

```text
input   → M-Pesa SMS
target  → structured transaction record
```

The dataset is split using a leakage-safe strategy:

| Split | Records |
|---|---:|
| Training | 140 |
| Validation | 30 |
| Test | 30 |
| **Total** | **200** |

The notebook checks for conflicting labels and prevents normalized SMS messages from appearing across multiple splits.

### Data cleaning

The pipeline:

1. Loads the source dataset.
2. Detects the SMS and target columns.
3. Normalizes JSON targets.
4. Canonicalizes the five-field financial schema.
5. Removes duplicate SMS/target pairs.
6. Checks for conflicting SMS labels.
7. Performs a leakage-safe train/validation/test split.
8. Saves processed datasets as JSONL.

---

# Fine-Tuning

The base model is:

```text
google/gemma-3-270m-it
```

LoRA is used to specialize the model without updating all base-model parameters.

### LoRA configuration

| Parameter | Value |
|---|---:|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Learning rate | 1e-4 |
| Epochs | 5 |
| Maximum sequence length | 512 |
| Gradient accumulation | 8 |
| Training batch size | 1 |
| Evaluation batch size | 1 |
| Weight decay | 0.01 |
| Max gradient norm | 1.0 |
| Training dtype | FP32 |
| Seed | 42 |

The LoRA adapters target:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Only about **0.7% of model parameters** are trainable during LoRA fine-tuning.

---

# Training Results

The notebook completed five training epochs.

Validation loss decreased consistently:

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 1.666162 | 0.373569 |
| 2 | 0.265191 | 0.099188 |
| 3 | 0.104493 | 0.069702 |
| 4 | 0.070435 | 0.056365 |
| 5 | 0.060221 | **0.051316** |

Final reported validation loss:

```text
0.05131591856479645
```

The notebook also performs numerical sanity checks before training, including parameter finiteness, forward-pass loss, gradient finiteness, and gradient-norm validation.

---

# Post-Training Extraction Test

The merged model was tested on an M-Pesa SMS:

```text
You have received Ksh20,000.00 from Ann Mueni.
Transaction cost Ksh0.00.
New balance Ksh159,583.00.
```

The model produced:

```json
{
  "entity": "Ann Mueni",
  "amount": 20000.0,
  "balance": 159583.0,
  "date": "2026-03-05",
  "type": "income"
}
```

The notebook validated:

- All five required fields were present.
- No extra fields were returned.
- Transaction type was valid.
- Balance was successfully extracted.
- JSON parsing succeeded.

The notebook also includes robust JSON extraction logic capable of recovering the first balanced JSON object from model output.

---

# End-to-End Ledger

After extraction, model outputs are converted into a transaction DataFrame.

The notebook demonstrates a local financial intelligence layer operating on the generated ledger.

Example summary from the notebook's model-generated ledger test:

```text
Transactions        : 10
Income              : KES 15,300.00
Expenses            : KES 13,890.00
Net cash flow       : KES 1,410.00
Latest balance      : KES 139,583.00
Top spending entity : SHELL WESTLANDS (KES 5,600.00)
Savings rate        : 9.22%
```

The extraction-quality check on that 10-record test produced:

```text
total_records        : 10
valid_entity         : 10
valid_amount         : 10
valid_balance        : 10
valid_date           : 9
valid_type           : 10
income_records       : 2
expense_records      : 8
unknown_type_records : 0
```

The notebook deliberately keeps the canonical ledger to five fields rather than introducing assumptions such as `balance_before` or `balance_after` into the final five-field model output.

---

# Financial Intelligence Layer

Once the SMS messages have been transformed into structured records, the system can answer natural-language financial questions without asking the LLM to perform the underlying arithmetic.

Examples demonstrated in the notebook include:

```text
Where is most of my money going?

How much did I spend this week?

How much did I spend this month?

What is my account balance?

Show me my cash flow.

What amount of loan can I apply?
```

Example outputs include:

```text
Your largest spending entity is SHELL WESTLANDS,
where you spent KES 5,600.00.

Your last observed M-PESA balance is
KES 139,583.00.

Your recorded income is KES 15,300.00,
your recorded expenses are KES 13,890.00,
and your net cash flow is KES 1,410.00.
```

The loan component is explicitly an **illustrative analytical heuristic**, not a lending decision, approval, or guaranteed borrowing amount.

---

# Model Export

After training:

```text
LoRA adapter
     │
     ▼
Merged Hugging Face model
     │
     ▼
FP16 GGUF
     │
     ▼
Q4_K_M GGUF
     │
     ▼
llama.cpp
```

The notebook clones `llama.cpp`, installs the conversion dependencies, converts the merged Hugging Face model to GGUF, and quantizes it using `Q4_K_M`.

### Generated artifacts

| Artifact | Purpose |
|---|---|
| LoRA adapter | Lightweight fine-tuning weights |
| Merged model | Base model + LoRA weights |
| FP16 GGUF | Unquantized GGUF deployment artifact |
| Q4_K_M GGUF | Compact deployment model for local inference |
| Transaction ledger | Structured financial records |
| Analysis ledger | Records prepared for deterministic analytics |
| Financial snapshot | Financial intelligence summary |
| Spending-by-entity CSV | Spending analysis |
| Final model report | Reproducibility metadata |

---

# Quantized Model

The final deployment target is:

```text
gemma3-financial-intelligence-Q4_K_M.gguf
```

Observed artifact size in the notebook:

```text
Q4_K_M GGUF: ~241.39 MB
FP16 GGUF  : ~517.69 MB
```

The quantized model is intended for lightweight local inference through `llama.cpp`.

During quantization, the resulting GGUF was approximately:

```text
235.16 MiB
7.36 bits/weight
```

The quantizer reported fallback quantization for some tensors, which is expected for the model's supported tensor configuration.

---

# ADTC Profiler Benchmark

A separate ADTC participant-mode benchmark was run against the generated Q4_K_M GGUF using `llama.cpp`.

### Participant benchmark snapshot

| Metric | Result |
|---|---:|
| Parameters detected | 268,098,176 |
| Generation throughput | **44.32 tokens/sec** |
| First-token latency | 1,985.11 ms |
| Peak RSS | **382.34 MB** |
| Steady-state RSS | 354.44 MB |
| CPU p99 | 58.8% |
| CPU throttling | False |
| GPU | None |
| Architecture | Gemma 3 |
| Parameter claim match | True |

The benchmark was executed in the Kaggle participant environment, which reported:

```text
CPU : Intel(R) Xeon(R) CPU @ 2.20GHz
RAM : 31.3 GB
GPU : none
OS  : Ubuntu 22.04.5 LTS
```

**Important:** these profiler numbers demonstrate that the GGUF runs successfully and provide a reproducible benchmark in that environment. They should not be presented as measurements from an 8 GB consumer laptop.

The ADTC profiler's built-in accuracy stage used `ARC-Easy`; its reported score was **0.52 on 50 samples**. This is a general benchmark and should not be interpreted as the model's M-Pesa extraction accuracy. The notebook's task-specific extraction test is a separate evaluation.

---

# Repository / Notebook Outputs

The notebook creates an output structure similar to:

```text
/kaggle/working/
│
├── data/
│   └── processed/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
│
├── models/
│   ├── lora/
│   ├── merged/
│   ├── checkpoints/
│   └── gguf/
│       ├── gemma3-financial-intelligence-f16.gguf
│       └── gemma3-financial-intelligence-Q4_K_M.gguf
│
├── export/
│   ├── transaction_ledger.csv
│   ├── analysis_ledger.csv
│   ├── spending_by_entity.csv
│   ├── financial_intelligence_snapshot.json
│   └── gguf_export_report.json
│
└── reports/
    └── final_model_report.json
```

---

# Reproducibility

The notebook fixes the random seed to:

```text
42
```

and records the main training configuration and generated artifact locations in:

```text
reports/final_model_report.json
```

The notebook also validates:

- Base-model parameters
- LoRA parameters
- Merged-model parameters
- NaN/Inf values
- Forward-pass numerical stability
- Gradient numerical stability
- JSON parsing
- Required extraction fields
- Transaction types
- Balance extraction

---

# Running the Notebook

The notebook was developed for a Kaggle GPU environment and uses a Tesla T4 during fine-tuning.

### Main dependencies

```text
Python 3.x
PyTorch
Transformers
PEFT
Datasets
Pandas
NumPy
scikit-learn
psutil
llama.cpp
```

The training environment used:

```text
PyTorch       2.10.0+cu128
Transformers  5.15.0
Datasets      5.0.1
PEFT          0.20.0
GPU           NVIDIA Tesla T4
```

The notebook installs/upgrades the required Python packages as part of the environment setup.

---

# Deployment Concept

The intended deployment architecture is **offline-first**:

```text
             M-Pesa SMS notification
                       │
                       ▼
              Local extraction model
                       │
                       ▼
              Structured transaction
                       │
                       ▼
                Local ledger
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Financial analytics       User questions
          │                         │
          └────────────┬────────────┘
                       ▼
              Local financial
                intelligence
```

The design minimizes the need for users to upload large financial statements and keeps the core transaction-processing workflow local.

---

# Security & Privacy Design

The architecture is designed around local processing:

- M-Pesa messages can be transformed locally into structured records.
- Financial calculations are performed locally.
- The compact GGUF model is suitable for local inference.
- The LLM does not need to perform the financial arithmetic itself.
- The system avoids requiring large statement uploads for the core ledger workflow.

Actual production deployment should still apply appropriate device security, data retention, encryption, access controls, and consent mechanisms.

---

# Limitations

This is a research/prototype pipeline rather than a production banking system.

Important limitations include:

1. The training dataset contains 200 records, so broader SMS diversity should be evaluated before production use.
2. The model is optimized for M-Pesa-style financial SMS extraction, not general financial reasoning.
3. The five-field extraction output depends on information available in the SMS and supplied context.
4. The loan component is an illustrative heuristic and is **not** a lender decision.
5. Deterministic analytics depend on the correctness of the extracted transaction records.
6. The ADTC ARC-Easy benchmark is not a direct measure of M-Pesa extraction quality.
7. The reported ADTC performance measurements were obtained in Kaggle rather than on an 8 GB consumer laptop.

---

# Future Work

Potential next steps include:

- Expand the M-Pesa SMS dataset substantially.
- Evaluate exact-record extraction accuracy on the held-out test set.
- Measure field-level accuracy for `entity`, `amount`, `balance`, `date`, and `type`.
- Add more M-Pesa transaction templates and edge cases.
- Test robustness to spelling, formatting, and message-template variation.
- Add multilingual/local-language SMS handling.
- Improve structured-output reliability.
- Benchmark the Q4_K_M model on an actual 8 GB laptop.
- Add continuous SMS ingestion and incremental ledger updates.
- Add stronger privacy and encrypted local storage.
- Evaluate the financial intelligence layer against independently verified ledger calculations.

---

# Project Status

### Completed

- [x] M-Pesa SMS dataset loading
- [x] Data cleaning and normalization
- [x] Leakage-safe train/validation/test split
- [x] Five-field structured extraction schema
- [x] Gemma 3 270M LoRA fine-tuning
- [x] Numerical sanity checks
- [x] JSON extraction and validation
- [x] Transaction ledger construction
- [x] Cash-flow analytics
- [x] Spending analysis
- [x] Natural-language financial question engine
- [x] LoRA adapter export
- [x] LoRA merge
- [x] FP16 GGUF conversion
- [x] Q4_K_M quantization
- [x] llama.cpp compatibility
- [x] ADTC profiler participant benchmark

### Next

- [ ] Large-scale task-specific extraction evaluation
- [ ] Exact-record accuracy benchmark
- [ ] Real 8 GB laptop benchmark
- [ ] Production-grade SMS ingestion
- [ ] Privacy/security hardening

---

## Core Idea

**Turn lightweight local language understanding into structured financial data, then let deterministic analytics turn that data into useful financial intelligence.**

The result is a compact, offline-oriented architecture designed around the hardware realities of African users rather than assuming cloud-scale compute.

---

## License

Add the project's intended license here before publishing the repository.
