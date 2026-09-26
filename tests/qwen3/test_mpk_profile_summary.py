import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "summarize_qwen3_mpk_profile.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_qwen3_mpk_profile", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SGLANG_MODULE_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "summarize_sglang_trace.py"
)
SGLANG_SPEC = importlib.util.spec_from_file_location(
    "summarize_sglang_trace", SGLANG_MODULE_PATH
)
SGLANG_MODULE = importlib.util.module_from_spec(SGLANG_SPEC)
SGLANG_SPEC.loader.exec_module(SGLANG_MODULE)


def test_task_categories_cover_qwen3_decode_tasks():
    assert MODULE.task_category("TASK_PAGED_ATTENTION_HOPPER") == "attention"
    assert MODULE.task_category("TASK_RMS_NORM_HOPPER") == "norm"
    assert MODULE.task_category("TASK_LINEAR_HOPPER") == "linear"
    assert MODULE.task_category("TASK_SILU_MUL_HOPPER") == "mlp_activation"
    assert MODULE.task_category("TASK_ARGMAX_REDUCE") == "sampling"
    assert MODULE.task_category("TASK_SCHD_EVENTS") == "scheduler"


def test_aggregate_reports_worker_time_without_calling_it_wall_time():
    records = [
        {"category": "attention", "task_type_name": "A", "duration_ns": 1000},
        {"category": "attention", "task_type_name": "A", "duration_ns": 3000},
        {"category": "linear", "task_type_name": "B", "duration_ns": 6000},
    ]
    rows = MODULE.aggregate(records, "category", decode_steps=2)
    assert rows[0]["category"] == "linear"
    assert rows[0]["total_worker_time_ms"] == 0.006
    assert rows[0]["worker_time_ms_per_decode_step"] == 0.003
    attention = next(row for row in rows if row["category"] == "attention")
    assert attention["worker_time_share"] == 0.4
    assert attention["p50_execution_us"] == 2.0


def test_sglang_kernel_categories_and_trace_filter():
    assert SGLANG_MODULE.kernel_category("flashinfer::BatchDecode") == "attention"
    assert SGLANG_MODULE.kernel_category("ampere_bf16_s16816gemm") == "gemm"
    assert SGLANG_MODULE.kernel_category("fused_rmsnorm") == "norm"
    events = [
        {"ph": "X", "cat": "kernel", "name": "flashinfer_attention", "dur": 7},
        {"ph": "X", "cat": "cpu_op", "name": "aten::matmul", "dur": 100},
    ]
    records = SGLANG_MODULE.cuda_kernel_events(events)
    assert len(records) == 1
    assert records[0]["duration_us"] == 7
