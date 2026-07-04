from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from typing import Any

from .batch_eval import BatchCase, BatchCaseResult, load_dataset_cases, run_batch_cases


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class DesktopBatchGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Paper Agent Batch Evaluation")
        self.root.geometry("1120x720")
        self.dataset_path = tk.StringVar()
        self.api_base = tk.StringVar(value="http://127.0.0.1:8080")
        self.config = tk.StringVar(value="default.yaml")
        self.mode = tk.StringVar(value="mock")
        self.limit = tk.StringVar(value="20")
        self.timeout_seconds = tk.StringVar(value="60")
        self.workers = tk.StringVar(value="4")
        self.status_text = tk.StringVar(value="Ready")
        self.summary_text = tk.StringVar(value="0 cases")
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_requested = False
        self.worker_thread: threading.Thread | None = None
        self.results: list[BatchCaseResult] = []
        self._build()
        self.root.after(150, self._drain_events)

    def _build(self) -> None:
        shell = ttk.Frame(self.root, padding=14)
        shell.pack(fill=BOTH, expand=True)

        top = ttk.Frame(shell)
        top.pack(fill=X)

        ttk.Label(top, text="Dataset").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(top, textvariable=self.dataset_path).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(top, text="Browse", command=self._browse_dataset).grid(row=0, column=2, padx=(8, 0), pady=4)

        ttk.Label(top, text="API").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(top, textvariable=self.api_base, width=28).grid(row=1, column=1, sticky="w", pady=4)

        controls = ttk.Frame(top)
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for label, variable, width in [
            ("Config", self.config, 24),
            ("Mode", self.mode, 10),
            ("Limit", self.limit, 8),
            ("Timeout", self.timeout_seconds, 8),
            ("Workers", self.workers, 8),
        ]:
            ttk.Label(controls, text=label).pack(side=LEFT, padx=(0, 5))
            ttk.Entry(controls, textvariable=variable, width=width).pack(side=LEFT, padx=(0, 14))

        self.run_button = ttk.Button(controls, text="Run batch", command=self._start_batch)
        self.run_button.pack(side=RIGHT)
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop_batch, state="disabled")
        self.stop_button.pack(side=RIGHT, padx=(0, 8))

        top.columnconfigure(1, weight=1)

        summary = ttk.Frame(shell)
        summary.pack(fill=X, pady=(14, 8))
        ttk.Label(summary, textvariable=self.status_text).pack(side=LEFT)
        ttk.Label(summary, textvariable=self.summary_text).pack(side=RIGHT)

        columns = ("case", "status", "f1", "precision", "recall", "tokens", "elapsed", "message")
        self.table = ttk.Treeview(shell, columns=columns, show="headings", height=24)
        for column, label, width in [
            ("case", "Case", 90),
            ("status", "Status", 100),
            ("f1", "F1", 80),
            ("precision", "Precision", 90),
            ("recall", "Recall", 90),
            ("tokens", "Tokens", 90),
            ("elapsed", "Elapsed", 90),
            ("message", "Message", 420),
        ]:
            self.table.heading(column, text=label)
            self.table.column(column, width=width, anchor="w")
        self.table.pack(fill=BOTH, expand=True)

    def _browse_dataset(self) -> None:
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Supported datasets", "*.json *.jsonl *.csv *.tsv"),
                ("JSON datasets", "*.json *.jsonl"),
                ("Delimited datasets", "*.csv *.tsv"),
                ("All files", "*.*"),
            ]
        )
        if filename:
            self.dataset_path.set(filename)

    def _start_batch(self) -> None:
        path = Path(self.dataset_path.get())
        if not path.exists():
            messagebox.showerror("Dataset missing", "Select an existing JSON, JSONL, CSV, or TSV dataset.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            return

        self.stop_requested = False
        self.results = []
        self.table.delete(*self.table.get_children())
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_text.set("Loading dataset")
        self.worker_thread = threading.Thread(target=self._run_batch_worker, daemon=True)
        self.worker_thread.start()

    def _stop_batch(self) -> None:
        self.stop_requested = True
        self.status_text.set("Stopping batch")
        self.stop_button.configure(state="disabled")

    def _run_batch_worker(self) -> None:
        try:
            limit = int(self.limit.get() or "0") or None
            timeout_seconds = float(self.timeout_seconds.get() or "60")
            workers = int(self.workers.get() or "4")
            cases = load_dataset_cases(self.dataset_path.get(), limit=limit)
            self.events.put(("loaded", len(cases)))
            results = run_batch_cases(
                cases,
                self._run_case_via_api(timeout_seconds),
                per_case_timeout_seconds=timeout_seconds,
                max_workers=workers,
                should_stop=lambda: self.stop_requested,
                on_result=lambda result: self.events.put(("result", result)),
            )
            self.events.put(("done", results))
        except Exception as exc:  # noqa: BLE001 - GUI should surface any batch startup failure.
            self.events.put(("error", str(exc)))

    def _run_case_via_api(self, timeout_seconds: float):
        def run(case: BatchCase) -> dict[str, Any]:
            api_base = self.api_base.get().rstrip("/")
            started = time.perf_counter()
            payload = {
                "dataset": case.dataset,
                "case_index": case.case_index,
                "config": self.config.get(),
                "mode": self.mode.get(),
                "trace_gold": True,
            }
            current = _post_json(f"{api_base}/api/eval/case", payload, timeout=min(timeout_seconds, 10))
            job_id = current.get("job_id")
            while job_id and not current.get("eval_metrics") and time.perf_counter() - started < timeout_seconds:
                if self.stop_requested:
                    try:
                        _post_json(f"{api_base}/api/search/{job_id}/cancel", {}, timeout=3)
                    except Exception:
                        pass
                    raise RuntimeError("cancelled")
                time.sleep(1.5)
                current = _get_json(f"{api_base}/api/search/{job_id}", timeout=min(timeout_seconds, 10))
            return current

        return run

    def _drain_events(self) -> None:
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "loaded":
                self.status_text.set(f"Running {payload} cases")
            elif event == "result":
                self._append_result(payload)
            elif event == "done":
                self._finish(payload)
            elif event == "error":
                self.status_text.set(f"Error: {payload}")
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
        self.root.after(150, self._drain_events)

    def _append_result(self, result: BatchCaseResult) -> None:
        self.results.append(result)
        metrics = (result.payload or {}).get("eval_metrics") if isinstance(result.payload, dict) else None
        budget = (result.payload or {}).get("budget") if isinstance(result.payload, dict) else None
        self.table.insert(
            "",
            END,
            values=(
                f"#{result.case.case_index}",
                result.status,
                _metric(metrics, "f1"),
                _metric(metrics, "precision"),
                _metric(metrics, "recall"),
                str((budget or {}).get("token_estimate", "--")),
                f"{result.elapsed_seconds:.1f}s",
                result.error or (result.payload or {}).get("job_id", ""),
            ),
        )
        self._update_summary()

    def _finish(self, results: list[BatchCaseResult]) -> None:
        self.results = results
        self.status_text.set("Stopped" if self.stop_requested else "Complete")
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._update_summary()

    def _update_summary(self) -> None:
        completed = [result for result in self.results if result.status == "succeeded"]
        failed = [result for result in self.results if result.status != "succeeded"]
        f1_values = [
            result.payload.get("eval_metrics", {}).get("f1")
            for result in completed
            if isinstance(result.payload, dict)
        ]
        f1_values = [value for value in f1_values if isinstance(value, (int, float))]
        avg_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
        self.summary_text.set(f"{len(completed)} succeeded / {len(failed)} failed or stopped / avg F1 {avg_f1:.3f}")


def _metric(metrics: dict[str, Any] | None, key: str) -> str:
    value = (metrics or {}).get(key)
    return f"{value:.3f}" if isinstance(value, (int, float)) else "--"


def main() -> None:
    root = tk.Tk()
    DesktopBatchGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
