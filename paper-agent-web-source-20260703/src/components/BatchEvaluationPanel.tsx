import { useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { FileUp, ListPlus, Play, RotateCcw, Square, Trash2 } from "lucide-react";
import { cancelSearchJob, getEvalCaseResult, runEvalCase } from "../lib/api";
import type { EvalCaseResult } from "../types/api";
import { MetricCard, Panel, StatusPill } from "./Common";

type BatchStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

type BatchEvalItem = {
  id: string;
  dataset: string;
  caseIndex: number;
  status: BatchStatus;
  result?: EvalCaseResult;
  error?: string;
};

const MAX_BATCH_CASES = 20;
const POLL_INTERVAL_MS = 2500;
const MAX_POLL_ATTEMPTS = 120;

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function formatScore(value: number | null | undefined) {
  return typeof value === "number" ? value.toFixed(3) : "--";
}

function formatCount(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

function isEvalPending(result: EvalCaseResult) {
  return !result.eval_metrics && result.progress.some((stage) => stage.status === "queued" || stage.status === "running");
}

function parseIndex(value: unknown): number | null {
  if (typeof value === "number" && Number.isInteger(value) && value >= 0) {
    return value;
  }
  if (typeof value === "string" && /^\d+$/.test(value.trim())) {
    return Number(value.trim());
  }
  return null;
}

function statusTone(status: BatchStatus) {
  if (status === "succeeded") {
    return "success";
  }
  if (status === "failed") {
    return "danger";
  }
  if (status === "cancelled") {
    return "warning";
  }
  if (status === "running") {
    return "warning";
  }
  return "muted";
}

function recordToItem(record: unknown, defaultDataset: string, inferredIndex: number, sequence: number): BatchEvalItem | null {
  if (typeof record === "number" || typeof record === "string") {
    const caseIndex = parseIndex(record);
    if (caseIndex === null) {
      return null;
    }
    return {
      id: `${defaultDataset}:${caseIndex}:${sequence}`,
      dataset: defaultDataset,
      caseIndex,
      status: "queued",
    };
  }

  if (!record || typeof record !== "object" || Array.isArray(record)) {
    return null;
  }

  const row = record as Record<string, unknown>;
  const explicitIndex = parseIndex(row.case_index ?? row.caseIndex ?? row.index);
  const caseIndex = explicitIndex ?? inferredIndex;
  const dataset = typeof row.dataset === "string" && row.dataset.trim() ? row.dataset.trim() : defaultDataset;

  return {
    id: `${dataset}:${caseIndex}:${sequence}`,
    dataset,
    caseIndex,
    status: "queued",
  };
}

function parseBatchInput(text: string, defaultDataset: string) {
  const trimmed = text.trim();
  const items: BatchEvalItem[] = [];
  let skipped = 0;

  if (!trimmed) {
    return { items, skipped };
  }

  if (trimmed.startsWith("[")) {
    try {
      const records = JSON.parse(trimmed) as unknown;
      if (!Array.isArray(records)) {
        return { items, skipped: 1 };
      }
      records.forEach((record, index) => {
        const item = recordToItem(record, defaultDataset, index, items.length);
        if (item) {
          items.push(item);
        } else {
          skipped += 1;
        }
      });
    } catch {
      return { items, skipped: 1 };
    }
  } else {
    const lines = trimmed.split(/\r?\n/).filter(Boolean);
    lines.forEach((line, lineIndex) => {
      const cleanLine = line.trim();
      if (!cleanLine) {
        return;
      }

      if (cleanLine.startsWith("{")) {
        try {
          const item = recordToItem(JSON.parse(cleanLine), defaultDataset, lineIndex, items.length);
          if (item) {
            items.push(item);
          } else {
            skipped += 1;
          }
        } catch {
          skipped += 1;
        }
        return;
      }

      cleanLine.split(/[\s,]+/).forEach((token) => {
        const item = recordToItem(token, defaultDataset, lineIndex, items.length);
        if (item) {
          items.push(item);
        } else if (token.trim()) {
          skipped += 1;
        }
      });
    });
  }

  const seen = new Set<string>();
  const uniqueItems = items.filter((item) => {
    const key = `${item.dataset}:${item.caseIndex}`;
    if (seen.has(key)) {
      skipped += 1;
      return false;
    }
    seen.add(key);
    return true;
  });

  const limitedItems = uniqueItems.slice(0, MAX_BATCH_CASES);
  skipped += Math.max(0, uniqueItems.length - limitedItems.length);

  return { items: limitedItems, skipped };
}

async function waitForEvalCompletion(initialResult: EvalCaseResult, shouldStop: () => boolean) {
  let result = initialResult;
  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS && isEvalPending(result) && !shouldStop(); attempt += 1) {
    await sleep(POLL_INTERVAL_MS);
    result = await getEvalCaseResult(result.job_id);
  }
  return result;
}

function sumProgressSeconds(result: EvalCaseResult | undefined) {
  return result?.progress.reduce((total, stage) => total + stage.elapsed_seconds, 0) ?? 0;
}

export function BatchEvaluationPanel({ dataset, config }: { dataset: string; config: string }) {
  const [batchInput, setBatchInput] = useState("");
  const [batchItems, setBatchItems] = useState<BatchEvalItem[]>([]);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [isStoppingBatch, setIsStoppingBatch] = useState(false);
  const [parseMessage, setParseMessage] = useState("No batch queued");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const stopBatchRequestedRef = useRef(false);
  const activeJobIdRef = useRef<string | null>(null);

  const completedItems = batchItems.filter((item) => item.status === "succeeded" && item.result?.eval_metrics);
  const finishedCount = batchItems.filter((item) => item.status === "succeeded" || item.status === "failed" || item.status === "cancelled").length;
  const failedCount = batchItems.filter((item) => item.status === "failed").length;
  const panelMeta = batchItems.length
    ? `${isBatchRunning ? finishedCount : completedItems.length}/${batchItems.length} completed${failedCount ? `, ${failedCount} failed` : ""}`
    : parseMessage;
  const batchSummary = useMemo(() => {
    const metricCount = completedItems.length;
    const totals = completedItems.reduce(
      (acc, item) => {
        const metrics = item.result?.eval_metrics;
        const budget = item.result?.budget;
        acc.f1 += metrics?.f1 ?? 0;
        acc.precision += metrics?.precision ?? 0;
        acc.recall += metrics?.recall ?? 0;
        acc.hits += metrics?.hits ?? 0;
        acc.gold += metrics?.gold_total ?? 0;
        acc.outputs += metrics?.output_total ?? 0;
        acc.elapsed += budget?.elapsed_seconds ?? sumProgressSeconds(item.result);
        acc.apiCalls += budget?.api_calls ?? 0;
        acc.llmCalls += budget?.llm_calls ?? 0;
        acc.tokens += budget?.token_estimate ?? 0;
        return acc;
      },
      {
        f1: 0,
        precision: 0,
        recall: 0,
        hits: 0,
        gold: 0,
        outputs: 0,
        elapsed: 0,
        apiCalls: 0,
        llmCalls: 0,
        tokens: 0,
      },
    );

    return {
      metricCount,
      averageF1: metricCount ? totals.f1 / metricCount : null,
      averagePrecision: metricCount ? totals.precision / metricCount : null,
      averageRecall: metricCount ? totals.recall / metricCount : null,
      hits: totals.hits,
      gold: totals.gold,
      outputs: totals.outputs,
      elapsed: totals.elapsed,
      apiCalls: totals.apiCalls,
      llmCalls: totals.llmCalls,
      tokens: totals.tokens,
    };
  }, [completedItems]);

  function updateBatchItem(id: string, patch: Partial<BatchEvalItem>) {
    setBatchItems((current) => current.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }

  function buildQueueFromText(text: string) {
    const parsed = parseBatchInput(text, dataset);
    setBatchItems(parsed.items);
    setParseMessage(`${parsed.items.length} queued${parsed.skipped ? `, ${parsed.skipped} skipped` : ""}`);
  }

  async function handleFileImport(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const text = await file.text();
    setBatchInput(text);
    buildQueueFromText(text);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function runOne(item: BatchEvalItem) {
    updateBatchItem(item.id, { status: "running", error: undefined, result: undefined });
    const initial = await runEvalCase({
      dataset: item.dataset,
      case_index: item.caseIndex,
      config,
      mode: "research",
      trace_gold: true,
    });
    activeJobIdRef.current = initial.job_id;
    const completed = await waitForEvalCompletion(initial, () => stopBatchRequestedRef.current);
    activeJobIdRef.current = null;
    if (stopBatchRequestedRef.current) {
      await cancelSearchJob(completed.job_id);
      updateBatchItem(item.id, { status: "cancelled", result: completed, error: "Batch stopped by user" });
      return;
    }
    if (isEvalPending(completed)) {
      throw new Error("Timed out while waiting for metrics");
    }
    updateBatchItem(item.id, { status: "succeeded", result: completed });
  }

  async function handleRunBatch() {
    if (!batchItems.length || isBatchRunning) {
      return;
    }

    const queue = batchItems.map((item) => ({ ...item, status: "queued" as const, result: undefined, error: undefined }));
    setBatchItems(queue);
    stopBatchRequestedRef.current = false;
    activeJobIdRef.current = null;
    setIsStoppingBatch(false);
    setIsBatchRunning(true);
    try {
      for (const item of queue) {
        if (stopBatchRequestedRef.current) {
          break;
        }
        try {
          await runOne(item);
        } catch (error) {
          updateBatchItem(item.id, {
            status: stopBatchRequestedRef.current ? "cancelled" : "failed",
            error: error instanceof Error ? error.message : "Unknown evaluation error",
          });
        }
      }
    } finally {
      activeJobIdRef.current = null;
      setIsBatchRunning(false);
      setIsStoppingBatch(false);
    }
  }

  async function handleStopBatch() {
    stopBatchRequestedRef.current = true;
    setIsStoppingBatch(true);
    if (activeJobIdRef.current) {
      await cancelSearchJob(activeJobIdRef.current);
    }
  }

  function handleUseSample() {
    const sample = `0\n1\n{"dataset":"${dataset}","case_index":2}\n{"query":"example raw benchmark row","gold":["reference title"]}`;
    setBatchInput(sample);
    buildQueueFromText(sample);
  }

  function handleClearBatch() {
    setBatchInput("");
    setBatchItems([]);
    setParseMessage("No batch queued");
  }

  return (
    <Panel title="Batch evaluation import" meta={`${panelMeta} / max ${MAX_BATCH_CASES}`}>
      <div className="batch-eval-layout">
        <div className="batch-import-panel">
          <textarea
            className="batch-input"
            rows={8}
            value={batchInput}
            onChange={(event) => setBatchInput(event.target.value)}
            placeholder={`0\n1\n{"dataset":"${dataset}","case_index":2}`}
          />
          <div className="batch-actions">
            <button className="button secondary" type="button" onClick={handleUseSample} disabled={isBatchRunning}>
              <RotateCcw size={16} />
              Sample
            </button>
            <label className={`button secondary batch-file-button ${isBatchRunning ? "disabled" : ""}`}>
              <FileUp size={16} />
              Import
              <input ref={fileInputRef} type="file" accept=".jsonl,.json,.txt" onChange={handleFileImport} disabled={isBatchRunning} />
            </label>
            <button className="button secondary" type="button" onClick={() => buildQueueFromText(batchInput)} disabled={isBatchRunning}>
              <ListPlus size={16} />
              Queue
            </button>
            <button className="button primary" type="button" onClick={handleRunBatch} disabled={isBatchRunning || !batchItems.length}>
              <Play size={16} />
              {isBatchRunning ? "Running" : "Run batch"}
            </button>
            {isBatchRunning ? (
              <button className="button danger" type="button" onClick={() => void handleStopBatch()} disabled={isStoppingBatch}>
                <Square size={16} />
                {isStoppingBatch ? "Stopping" : "Stop"}
              </button>
            ) : null}
            <button className="icon-button danger-hover" type="button" onClick={handleClearBatch} disabled={isBatchRunning} aria-label="Clear batch">
              <Trash2 size={16} />
            </button>
          </div>
        </div>

        <div className="batch-summary-grid">
          <MetricCard label="Avg F1" value={formatScore(batchSummary.averageF1)} detail={`${batchSummary.metricCount}/${batchItems.length} completed`} />
          <MetricCard label="Avg recall" value={formatScore(batchSummary.averageRecall)} detail={`${batchSummary.hits}/${batchSummary.gold} gold hits`} />
          <MetricCard label="Tokens" value={formatCount(batchSummary.tokens)} detail={`${batchSummary.apiCalls} API / ${batchSummary.llmCalls} LLM calls`} />
          <MetricCard label="Elapsed" value={`${batchSummary.elapsed.toFixed(1)}s`} detail={`${batchSummary.outputs} final papers`} />
        </div>
      </div>

      <div className="table-wrap scrollbar-thin batch-table-wrap">
        <table className="data-table batch-table">
          <thead>
            <tr>
              <th>Case</th>
              <th>Status</th>
              <th>F1</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>Budget</th>
              <th>Job</th>
            </tr>
          </thead>
          <tbody>
            {batchItems.map((item) => (
              <tr key={item.id}>
                <td>
                  <strong>{item.dataset}</strong>
                  <span>#{item.caseIndex}</span>
                </td>
                <td>
                  <StatusPill tone={statusTone(item.status)} label={item.status} />
                  {item.error ? <small className="batch-error">{item.error}</small> : null}
                </td>
                <td>{formatScore(item.result?.eval_metrics?.f1)}</td>
                <td>{formatScore(item.result?.eval_metrics?.precision)}</td>
                <td>{formatScore(item.result?.eval_metrics?.recall)}</td>
                <td>
                  <span>{formatCount(item.result?.budget?.token_estimate ?? 0)} tokens</span>
                  <small>{(item.result?.budget?.elapsed_seconds ?? sumProgressSeconds(item.result)).toFixed(1)}s</small>
                </td>
                <td>{item.result?.job_id ?? "--"}</td>
              </tr>
            ))}
            {!batchItems.length ? (
              <tr>
                <td colSpan={7}>No batch cases queued.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
