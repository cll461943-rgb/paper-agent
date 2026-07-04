import type {
  BackendProbeResult,
  DatabaseStatus,
  EvalCaseRequest,
  EvalCaseResult,
  GraphResponse,
  LogEntry,
  Paper,
  ProviderStatus,
  ResultsResponse,
  SearchJob,
  SearchRequest,
  StageArtifactResponse,
  SystemStatus,
} from "../types/api";
import { getRuntimeConnectionConfig } from "./runtimeConfig";
import {
  mockConfigs,
  mockDatasets,
  mockPapers,
  mockProviders,
} from "./mockData";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const LOCAL_SEARCH_JOBS_KEY = "paper-agent.local-search-jobs";
const LOCAL_DISMISSED_RUNS_KEY = "paper-agent.dismissed-runs";
const SEARCH_RUN_STARTED_EVENT = "scholar-agent:search-run-started";

type LocalSearchJob = SearchJob & {
  request: SearchRequest;
  created_at: string;
};

function readLocalSearchJobs(): LocalSearchJob[] {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    return JSON.parse(window.localStorage.getItem(LOCAL_SEARCH_JOBS_KEY) ?? "[]") as LocalSearchJob[];
  } catch {
    return [];
  }
}

function writeLocalSearchJob(job: LocalSearchJob) {
  if (typeof window === "undefined") {
    return;
  }

  const jobs = readLocalSearchJobs().filter((item) => item.job_id !== job.job_id);
  window.localStorage.setItem(LOCAL_SEARCH_JOBS_KEY, JSON.stringify([job, ...jobs].slice(0, 8)));
}

function removeLocalSearchJob(jobId: string) {
  if (typeof window === "undefined") {
    return;
  }

  const jobs = readLocalSearchJobs().filter((item) => item.job_id !== jobId);
  window.localStorage.setItem(LOCAL_SEARCH_JOBS_KEY, JSON.stringify(jobs));
}

function readDismissedRunIds(): Set<string> {
  if (typeof window === "undefined") {
    return new Set();
  }

  try {
    const ids = JSON.parse(window.localStorage.getItem(LOCAL_DISMISSED_RUNS_KEY) ?? "[]") as string[];
    return new Set(ids);
  } catch {
    return new Set();
  }
}

function dismissRun(jobId: string) {
  if (typeof window === "undefined") {
    return;
  }

  const ids = Array.from(readDismissedRunIds());
  window.localStorage.setItem(LOCAL_DISMISSED_RUNS_KEY, JSON.stringify([jobId, ...ids.filter((id) => id !== jobId)].slice(0, 32)));
}

function notifySearchRunStarted(job: SearchJob) {
  if (typeof window === "undefined") {
    return;
  }

  window.dispatchEvent(new CustomEvent(SEARCH_RUN_STARTED_EVENT, { detail: job }));
}

function getLocalSearchJob(jobId: string): LocalSearchJob | undefined {
  return readLocalSearchJobs().find((job) => job.job_id === jobId);
}

function mergeRecentRuns(...groups: SearchJob[][]): SearchJob[] {
  const seen = new Set<string>();
  const merged: SearchJob[] = [];

  for (const group of groups) {
    for (const run of group) {
      if (seen.has(run.job_id)) {
        continue;
      }
      seen.add(run.job_id);
      merged.push(run);
    }
  }

  return merged;
}

function getApiBaseUrl(): string {
  const runtimeConfig = getRuntimeConnectionConfig();
  return runtimeConfig.apiBaseUrl || API_BASE;
}

function buildUrl(path: string): string {
  const apiBaseUrl = getApiBaseUrl();
  return apiBaseUrl ? `${apiBaseUrl}${path}` : path;
}

function buildRequestInit(init: RequestInit, timeoutMsOverride?: number): { requestInit: RequestInit; timeoutMs: number } {
  const runtimeConfig = getRuntimeConnectionConfig();
  const headers = new Headers(init.headers ?? {});
  const timeoutMs = timeoutMsOverride ?? runtimeConfig.requestTimeoutMs;

  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  if (runtimeConfig.apiKey) {
    const authValue = runtimeConfig.apiAuthScheme ? `${runtimeConfig.apiAuthScheme} ${runtimeConfig.apiKey}` : runtimeConfig.apiKey;
    headers.set(runtimeConfig.apiAuthHeader, authValue);
  }

  if (runtimeConfig.llmProvider) {
    headers.set("X-Scholar-LLM-Provider", runtimeConfig.llmProvider);
  }
  if (runtimeConfig.llmBaseUrl) {
    headers.set("X-Scholar-LLM-Base-URL", runtimeConfig.llmBaseUrl);
  }
  if (runtimeConfig.llmApiKey) {
    headers.set("X-Scholar-LLM-API-Key", runtimeConfig.llmApiKey);
  }
  if (runtimeConfig.llmModel) {
    headers.set("X-Scholar-LLM-Model", runtimeConfig.llmModel);
  }
  if (runtimeConfig.defaultConfig) {
    headers.set("X-Scholar-Default-Config", runtimeConfig.defaultConfig);
  }
  headers.set("X-Scholar-Default-Mode", runtimeConfig.defaultSearchMode);

  return {
    timeoutMs,
    requestInit: {
      ...init,
      headers,
    },
  };
}

async function fetchJson(path: string, init: RequestInit, timeoutMsOverride?: number): Promise<Response> {
  const { requestInit, timeoutMs } = buildRequestInit(init, timeoutMsOverride);
  const controller = new AbortController();
  const upstreamSignal = requestInit.signal;
  const finalInit = {
    ...requestInit,
    signal: controller.signal,
  };
  const timeoutHandle = window.setTimeout(() => controller.abort(), timeoutMs);

  if (upstreamSignal instanceof AbortSignal) {
    if (upstreamSignal.aborted) {
      controller.abort();
    } else {
      upstreamSignal.addEventListener("abort", () => controller.abort(), { once: true });
    }
  }

  try {
    return await fetch(buildUrl(path), finalInit);
  } finally {
    window.clearTimeout(timeoutHandle);
  }
}

async function requestJson<T>(path: string, init: RequestInit, fallback: () => T | Promise<T>): Promise<T> {
  try {
    const response = await fetchJson(path, init);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return (await response.json()) as T;
  } catch (error) {
    console.warn(`[Scholar Agent API fallback] ${path}`, error);
    return fallback();
  }
}

function describeError(error: unknown): string {
  if (error instanceof Error) {
    if (error.name === "AbortError") {
      return "Request timed out";
    }
    return error.message;
  }
  return String(error);
}

async function requestJsonStrict<T>(path: string, init: RequestInit, timeoutMsOverride?: number): Promise<T> {
  try {
    const response = await fetchJson(path, init, timeoutMsOverride);
    if (!response.ok) {
      let detail = "";
      try {
        const body = (await response.json()) as { error?: string; message?: string };
        detail = body.error || body.message || "";
      } catch {
        detail = await response.text().catch(() => "");
      }
      throw new Error(`HTTP ${response.status}${detail ? `: ${detail}` : ""}`);
    }
    return (await response.json()) as T;
  } catch (error) {
    throw new Error(`Backend API request failed for ${path}: ${describeError(error)}`);
  }
}

function degradedSystemStatus(error: unknown): SystemStatus {
  const runtimeConfig = getRuntimeConnectionConfig();
  return {
    status: "degraded",
    message: `Backend API unavailable: ${describeError(error)}`,
    mode: runtimeConfig.defaultSearchMode,
    config: runtimeConfig.defaultConfig,
    cache_hit_rate: 0,
    local_index_ready: false,
    vector_index_ready: false,
    provider_count: {
      healthy: 0,
      total: 0,
    },
  };
}

function cancelledFallbackJob(jobId: string, localJob?: LocalSearchJob): SearchJob {
  return {
    job_id: jobId,
    status: "cancelled",
    stage: localJob?.stage ?? "cancelled",
    progress: localJob?.progress ?? 0,
    elapsed_seconds: localJob?.elapsed_seconds ?? 0,
    error: "Cancelled locally because the backend did not acknowledge the stop request.",
  };
}

function failedFallbackJob(jobId: string, error: unknown, localJob?: LocalSearchJob): SearchJob {
  return {
    job_id: jobId,
    status: "failed",
    stage: localJob?.stage ?? "api",
    progress: localJob?.progress ?? 0,
    elapsed_seconds: localJob?.elapsed_seconds ?? 0,
    error: describeError(error),
  };
}

export async function startSearch(payload: SearchRequest): Promise<SearchJob> {
  const job = await requestJsonStrict<SearchJob>(
    payload.retrieval_only ? "/api/search/retrieval-only" : "/api/search",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );

  writeLocalSearchJob({
    ...job,
    request: payload,
    created_at: new Date().toISOString(),
  });
  notifySearchRunStarted(job);
  return job;
}

export async function deleteSearchJob(jobId: string): Promise<void> {
  dismissRun(jobId);
  removeLocalSearchJob(jobId);

  try {
    const response = await fetchJson(`/api/search/${jobId}`, { method: "DELETE" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    console.warn(`[Scholar Agent API fallback] delete ${jobId}`, error);
  }
}

export async function cancelSearchJob(jobId: string): Promise<SearchJob> {
  return requestJson<SearchJob>(
    `/api/search/${jobId}/cancel`,
    { method: "POST" },
    async () => {
      const localJob = getLocalSearchJob(jobId);
      const cancelledJob = cancelledFallbackJob(jobId, localJob);
      if (localJob) {
        writeLocalSearchJob({ ...localJob, ...cancelledJob });
      }
      return cancelledJob;
    },
  );
}

export async function getSearchJob(jobId: string): Promise<SearchJob> {
  try {
    return await requestJsonStrict<SearchJob>(`/api/search/${jobId}`, { method: "GET" });
  } catch (error) {
    const localJob = getLocalSearchJob(jobId);
    if (localJob) {
      return failedFallbackJob(jobId, error, localJob);
    }
    throw error;
  }
}

export async function getResults(jobId: string): Promise<ResultsResponse> {
  return requestJsonStrict<ResultsResponse>(`/api/results/${jobId}`, { method: "GET" });
}

export async function getStageArtifact(jobId: string, stageName: string): Promise<StageArtifactResponse> {
  return requestJsonStrict<StageArtifactResponse>(`/api/results/${jobId}/stage/${stageName}`, { method: "GET" });
}

export async function getConfigs(): Promise<string[]> {
  return requestJson<string[]>("/api/configs", { method: "GET" }, async () => mockConfigs);
}

export async function getProviders(): Promise<ProviderStatus[]> {
  return requestJson<ProviderStatus[]>("/api/providers", { method: "GET" }, async () => mockProviders);
}

export async function getSystemStatus(): Promise<SystemStatus> {
  try {
    return await requestJsonStrict<SystemStatus>("/api/system/status", { method: "GET" });
  } catch (error) {
    console.warn("[Scholar Agent API degraded] /api/system/status", error);
    return degradedSystemStatus(error);
  }
}

export async function getDatabaseStatus(): Promise<DatabaseStatus | null> {
  return requestJson<DatabaseStatus | null>("/api/database/status", { method: "GET" }, async () => null);
}

export async function getDatasets(): Promise<string[]> {
  return requestJson<string[]>("/api/datasets", { method: "GET" }, async () => mockDatasets);
}

export async function runEvalCase(payload: EvalCaseRequest): Promise<EvalCaseResult> {
  return requestJsonStrict<EvalCaseResult>(
    "/api/eval/case",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export async function getEvalCaseResult(jobId: string): Promise<EvalCaseResult> {
  return requestJsonStrict<EvalCaseResult>(`/api/search/${jobId}`, { method: "GET" });
}

export async function runRandomEvalCase(): Promise<EvalCaseResult> {
  return requestJsonStrict<EvalCaseResult>(
    "/api/eval/random-case",
    {
      method: "POST",
      body: JSON.stringify({}),
    },
  );
}

export async function getGraph(jobId: string): Promise<GraphResponse> {
  return requestJsonStrict<GraphResponse>(`/api/graph/${jobId}`, { method: "GET" });
}

export async function getPaper(paperId: string): Promise<Paper | undefined> {
  return requestJson<Paper | undefined>(
    `/api/papers/${paperId}`,
    { method: "GET" },
    async () => mockPapers.find((paper) => paper.paper_id === paperId),
  );
}

export async function getLogs(jobId: string): Promise<LogEntry[]> {
  return requestJsonStrict<LogEntry[]>(`/api/logs/${jobId}`, { method: "GET" });
}

export async function getRecentRuns(): Promise<SearchJob[]> {
  const dismissedIds = readDismissedRunIds();
  const localRuns = readLocalSearchJobs();
  const backendRuns = await requestJson<SearchJob[]>("/api/recent-runs", { method: "GET" }, async () => []);
  return mergeRecentRuns(localRuns, backendRuns).filter((run) => !dismissedIds.has(run.job_id));
}

export async function probeBackendConnection(): Promise<BackendProbeResult> {
  const path = "/api/system/status";
  const url = buildUrl(path);

  try {
    const response = await fetchJson(path, { method: "GET" }, 8000);
    const contentType = response.headers.get("content-type") ?? "";

    if (!response.ok) {
      return {
        ok: false,
        url,
        status: response.status,
        contentType,
        message: `HTTP ${response.status}`,
      };
    }

    if (!contentType.includes("application/json")) {
      return {
        ok: false,
        url,
        status: response.status,
        contentType,
        message: `Expected JSON but received ${contentType || "unknown content type"}`,
      };
    }

    return {
      ok: true,
      url,
      status: response.status,
      contentType,
      message: "Connected to backend API",
    };
  } catch (error) {
    return {
      ok: false,
      url,
      status: null,
      message: error instanceof Error ? error.message : "Unknown connection error",
    };
  }
}
