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
  mockEvalCaseResult,
  mockGraphResponse,
  mockLogs,
  mockPapers,
  mockProviders,
  mockRecentRuns,
  mockResultsResponse,
  mockSearchJob,
  mockStageArtifacts,
  mockSystemStatus,
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

function createLocalSearchJob(payload: SearchRequest): LocalSearchJob {
  const timestamp = new Date();
  const suffix = timestamp
    .toISOString()
    .replace(/\D/g, "")
    .slice(0, 14);

  return {
    ...mockSearchJob,
    job_id: `search_${suffix}`,
    status: "succeeded",
    stage: "synthesis",
    progress: 100,
    elapsed_seconds: payload.mode === "mock" ? 3.2 : 18.7,
    request: payload,
    created_at: timestamp.toISOString(),
  };
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

function withLocalQuery<T extends ResultsResponse>(response: T, job: LocalSearchJob): T {
  if (!response.result) {
    return {
      ...response,
      job_id: job.job_id,
    };
  }

  return {
    ...response,
    job_id: job.job_id,
    result: {
      ...response.result,
      original_query: job.request.query,
      query_plan: {
        ...response.result.query_plan,
        original_query: job.request.query,
      },
    },
  };
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

function randomJob(status: SearchJob["status"] = "running"): SearchJob {
  return {
    ...mockSearchJob,
    status,
    stage: status === "succeeded" ? "synthesis" : "retrieval",
    progress: status === "succeeded" ? 100 : 42,
    elapsed_seconds: status === "succeeded" ? 18.7 : 6.1,
  };
}

export async function startSearch(payload: SearchRequest): Promise<SearchJob> {
  const job = await requestJson<SearchJob>(
    payload.retrieval_only ? "/api/search/retrieval-only" : "/api/search",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    async () => {
      const localJob = createLocalSearchJob(payload);
      writeLocalSearchJob(localJob);
      return localJob;
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
      const cancelledJob: SearchJob = {
        ...(localJob ?? mockSearchJob),
        job_id: jobId,
        status: "cancelled",
        stage: localJob?.stage ?? "cancelled",
        progress: localJob?.progress ?? 0,
        elapsed_seconds: localJob?.elapsed_seconds ?? 0,
        error: "Cancelled locally",
      };
      if (localJob) {
        writeLocalSearchJob({ ...localJob, ...cancelledJob });
      }
      return cancelledJob;
    },
  );
}

export async function getSearchJob(jobId: string): Promise<SearchJob> {
  return requestJson<SearchJob>(
    `/api/search/${jobId}`,
    { method: "GET" },
    async () => getLocalSearchJob(jobId) ?? { ...randomJob("succeeded"), job_id: jobId },
  );
}

export async function getResults(jobId: string): Promise<ResultsResponse> {
  return requestJson<ResultsResponse>(
    `/api/results/${jobId}`,
    { method: "GET" },
    async () => {
      const localJob = getLocalSearchJob(jobId);
      return localJob ? withLocalQuery(mockResultsResponse, localJob) : { ...mockResultsResponse, job_id: jobId };
    },
  );
}

export async function getStageArtifact(jobId: string, stageName: string): Promise<StageArtifactResponse> {
  return requestJson<StageArtifactResponse>(
    `/api/results/${jobId}/stage/${stageName}`,
    { method: "GET" },
    async () => ({
      ...(mockStageArtifacts[stageName] ?? {
        stage: stageName,
        job_id: jobId,
        data: {},
      }),
      job_id: jobId,
    }),
  );
}

export async function getConfigs(): Promise<string[]> {
  return requestJson<string[]>("/api/configs", { method: "GET" }, async () => mockConfigs);
}

export async function getProviders(): Promise<ProviderStatus[]> {
  return requestJson<ProviderStatus[]>("/api/providers", { method: "GET" }, async () => mockProviders);
}

export async function getSystemStatus(): Promise<SystemStatus> {
  return requestJson<SystemStatus>("/api/system/status", { method: "GET" }, async () => mockSystemStatus);
}

export async function getDatabaseStatus(): Promise<DatabaseStatus | null> {
  return requestJson<DatabaseStatus | null>("/api/database/status", { method: "GET" }, async () => null);
}

export async function getDatasets(): Promise<string[]> {
  return requestJson<string[]>("/api/datasets", { method: "GET" }, async () => mockDatasets);
}

export async function runEvalCase(payload: EvalCaseRequest): Promise<EvalCaseResult> {
  return requestJson<EvalCaseResult>(
    "/api/eval/case",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    async () => ({
      ...mockEvalCaseResult,
      dataset: payload.dataset,
      case_index: payload.case_index,
    }),
  );
}

export async function getEvalCaseResult(jobId: string): Promise<EvalCaseResult> {
  return requestJson<EvalCaseResult>(`/api/search/${jobId}`, { method: "GET" }, async () => ({
    ...mockEvalCaseResult,
    job_id: jobId,
  }));
}

export async function runRandomEvalCase(): Promise<EvalCaseResult> {
  return requestJson<EvalCaseResult>(
    "/api/eval/random-case",
    {
      method: "POST",
      body: JSON.stringify({}),
    },
    async () => mockEvalCaseResult,
  );
}

export async function getGraph(jobId: string): Promise<GraphResponse> {
  return requestJson<GraphResponse>(
    `/api/graph/${jobId}`,
    { method: "GET" },
    async () => {
      const localJob = getLocalSearchJob(jobId);
      return {
        ...mockGraphResponse,
        job_id: jobId,
        query: localJob?.request.query ?? mockGraphResponse.query,
      };
    },
  );
}

export async function getPaper(paperId: string): Promise<Paper | undefined> {
  return requestJson<Paper | undefined>(
    `/api/papers/${paperId}`,
    { method: "GET" },
    async () => mockPapers.find((paper) => paper.paper_id === paperId),
  );
}

export async function getLogs(jobId: string): Promise<LogEntry[]> {
  return requestJson<LogEntry[]>(
    `/api/logs/${jobId}`,
    { method: "GET" },
    async () => mockLogs.map((log) => ({ ...log, id: `${jobId}-${log.id}` })),
  );
}

export async function getRecentRuns(): Promise<SearchJob[]> {
  const dismissedIds = readDismissedRunIds();
  const localRuns = readLocalSearchJobs();
  const backendRuns = await requestJson<SearchJob[]>("/api/recent-runs", { method: "GET" }, async () => []);
  return mergeRecentRuns(localRuns, backendRuns, mockRecentRuns).filter((run) => !dismissedIds.has(run.job_id));
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
