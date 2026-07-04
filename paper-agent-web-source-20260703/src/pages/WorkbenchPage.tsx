import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, CheckCircle2, Play, RotateCcw, Square } from "lucide-react";
import { cancelSearchJob, getConfigs, getProviders, getSearchJob, startSearch } from "../lib/api";
import { mockSearchRequest, mockWorkflowStages } from "../lib/mockData";
import { getRuntimeConnectionConfig } from "../lib/runtimeConfig";
import type { ProviderStatus, SearchJob, SearchMode, SearchRequest } from "../types/api";
import { MetricCard, PageHeader, Panel, ProgressBar, StatusPill } from "../components/Common";

function isActiveJob(job: SearchJob | null): job is SearchJob {
  return job?.status === "queued" || job?.status === "running";
}

function jobTone(status: SearchJob["status"]) {
  if (status === "succeeded") {
    return "success";
  }
  if (status === "failed") {
    return "danger";
  }
  if (status === "cancelled") {
    return "warning";
  }
  return "info";
}

function describeRunError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function createFailedStartJob(error: unknown, mode: SearchMode): SearchJob {
  const hint =
    mode === "local"
      ? "Local mode still needs the backend service, but it does not spend external LLM quota."
      : "If the model gateway is out of quota, switch Mode to local and run again.";

  return {
    job_id: `api_error_${Date.now()}`,
    status: "failed",
    stage: "api",
    progress: 0,
    elapsed_seconds: 0,
    error: `${describeRunError(error)} ${hint}`,
  };
}

export function WorkbenchPage() {
  const navigate = useNavigate();
  const runtimeConfig = getRuntimeConnectionConfig();
  const [configs, setConfigs] = useState<string[]>([]);
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [job, setJob] = useState<SearchJob | null>(null);
  const [request, setRequest] = useState<SearchRequest>({
    ...mockSearchRequest,
    config: runtimeConfig.defaultConfig || mockSearchRequest.config,
    mode: runtimeConfig.defaultSearchMode || mockSearchRequest.mode,
  });
  const [isRunning, setIsRunning] = useState(false);
  const canOpenResult = job ? !job.job_id.startsWith("api_error_") : false;
  const resultJobId = canOpenResult ? job?.job_id : null;

  useEffect(() => {
    void Promise.all([getConfigs(), getProviders()]).then(([nextConfigs, nextProviders]) => {
      setConfigs(nextConfigs);
      setProviders(nextProviders);
    });
  }, []);

  useEffect(() => {
    setRequest((current) => ({
      ...current,
      config: runtimeConfig.defaultConfig || current.config,
      mode: runtimeConfig.defaultSearchMode || current.mode,
    }));
  }, [runtimeConfig.defaultConfig, runtimeConfig.defaultSearchMode]);

  useEffect(() => {
    if (!isActiveJob(job)) {
      setIsRunning(false);
      return;
    }

    const timer = window.setTimeout(() => {
      void getSearchJob(job.job_id).then(setJob);
    }, 1500);

    return () => window.clearTimeout(timer);
  }, [job]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setIsRunning(true);
    try {
      const nextJob = await startSearch(request);
      setJob(nextJob);
    } catch (error) {
      setJob(createFailedStartJob(error, request.mode));
      setIsRunning(false);
    }
  }

  async function handleCancelJob() {
    if (!job || !isActiveJob(job)) {
      return;
    }
    setJob(await cancelSearchJob(job.job_id));
    setIsRunning(false);
  }

  function toggleProvider(name: string) {
    setRequest((current) => ({
      ...current,
      providers: current.providers.includes(name)
        ? current.providers.filter((provider) => provider !== name)
        : [...current.providers, name],
    }));
  }

  return (
    <>
      <PageHeader
        eyebrow="Unified Workbench"
        title="Research run control"
        description="Compose a query, choose providers, run the pipeline, and keep the result path one click away."
        actions={
          resultJobId ? (
            <button className="button primary" type="button" onClick={() => navigate(`/results/${resultJobId}`)}>
              <ArrowRight size={16} />
              Open result
            </button>
          ) : null
        }
      />

      <div className="workbench-grid">
        <Panel title="Search request" className="request-panel">
          <form className="search-form" onSubmit={handleSubmit}>
            <label>
              Query
              <textarea
                value={request.query}
                onChange={(event) => setRequest((current) => ({ ...current, query: event.target.value }))}
                rows={7}
              />
            </label>

            <div className="form-row">
              <label>
                Mode
                <select value={request.mode} onChange={(event) => setRequest((current) => ({ ...current, mode: event.target.value as SearchMode }))}>
                  <option value="research">research</option>
                  <option value="live">live</option>
                  <option value="local">local</option>
                  <option value="mock">mock</option>
                </select>
              </label>
              <label>
                Config
                <select value={request.config} onChange={(event) => setRequest((current) => ({ ...current, config: event.target.value }))}>
                  {(configs.length ? configs : [request.config]).map((config) => (
                    <option key={config} value={config}>
                      {config}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="toggle-row">
              <label>
                <input
                  type="checkbox"
                  checked={request.retrieval_only}
                  onChange={(event) => setRequest((current) => ({ ...current, retrieval_only: event.target.checked }))}
                />
                Retrieval only
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={request.use_local_index}
                  onChange={(event) => setRequest((current) => ({ ...current, use_local_index: event.target.checked }))}
                />
                Local index
              </label>
            </div>

            <div className="provider-grid">
              {providers.map((provider) => (
                <button
                  className={`provider-tile ${request.providers.includes(provider.name) ? "selected" : ""}`}
                  key={provider.name}
                  type="button"
                  onClick={() => toggleProvider(provider.name)}
                >
                  <span>{provider.name}</span>
                  <StatusPill tone={provider.status === "healthy" ? "success" : "warning"} label={`${provider.latency_ms}ms`} />
                </button>
              ))}
            </div>

            <div className="form-actions">
              <button
                className="button secondary"
                type="button"
                onClick={() =>
                  setRequest({
                    ...mockSearchRequest,
                    config: runtimeConfig.defaultConfig || mockSearchRequest.config,
                    mode: runtimeConfig.defaultSearchMode || mockSearchRequest.mode,
                  })
                }
              >
                <RotateCcw size={16} />
                Reset
              </button>
              <button className="button primary" type="submit" disabled={isRunning || !request.query.trim()}>
                <Play size={16} />
                {isRunning ? "Running" : "Run pipeline"}
              </button>
            </div>
          </form>
        </Panel>

        <div className="side-stack">
          <Panel title="Current job">
            {job ? (
              <div className="job-card">
                <div className="job-card-top">
                  <strong>{job.job_id}</strong>
                  <StatusPill tone={jobTone(job.status)} label={job.status} />
                </div>
                <ProgressBar value={job.progress ?? 0} />
                <dl className="compact-dl two-col">
                  <div>
                    <dt>Stage</dt>
                    <dd>{job.stage}</dd>
                  </div>
                  <div>
                    <dt>Elapsed</dt>
                    <dd>{job.elapsed_seconds?.toFixed(1) ?? "--"}s</dd>
                  </div>
                </dl>
                <div className="job-card-actions">
                  {canOpenResult ? (
                    <button className="button secondary small" type="button" onClick={() => navigate(`/results/${job.job_id}`)}>
                      <ArrowRight size={12} />
                      Result
                    </button>
                  ) : null}
                  {isActiveJob(job) ? (
                    <button className="button danger small" type="button" onClick={() => void handleCancelJob()}>
                      <Square size={12} />
                      Stop
                    </button>
                  ) : null}
                </div>
                {job.error ? <p className="graph-empty-note">{job.error}</p> : null}
              </div>
            ) : (
              <div className="empty-state">
                <CheckCircle2 size={24} />
                <p>No active run. The next submitted query will appear here.</p>
              </div>
            )}
          </Panel>

          <Panel title="Pipeline stages" meta="baseline">
            <div className="stage-list">
              {mockWorkflowStages.map((stage) => (
                <div className="stage-item" key={stage.key}>
                  <span>{stage.label}</span>
                  <strong>{stage.elapsed.toFixed(1)}s</strong>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>

      <div className="metric-grid">
        <MetricCard label="Providers" value={providers.length || 4} detail="backend/local routed" />
        <MetricCard label="Query mode" value={request.mode} detail={request.config} />
        <MetricCard label="Selected sources" value={request.providers.length} detail="remote + local" />
        <MetricCard label="Pipeline" value="6 stages" detail="retrieval to synthesis" />
      </div>
    </>
  );
}
