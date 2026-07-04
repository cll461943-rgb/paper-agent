import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { GitBranch, RefreshCw, Square } from "lucide-react";
import { cancelSearchJob, getResults, getSearchJob, getStageArtifact } from "../lib/api";
import type { ResultsResponse, SearchJob, StageArtifactResponse } from "../types/api";
import { ContestReadiness } from "../components/ContestReadiness";
import { JsonBlock, MetricCard, PageHeader, Panel, PaperCard, ProgressBar, StatusPill } from "../components/Common";

const stageNames = ["query_plan", "retrieval", "selection", "ranking", "synthesis"];

function isActiveStatus(status: string | undefined) {
  return status === "queued" || status === "running";
}

function statusTone(status: string | undefined) {
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

export function ResultsPage() {
  const { jobId = "search_20260630_1042" } = useParams();
  const [result, setResult] = useState<ResultsResponse | null>(null);
  const [job, setJob] = useState<SearchJob | null>(null);
  const [artifact, setArtifact] = useState<StageArtifactResponse | null>(null);
  const [activeStage, setActiveStage] = useState(stageNames[0]);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    async function refreshRun() {
      const [jobOutcome, resultOutcome] = await Promise.allSettled([getSearchJob(jobId), getResults(jobId)]);
      if (cancelled) {
        return;
      }

      const nextJob = jobOutcome.status === "fulfilled" ? jobOutcome.value : null;
      const nextResult = resultOutcome.status === "fulfilled" ? resultOutcome.value : null;
      const errors = [
        jobOutcome.status === "rejected" ? describeRunError(jobOutcome.reason) : "",
        resultOutcome.status === "rejected" ? describeRunError(resultOutcome.reason) : "",
      ].filter(Boolean);

      if (nextJob) {
        setJob(nextJob);
      }
      if (nextResult) {
        setResult(nextResult);
      }
      setLoadError(errors.length && !nextResult ? errors.join(" / ") : "");

      const status = nextJob?.status ?? nextResult?.status;
      const stillRunning = isActiveStatus(status) || (!status && !nextResult?.result);
      if (stillRunning && !errors.length) {
        timer = window.setTimeout(refreshRun, 5000);
      }
    }

    void refreshRun();
    return () => {
      cancelled = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [jobId]);

  useEffect(() => {
    setArtifact(null);
    void getStageArtifact(jobId, activeStage)
      .then(setArtifact)
      .catch((error) =>
        setArtifact({
          stage: activeStage,
          job_id: jobId,
          data: { error: describeRunError(error) },
        }),
      );
  }, [activeStage, jobId]);

  const papers = useMemo(() => {
    if (!result?.result) {
      return [];
    }
    return [
      ...result.result.highly_relevant_papers,
      ...result.result.partially_relevant_papers,
      ...result.result.supporting_papers,
    ];
  }, [result]);

  if (!result) {
    const terminalStatus = job?.status && !isActiveStatus(job.status);
    if (loadError || terminalStatus) {
      return (
        <>
          <PageHeader
            eyebrow="Results"
            title="Result unavailable"
            description={loadError || job?.error || `Run ended with status ${job?.status ?? "unknown"}.`}
            actions={
              <Link className="button secondary" to="/">
                Back to workbench
              </Link>
            }
          />
          <Panel title="Run state" meta={job?.status ?? "api"}>
            <div className="job-card">
              <div className="job-card-top">
                <strong>{jobId}</strong>
                <StatusPill tone={statusTone(job?.status)} label={job?.status ?? "unavailable"} />
              </div>
              <ProgressBar value={job?.progress ?? 0} />
              <dl className="compact-dl two-col">
                <div>
                  <dt>Stage</dt>
                  <dd>{job?.stage ?? "api"}</dd>
                </div>
                <div>
                  <dt>Elapsed</dt>
                  <dd>{job?.elapsed_seconds?.toFixed(1) ?? "--"}s</dd>
                </div>
              </dl>
              <p className="graph-empty-note">{loadError || job?.error}</p>
            </div>
          </Panel>
        </>
      );
    }

    return <PageHeader eyebrow="Results" title="Loading result" description="Fetching run artifacts from the backend API." />;
  }

  if (!result.result) {
    const status = job?.status ?? result.status;
    const progress = job?.progress ?? 0;
    const canCancel = isActiveStatus(status);
    const handleCancel = async () => {
      if (canCancel) {
        setJob(await cancelSearchJob(jobId));
      }
    };
    return (
      <>
        <PageHeader
          eyebrow="Result Workspace"
          title={jobId}
          description={`Pipeline is ${status}${job?.stage ? ` / ${job.stage}` : ""}. Results will refresh automatically.`}
        />

        <Panel title="Run in progress" meta={status}>
          <div className="job-card">
            <div className="job-card-top">
              <strong>{jobId}</strong>
              <StatusPill tone={statusTone(status)} label={status} />
            </div>
            <ProgressBar value={progress} />
            <dl className="compact-dl two-col">
              <div>
                <dt>Stage</dt>
                <dd>{job?.stage ?? "queued"}</dd>
              </div>
              <div>
                <dt>Elapsed</dt>
                <dd>{job?.elapsed_seconds?.toFixed(1) ?? "--"}s</dd>
              </div>
            </dl>
            {canCancel ? (
              <div className="job-card-actions">
                <button className="button danger small" type="button" onClick={() => void handleCancel()}>
                  <Square size={12} />
                  Stop
                </button>
              </div>
            ) : null}
            {job?.error ? <p className="graph-empty-note">{job.error}</p> : null}
          </div>
        </Panel>
      </>
    );
  }

  const metrics = result.result.run_metrics;
  const finalPaperCount =
    result.result.highly_relevant_papers.length +
    result.result.partially_relevant_papers.length;

  return (
    <>
      <PageHeader
        eyebrow="Result Workspace"
        title={result.job_id}
        description={result.result.original_query}
        actions={
          <Link className="button primary" to={`/graph/${result.job_id}`}>
            <GitBranch size={16} />
            Open graph
          </Link>
        }
      />

      <div className="metric-grid">
        <MetricCard label="Candidates" value={metrics.candidate_pool_size.toLocaleString()} detail="retrieved pool" />
        <MetricCard label="Final papers" value={finalPaperCount || metrics.final_papers} detail={`dynamic k=${result.result.dynamic_k_chosen ?? "--"}`} />
        <MetricCard label="LLM calls" value={metrics.llm_calls_used} detail={`${metrics.token_estimate.toLocaleString()} tokens`} />
        <MetricCard label="Elapsed" value={`${metrics.elapsed_seconds.toFixed(1)}s`} detail={`${metrics.cache_hits} cache hits`} />
      </div>

      <ContestReadiness result={result.result} jobId={result.job_id} />

      <div className="results-grid">
        <Panel title="Ranked papers" meta={`${papers.length} visible`} className="results-list">
          <div className="paper-list scrollbar-thin">
            {papers.map((paper) => (
              <PaperCard item={paper} key={paper.paper.paper_id} />
            ))}
          </div>
        </Panel>

        <Panel title="Decision trace" meta="workflow">
          <div className="timeline-list">
            {result.result.search_process.map((round) => (
              <div className="timeline-item" key={round.round_index}>
                <span>Round {round.round_index}</span>
                <h3>{round.search_goal}</h3>
                <p>{round.review_conclusion}</p>
                <small>{round.candidates_found.toLocaleString()} candidates</small>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Stage artifact" meta={activeStage}>
          <div className="stage-tabs">
            {stageNames.map((stage) => (
              <button className={stage === activeStage ? "active" : ""} key={stage} type="button" onClick={() => setActiveStage(stage)}>
                {stage.replace("_", " ")}
              </button>
            ))}
          </div>
          <JsonBlock data={artifact?.data ?? { loading: true }} />
        </Panel>
      </div>

      <Panel title="Component metrics" meta="per stage">
        <div className="table-wrap scrollbar-thin">
          <table className="data-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Elapsed</th>
                <th>Items</th>
                <th>API</th>
                <th>LLM</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {metrics.component_metrics.map((metric) => (
                <tr key={metric.component}>
                  <td>{metric.component}</td>
                  <td>{metric.elapsed_seconds.toFixed(1)}s</td>
                  <td>{metric.items_delta.toLocaleString()}</td>
                  <td>{metric.api_calls_total}</td>
                  <td>{metric.llm_calls_total}</td>
                  <td>
                    <StatusPill tone={metric.errors_total ? "danger" : "success"} label={metric.errors_total ? "errors" : "clean"} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <button className="floating-refresh" type="button" onClick={() => void getResults(jobId).then(setResult)} aria-label="Refresh results">
        <RefreshCw size={18} />
      </button>
    </>
  );
}
