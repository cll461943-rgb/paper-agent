import { AlertTriangle, CheckCircle2, Gauge, GitBranch, ListChecks, Route, Target, Timer } from "lucide-react";
import { Link } from "react-router-dom";
import type { WorkflowResult } from "../types/api";
import { Panel, StatusPill } from "./Common";

type RequirementStatus = "ready" | "watch" | "missing";

interface RequirementRow {
  title: string;
  weight: string;
  evidence: string;
  status: RequirementStatus;
}

function formatScore(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "--";
}

function formatCompactNumber(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}k`;
  }
  return value.toString();
}

function getBestExpectedF1(result: WorkflowResult) {
  const curve = result.expected_f1_curve ? Object.values(result.expected_f1_curve).map(Number).filter(Number.isFinite) : [];
  return curve.length ? Math.max(...curve) : null;
}

function statusTone(status: RequirementStatus): "success" | "warning" | "danger" {
  if (status === "ready") {
    return "success";
  }
  if (status === "watch") {
    return "warning";
  }
  return "danger";
}

function statusLabel(status: RequirementStatus) {
  if (status === "ready") {
    return "ready";
  }
  if (status === "watch") {
    return "watch";
  }
  return "missing";
}

function buildRequirementRows(result: WorkflowResult): RequirementRow[] {
  const metrics = result.run_metrics;
  const queryPlan = result.query_plan;
  const highCount = result.highly_relevant_papers.length;
  const partialCount = result.partially_relevant_papers.length;
  const clusterCount = result.method_clusters.length;
  const timelineCount = result.timeline.length;
  const graphNodeCount = Array.isArray(result.citation_graph.nodes) ? result.citation_graph.nodes.length : 0;
  const hasEvalMetrics = Boolean(result.benchmark_metrics);

  return [
    {
      title: "F1 / gold evidence",
      weight: "70%",
      evidence: hasEvalMetrics
        ? `F1=${formatScore(result.benchmark_metrics?.f1)}, P=${formatScore(result.benchmark_metrics?.precision)}, R=${formatScore(result.benchmark_metrics?.recall)}`
        : `No gold labels attached; expected-F1 max=${formatScore(getBestExpectedF1(result))}`,
      status: hasEvalMetrics ? "ready" : "watch",
    },
    {
      title: "Query understanding",
      weight: "core",
      evidence: `${queryPlan.methods.length} methods, ${queryPlan.datasets.length} datasets, ${queryPlan.must_have_constraints.length} required constraints`,
      status: queryPlan.must_have_constraints.length && queryPlan.methods.length ? "ready" : "missing",
    },
    {
      title: "Iterative retrieval",
      weight: "core",
      evidence: `${metrics.retrieval_rounds_used} rounds, ${metrics.search_queries_used} search queries, ${metrics.candidate_pool_size.toLocaleString()} candidates`,
      status: metrics.retrieval_rounds_used >= 2 && metrics.candidate_pool_size >= 100 ? "ready" : "watch",
    },
    {
      title: "Ranked final answer",
      weight: "core",
      evidence: `${highCount} highly relevant, ${partialCount} partially relevant, dynamic K=${result.dynamic_k_chosen ?? "--"}`,
      status: highCount + partialCount > 0 ? "ready" : "missing",
    },
    {
      title: "Cost and latency",
      weight: "20%",
      evidence: `${metrics.elapsed_seconds.toFixed(1)}s, ${metrics.api_calls_used} API calls, ${formatCompactNumber(metrics.token_estimate)} tokens`,
      status: metrics.errors.length ? "watch" : "ready",
    },
    {
      title: "Structured output",
      weight: "10%",
      evidence: `${clusterCount} clusters, ${timelineCount} timeline points, ${graphNodeCount} graph nodes`,
      status: clusterCount && graphNodeCount ? "ready" : "watch",
    },
  ];
}

export function ContestReadiness({ result, jobId }: { result: WorkflowResult; jobId: string }) {
  const metrics = result.run_metrics;
  const bestExpectedF1 = getBestExpectedF1(result);
  const hasEvalMetrics = Boolean(result.benchmark_metrics);
  const requirementRows = buildRequirementRows(result);
  const readyCount = requirementRows.filter((row) => row.status === "ready").length;
  const watchCount = requirementRows.filter((row) => row.status === "watch").length;
  const structureArtifacts = [
    result.highly_relevant_papers.length + result.partially_relevant_papers.length > 0,
    result.method_clusters.length > 0,
    Array.isArray(result.citation_graph.nodes) && result.citation_graph.nodes.length > 0,
  ].filter(Boolean).length;

  return (
    <Panel title="Researcher contest review" meta={`${readyCount}/${requirementRows.length} ready`} className="contest-review-panel">
      <div className="researcher-verdict">
        <div className="verdict-icon">
          {hasEvalMetrics ? <CheckCircle2 size={22} /> : <AlertTriangle size={22} />}
        </div>
        <div>
          <span>科研用户评价</span>
          <p>
            这个网站已经能把复杂学术查询拆成可审计的检索流程，并给出排序、证据、聚类和图谱；但作为赛题作品，最迫切的缺口是把真实
            gold-based F1、召回流失和成本预算放到同一个驾驶舱里，否则研究者很难判断一次结果是否值得信任。
          </p>
        </div>
      </div>

      <div className="contest-score-grid">
        <div className="contest-score-card primary">
          <Target size={18} />
          <span>{hasEvalMetrics ? "Measured F1" : "Expected F1 proxy"}</span>
          <strong>{hasEvalMetrics ? formatScore(result.benchmark_metrics?.f1) : formatScore(bestExpectedF1)}</strong>
          <small>{hasEvalMetrics ? "from eval gold labels" : "needs /eval confirmation"}</small>
        </div>
        <div className="contest-score-card">
          <Timer size={18} />
          <span>Efficiency</span>
          <strong>{metrics.elapsed_seconds.toFixed(1)}s</strong>
          <small>{metrics.api_calls_used} API / {formatCompactNumber(metrics.token_estimate)} tokens</small>
        </div>
        <div className="contest-score-card">
          <GitBranch size={18} />
          <span>Structure</span>
          <strong>{structureArtifacts}/3</strong>
          <small>list, clusters, graph</small>
        </div>
      </div>

      <div className="requirement-table-wrap">
        <table className="requirement-table">
          <thead>
            <tr>
              <th>Requirement</th>
              <th>Weight</th>
              <th>Evidence</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {requirementRows.map((row) => (
              <tr key={row.title}>
                <td>{row.title}</td>
                <td>{row.weight}</td>
                <td>{row.evidence}</td>
                <td>
                  <StatusPill tone={statusTone(row.status)} label={statusLabel(row.status)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="urgent-needs-grid">
        <article>
          <ListChecks size={16} />
          <div>
            <h3>最迫切：评测驾驶舱</h3>
            <p>用 AutoScholar / RealScholar 单例和批量结果证明 F1、Precision、Recall，而不是只看漂亮列表。</p>
            <Link className="button secondary small" to="/eval">Open evaluation</Link>
          </div>
        </article>
        <article>
          <Route size={16} />
          <div>
            <h3>召回流失追踪</h3>
            <p>按 retrieval、selection、ranking、synthesis 显示 gold paper 在哪一层掉队，优先修召回瓶颈。</p>
            <Link className="button secondary small" to={`/logs/${jobId}`}>Open logs</Link>
          </div>
        </article>
        <article>
          <Gauge size={16} />
          <div>
            <h3>成本预算护栏</h3>
            <p>把 API 调用、Token、延迟和 provider 错误放在同一视图里，支撑赛题 20% 运行效率评分。</p>
            <Link className="button secondary small" to="/settings">Open settings</Link>
          </div>
        </article>
      </div>

      {watchCount ? <p className="contest-review-note">{watchCount} items need evidence before this run can be treated as contest-ready.</p> : null}
    </Panel>
  );
}
