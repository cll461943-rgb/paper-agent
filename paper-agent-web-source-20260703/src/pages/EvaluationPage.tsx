import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { FlaskConical, Shuffle } from "lucide-react";
import { getConfigs, getDatasets, runEvalCase, runRandomEvalCase } from "../lib/api";
import type { EvalCaseResult } from "../types/api";
import { PageHeader, Panel, StatusPill } from "../components/Common";

export function EvaluationPage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [configs, setConfigs] = useState<string[]>([]);
  const [dataset, setDataset] = useState("MedEval-2024");
  const [config, setConfig] = useState("local_full_pipeline.yaml");
  const [caseIndex, setCaseIndex] = useState(128);
  const [result, setResult] = useState<EvalCaseResult | null>(null);

  useEffect(() => {
    void Promise.all([getDatasets(), getConfigs()]).then(([nextDatasets, nextConfigs]) => {
      setDatasets(nextDatasets);
      setConfigs(nextConfigs);
      setDataset((current) => nextDatasets[0] ?? current);
      setConfig((current) => nextConfigs[0] ?? current);
    });
  }, []);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setResult(await runEvalCase({ dataset, case_index: caseIndex, config, mode: "research", trace_gold: true }));
  }

  async function handleRandom() {
    setResult(await runRandomEvalCase());
  }

  return (
    <>
      <PageHeader eyebrow="Evaluation" title="Single-case diagnostics" description="Run deterministic evaluation slices against datasets and inspect pipeline pass-through." />

      <div className="eval-grid">
        <Panel title="Case runner">
          <form className="search-form" onSubmit={handleSubmit}>
            <label>
              Dataset
              <select value={dataset} onChange={(event) => setDataset(event.target.value)}>
                {(datasets.length ? datasets : [dataset]).map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Case index
              <input type="number" value={caseIndex} min={0} onChange={(event) => setCaseIndex(Number(event.target.value))} />
            </label>
            <label>
              Config
              <select value={config} onChange={(event) => setConfig(event.target.value)}>
                {(configs.length ? configs : [config]).map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <div className="form-actions">
              <button className="button secondary" type="button" onClick={handleRandom}>
                <Shuffle size={16} />
                Random
              </button>
              <button className="button primary" type="submit">
                <FlaskConical size={16} />
                Run case
              </button>
            </div>
          </form>
        </Panel>

        <Panel title="Progress">
          <div className="stage-list">
            {(result?.progress ?? []).map((stage) => (
              <div className="stage-item" key={stage.stage}>
                <span>{stage.stage}</span>
                <strong>{stage.elapsed_seconds.toFixed(1)}s</strong>
              </div>
            ))}
            {!result ? <p className="muted-copy">Run a case to populate stage diagnostics.</p> : null}
          </div>
        </Panel>
      </div>

      {result ? (
        <Panel title={result.job_id} meta={`${result.dataset} / case ${result.case_index}`}>
          <div className="eval-summary">
            <div>
              <span>Query</span>
              <strong>{result.query}</strong>
            </div>
            <div>
              <span>Gold</span>
              <strong>{result.gold.join(", ")}</strong>
            </div>
          </div>
          <div className="table-wrap scrollbar-thin">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Paper</th>
                  <th>Source</th>
                  <th>Score</th>
                  <th>Pass</th>
                </tr>
              </thead>
              <tbody>
                {[...result.candidate_pool, ...result.selection_candidates, ...result.ranked_papers, ...result.final_output].map((row, index) => (
                  <tr key={`${row.stage}-${row.paper_id}-${index}`}>
                    <td>{row.stage}</td>
                    <td>{row.title}</td>
                    <td>{row.source}</td>
                    <td>{row.score?.toFixed(3) ?? "--"}</td>
                    <td>
                      <StatusPill tone={row.pass ? "success" : "danger"} label={row.pass ? "pass" : "hold"} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}
    </>
  );
}
