import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getLogs } from "../lib/api";
import type { LogEntry } from "../types/api";
import { JsonBlock, PageHeader, Panel, StatusPill } from "../components/Common";

const logTone: Record<LogEntry["level"], "success" | "warning" | "danger" | "info"> = {
  INFO: "info",
  WARN: "warning",
  ERROR: "danger",
  DEBUG: "success",
};

export function LogsPage() {
  const { jobId = "search_20260630_1042" } = useParams();
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [selected, setSelected] = useState<LogEntry | null>(null);

  useEffect(() => {
    void getLogs(jobId).then((nextLogs) => {
      setLogs(nextLogs);
      setSelected(nextLogs[0] ?? null);
    });
  }, [jobId]);

  return (
    <>
      <PageHeader eyebrow="Observability" title="Run logs" description={`Event stream for ${jobId}.`} />
      <div className="logs-grid">
        <Panel title="Events" meta={`${logs.length} lines`}>
          <div className="log-list scrollbar-thin">
            {logs.map((log) => (
              <button className={selected?.id === log.id ? "active" : ""} key={log.id} type="button" onClick={() => setSelected(log)}>
                <StatusPill tone={logTone[log.level]} label={log.level} />
                <span>{log.stage}</span>
                <strong>{log.message}</strong>
              </button>
            ))}
          </div>
        </Panel>
        <Panel title="Raw payload" meta={selected?.timestamp ? new Date(selected.timestamp).toLocaleString() : "none"}>
          <JsonBlock data={selected ?? { status: "waiting" }} />
        </Panel>
      </div>
    </>
  );
}
