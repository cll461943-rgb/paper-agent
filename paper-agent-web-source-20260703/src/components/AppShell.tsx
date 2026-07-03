import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Copy,
  Database,
  FlaskConical,
  GitBranch,
  LayoutDashboard,
  ScrollText,
  Settings,
  Trash2,
} from "lucide-react";
import { deleteSearchJob, getDatabaseStatus, getRecentRuns, getSystemStatus } from "../lib/api";
import type { DatabaseStatus, DatabaseStoreStatus, SearchJob, SystemStatus } from "../types/api";
import { StatusPill } from "./Common";
import scholarAgentLogo from "../assets/scholar-agent-icon.png";

const DEFAULT_JOB_ID = "search_20260630_1042";
const SEARCH_RUN_STARTED_EVENT = "scholar-agent:search-run-started";

export function AppShell() {
  const location = useLocation();
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [runs, setRuns] = useState<SearchJob[]>([]);
  const [dbStatus, setDbStatus] = useState<DatabaseStatus | null>(null);
  const [copiedDbPath, setCopiedDbPath] = useState("");

  const handleDeleteRun = async (jobId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (window.confirm(`Delete run ${jobId}?`)) {
      try {
        await deleteSearchJob(jobId);
        const nextRuns = await getRecentRuns();
        setRuns(nextRuns);
      } catch (err) {
        console.error("Failed to delete job", err);
      }
    }
  };

  const handleCopyDbPath = async (path: string) => {
    await navigator.clipboard.writeText(path);
    setCopiedDbPath(path);
    window.setTimeout(() => setCopiedDbPath(""), 1800);
  };

  useEffect(() => {
    let cancelled = false;
    let interval: number | undefined;

    async function refreshShellData() {
      const [nextStatus, nextRuns, nextDbStatus] = await Promise.all([
        getSystemStatus(),
        getRecentRuns(),
        getDatabaseStatus(),
      ]);
      if (cancelled) {
        return;
      }
      setStatus(nextStatus);
      setRuns(nextRuns);
      setDbStatus(nextDbStatus);
    }

    const refreshListener = () => {
      void refreshShellData();
    };

    void refreshShellData();
    interval = window.setInterval(refreshListener, 5000);
    window.addEventListener(SEARCH_RUN_STARTED_EVENT, refreshListener);

    return () => {
      cancelled = true;
      if (interval) {
        window.clearInterval(interval);
      }
      window.removeEventListener(SEARCH_RUN_STARTED_EVENT, refreshListener);
    };
  }, [location.pathname]);

  const latestJobId = runs[0]?.job_id ?? DEFAULT_JOB_ID;
  const navItems = [
    { to: "/", label: "Workbench", icon: LayoutDashboard },
    { to: `/results/${latestJobId}`, label: "Results", icon: BarChart3 },
    { to: `/graph/${latestJobId}`, label: "Graph", icon: GitBranch },
    { to: "/eval", label: "Evaluation", icon: FlaskConical },
    { to: `/logs/${latestJobId}`, label: "Logs", icon: ScrollText },
    { to: "/settings", label: "Settings", icon: Settings },
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark brand-mark-image">
            <img src={scholarAgentLogo} alt="Scholar Agent" />
          </div>
          <div>
            <strong>Scholar Agent</strong>
            <span>Academic intelligence console</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="Primary">
          {navItems.map((item) => (
            <NavLink key={item.to} to={item.to} className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
              <item.icon size={18} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-card">
          <div className="sidebar-card-head">
            <Activity size={16} />
            <span>Runtime</span>
          </div>
          <StatusPill tone={status?.status === "ok" ? "success" : "warning"} label={status?.status ?? "loading"} />
          <dl className="compact-dl">
            <div>
              <dt>Config</dt>
              <dd>{status?.config ?? "local_full_pipeline.yaml"}</dd>
            </div>
            <div>
              <dt>Cache</dt>
              <dd>{status ? `${status.cache_hit_rate.toFixed(1)}%` : "--"}</dd>
            </div>
          </dl>
        </div>

        <div className="sidebar-card recent-runs">
          <div className="sidebar-card-head">
            <Database size={16} />
            <span>Recent runs</span>
          </div>
          {runs.slice(0, 3).map((run) => (
            <NavLink key={run.job_id} to={`/results/${run.job_id}`} className="run-link">
              <div className="run-link-main">
                <span>{run.job_id.replace("search_", "")}</span>
                <StatusPill tone={run.status === "failed" ? "danger" : run.status === "succeeded" ? "success" : "info"} label={run.status} />
              </div>
              <button type="button" className="icon-button danger-hover" onClick={(e) => void handleDeleteRun(run.job_id, e)} aria-label={`Delete run ${run.job_id}`}>
                <Trash2 size={13} />
              </button>
            </NavLink>
          ))}
          {!runs.length ? <span className="sidebar-empty-copy">No recent runs yet.</span> : null}
        </div>

        <div className="sidebar-card database-status">
          <div className="sidebar-card-head">
            <Database size={16} />
            <span>Letos SQLite DB</span>
          </div>
          {dbStatus ? (
            <div className="db-status-list">
              <DatabaseStoreCard
                label="Local Index DB"
                metric={`${dbStatus.pasa_local_fts.size_mb} MB`}
                detail={`${dbStatus.pasa_local_fts.paper_count ?? 0} papers`}
                store={dbStatus.pasa_local_fts}
                copiedDbPath={copiedDbPath}
                onCopyPath={handleCopyDbPath}
              />
              <DatabaseStoreCard
                label="Session Hub DB"
                metric={`${dbStatus.session_hub_index.size_mb} MB`}
                detail={`${dbStatus.session_hub_index.run_count ?? 0} runs`}
                store={dbStatus.session_hub_index}
                copiedDbPath={copiedDbPath}
                onCopyPath={handleCopyDbPath}
              />
            </div>
          ) : (
            <span className="sidebar-empty-copy">DB metrics unavailable.</span>
          )}
        </div>
      </aside>

      <main className="workspace">
        <Outlet />
      </main>
    </div>
  );
}

function DatabaseStoreCard({
  label,
  metric,
  detail,
  store,
  copiedDbPath,
  onCopyPath,
}: {
  label: string;
  metric: string;
  detail: string;
  store: DatabaseStoreStatus;
  copiedDbPath: string;
  onCopyPath: (path: string) => Promise<void>;
}) {
  return (
    <div className="db-status-item">
      <div>
        <strong>{label}</strong>
        <span>{metric} / {detail}</span>
      </div>
      <button className="button secondary small" type="button" onClick={() => void onCopyPath(store.path)}>
        <Copy size={12} />
        {copiedDbPath === store.path ? "Copied" : "Copy path"}
      </button>
    </div>
  );
}
