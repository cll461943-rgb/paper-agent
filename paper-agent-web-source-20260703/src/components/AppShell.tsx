import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Database,
  FlaskConical,
  GitBranch,
  LayoutDashboard,
  ScrollText,
  Settings,
  Trash2,
} from "lucide-react";
import { getRecentRuns, getSystemStatus } from "../lib/api";
import type { SearchJob, SystemStatus } from "../types/api";
import { StatusPill } from "./Common";
import scholarAgentLogo from "../assets/scholar-agent-icon.png";

const DEFAULT_JOB_ID = "search_20260630_1042";
const SEARCH_RUN_STARTED_EVENT = "scholar-agent:search-run-started";

export function AppShell() {
  const location = useLocation();
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [runs, setRuns] = useState<SearchJob[]>([]);
  const [dbStatus, setDbStatus] = useState<any>(null);

  const handleDeleteRun = async (jobId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (confirm(`Are you sure you want to delete run ${jobId}?`)) {
      try {
        const baseUrl = localStorage.getItem("api_base_url") || "http://127.0.0.1:8000";
        await fetch(`${baseUrl}/api/search/${jobId}`, { method: "DELETE" });
        const nextRuns = await getRecentRuns();
        setRuns(nextRuns);
      } catch (err) {
        console.error("Failed to delete job", err);
      }
    }
  };

  useEffect(() => {
    let cancelled = false;
    let interval: number | undefined;

    async function refreshShellData() {
      const baseUrl = localStorage.getItem("api_base_url") || "http://127.0.0.1:8000";
      const [nextStatus, nextRuns, nextDbStatus] = await Promise.all([
        getSystemStatus(),
        getRecentRuns(),
        fetch(`${baseUrl}/api/database/status`).then(r => r.json()).catch(() => null)
      ]);
      if (cancelled) {
        return;
      }
      setStatus(nextStatus);
      setRuns(nextRuns);
      if (nextDbStatus) {
        setDbStatus(nextDbStatus);
      }
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
            <NavLink key={run.job_id} to={`/results/${run.job_id}`} className="run-link" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
                <span style={{ fontSize: "11px", fontWeight: 600 }}>{run.job_id.replace("search_", "")}</span>
                <StatusPill tone={run.status === "failed" ? "danger" : run.status === "succeeded" ? "success" : "info"} label={run.status} />
              </div>
              <button 
                type="button" 
                onClick={(e) => void handleDeleteRun(run.job_id, e)} 
                style={{ background: "none", border: "none", cursor: "pointer", padding: "4px", color: "#94a3b8", display: "flex", alignItems: "center" }}
                onMouseEnter={(e) => e.currentTarget.style.color = "#ef4444"}
                onMouseLeave={(e) => e.currentTarget.style.color = "#94a3b8"}
                aria-label="Delete run"
              >
                <Trash2 size={13} />
              </button>
            </NavLink>
          ))}
        </div>

        <div className="sidebar-card database-status" style={{ marginTop: "12px" }}>
          <div className="sidebar-card-head">
            <Database size={16} />
            <span>Letos SQLite DB</span>
          </div>
          {dbStatus ? (
            <div style={{ display: "grid", gap: "10px", marginTop: "8px" }}>
              <div style={{ borderBottom: "1px dashed #e2e8f0", paddingBottom: "8px" }}>
                <strong style={{ fontSize: "11px", color: "#334155", display: "block" }}>Local Index DB</strong>
                <span style={{ fontSize: "10px", color: "#64748b" }}>{dbStatus.pasa_local_fts.size_mb} MB ({dbStatus.pasa_local_fts.paper_count} papers)</span>
                <button 
                  className="button secondary small" 
                  type="button" 
                  style={{ width: "100%", fontSize: "9px", padding: "2px 4px", marginTop: "4px", height: "auto" }}
                  onClick={() => {
                    void navigator.clipboard.writeText(dbStatus.pasa_local_fts.path).then(() => {
                      alert("Path copied to clipboard! You can open it in Letos.");
                    });
                  }}
                >
                  Copy Letos Path
                </button>
              </div>
              <div>
                <strong style={{ fontSize: "11px", color: "#334155", display: "block" }}>Session Hub DB</strong>
                <span style={{ fontSize: "10px", color: "#64748b" }}>{dbStatus.session_hub_index.size_mb} MB ({dbStatus.session_hub_index.run_count} runs)</span>
                <button 
                  className="button secondary small" 
                  type="button" 
                  style={{ width: "100%", fontSize: "9px", padding: "2px 4px", marginTop: "4px", height: "auto" }}
                  onClick={() => {
                    void navigator.clipboard.writeText(dbStatus.session_hub_index.path).then(() => {
                      alert("Path copied to clipboard! You can open it in Letos.");
                    });
                  }}
                >
                  Copy Letos Path
                </button>
              </div>
            </div>
          ) : (
            <span style={{ fontSize: "11px", color: "#94a3b8" }}>Loading DB metrics...</span>
          )}
        </div>
      </aside>

      <main className="workspace">
        <Outlet />
      </main>
    </div>
  );
}
