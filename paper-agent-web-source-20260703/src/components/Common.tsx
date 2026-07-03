import { type ReactNode, useState } from "react";
import type { RankedPaper } from "../types/api";

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}

export function Panel({ title, meta, children, className = "" }: { title?: string; meta?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`panel panel-shadow ${className}`}>
      {title || meta ? (
        <div className="panel-title">
          <h2>{title}</h2>
          {meta ? <span>{meta}</span> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

export function MetricCard({ label, value, detail }: { label: string; value: string | number; detail?: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail ? <small>{detail}</small> : null}
    </div>
  );
}

export function StatusPill({ label, tone = "info" }: { label: string; tone?: "success" | "warning" | "danger" | "info" | "muted" }) {
  return <span className={`status-pill ${tone}`}>{label}</span>;
}

export function ProgressBar({ value }: { value: number }) {
  return (
    <div className="progress-track" aria-label={`Progress ${value}%`}>
      <span style={{ width: `${Math.max(0, Math.min(value, 100))}%` }} />
    </div>
  );
}

export function PaperCard({ item }: { item: RankedPaper }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const tags = Object.values(item.paper.metadata).flatMap((value) => (Array.isArray(value) ? value : []));

  const handleCopyBibtex = (e: React.MouseEvent) => {
    e.stopPropagation();
    const authorText = item.paper.authors && item.paper.authors.length > 0 ? item.paper.authors.join(" and ") : "Unknown";
    const cleanTitle = item.paper.title.replace(/\$([^$]+)\$/g, "$1");
    const bib = `@article{paper_${item.paper.paper_id.slice(0, 8)},\n  title={${cleanTitle}},\n  author={${authorText}},\n  year={${item.paper.year ?? new Date().getFullYear()}},\n  journal={${item.paper.venue ?? "Academic Search"}}\n}`;
    void navigator.clipboard.writeText(bib).then(() => {
      alert("BibTeX copied to clipboard!");
    });
  };

  return (
    <article 
      className={`paper-card ${isExpanded ? "expanded" : ""}`} 
      onClick={() => setIsExpanded(!isExpanded)}
      style={{ cursor: "pointer", transition: "all 0.2s ease" }}
    >
      <div className="paper-rank">#{item.rank}</div>
      <div className="paper-body">
        <div className="paper-title-row">
          <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#1e293b" }}>{item.paper.title}</h3>
          <strong>{item.final_score.toFixed(3)}</strong>
        </div>
        <p style={{ color: "#64748b", margin: "4px 0" }}>{item.selection.reason}</p>
        <div className="paper-meta">
          <span>{item.paper.year ?? "n/a"}</span>
          <span>{item.paper.venue ?? item.paper.source}</span>
          <span>{item.paper.citation_count ?? 0} citations</span>
        </div>
        
        {isExpanded && (
          <div className="paper-expanded-content" style={{ marginTop: "12px", borderTop: "1px solid #e2e8f0", paddingTop: "12px", display: "grid", gap: "8px" }} onClick={(e) => e.stopPropagation()}>
            {item.paper.authors && item.paper.authors.length > 0 && (
              <div>
                <strong style={{ fontSize: "11px", color: "#475569", display: "block" }}>Authors</strong>
                <p style={{ fontSize: "12px", color: "#334155", margin: "2px 0" }}>{item.paper.authors.join(", ")}</p>
              </div>
            )}
            
            {item.paper.abstract && (
              <div>
                <strong style={{ fontSize: "11px", color: "#475569", display: "block" }}>Abstract</strong>
                <p style={{ fontSize: "12px", color: "#475569", lineHeight: "1.4", margin: "2px 0", textAlign: "justify" }}>{item.paper.abstract}</p>
              </div>
            )}
            
            <div className="paper-actions" style={{ display: "flex", gap: "8px", marginTop: "8px", flexWrap: "wrap" }}>
              {item.paper.url && (
                <a className="button secondary small" href={item.paper.url} target="_blank" rel="noreferrer" style={{ padding: "4px 8px", fontSize: "11px" }}>
                  Open Link
                </a>
              )}
              {item.paper.doi && (
                <a className="button secondary small" href={`https://doi.org/${item.paper.doi}`} target="_blank" rel="noreferrer" style={{ padding: "4px 8px", fontSize: "11px" }}>
                  Open DOI
                </a>
              )}
              <button className="button secondary small" type="button" onClick={handleCopyBibtex} style={{ padding: "4px 8px", fontSize: "11px" }}>
                Copy BibTeX
              </button>
            </div>
          </div>
        )}

        <div className="tag-row" style={{ marginTop: "8px" }}>
          {tags.slice(0, 5).map((tag) => (
            <span key={String(tag)}>{String(tag)}</span>
          ))}
        </div>
      </div>
    </article>
  );
}

export function JsonBlock({ data }: { data: unknown }) {
  return <pre className="json-block scrollbar-thin">{JSON.stringify(data, null, 2)}</pre>;
}
