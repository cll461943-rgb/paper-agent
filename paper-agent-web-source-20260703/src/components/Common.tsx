import { type ReactNode, useState } from "react";
import { ChevronDown, Copy, ExternalLink } from "lucide-react";
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
  const [copiedBibtex, setCopiedBibtex] = useState(false);
  const tags = Object.values(item.paper.metadata).flatMap((value) => (Array.isArray(value) ? value.map(String) : []));

  const handleCopyBibtex = (e: React.MouseEvent) => {
    e.stopPropagation();
    const authorText = item.paper.authors && item.paper.authors.length > 0 ? item.paper.authors.join(" and ") : "Unknown";
    const cleanTitle = item.paper.title.replace(/\$([^$]+)\$/g, "$1");
    const bib = `@article{paper_${item.paper.paper_id.slice(0, 8)},\n  title={${cleanTitle}},\n  author={${authorText}},\n  year={${item.paper.year ?? new Date().getFullYear()}},\n  journal={${item.paper.venue ?? "Academic Search"}}\n}`;
    void navigator.clipboard.writeText(bib).then(() => {
      setCopiedBibtex(true);
      window.setTimeout(() => setCopiedBibtex(false), 1600);
    });
  };

  return (
    <article className={`paper-card ${isExpanded ? "expanded" : ""}`}>
      <div className="paper-rank">#{item.rank}</div>
      <div className="paper-body">
        <button className="paper-summary-button" type="button" onClick={() => setIsExpanded((current) => !current)} aria-expanded={isExpanded}>
          <span className="paper-title-row">
            <h3>{item.paper.title}</h3>
            <ChevronDown className="paper-expand-icon" size={16} />
          </span>
        </button>
        <div className="paper-score-row">
          <strong>{item.final_score.toFixed(3)}</strong>
          <StatusPill tone={item.selection.relevance_level === "high" ? "success" : item.selection.relevance_level === "medium" ? "info" : "muted"} label={item.selection.relevance_level} />
        </div>
        <p>{item.selection.reason}</p>
        <div className="paper-meta">
          <span>{item.paper.year ?? "n/a"}</span>
          <span>{item.paper.venue ?? item.paper.source}</span>
          <span>{item.paper.citation_count ?? 0} citations</span>
        </div>
        
        {isExpanded && (
          <div className="paper-expanded-content">
            {item.paper.authors && item.paper.authors.length > 0 && (
              <div>
                <strong>Authors</strong>
                <p>{item.paper.authors.join(", ")}</p>
              </div>
            )}
            
            {item.paper.abstract && (
              <div>
                <strong>Abstract</strong>
                <p>{item.paper.abstract}</p>
              </div>
            )}
            
            <div className="paper-actions">
              {item.paper.url && (
                <a className="button secondary small" href={item.paper.url} target="_blank" rel="noreferrer">
                  <ExternalLink size={12} />
                  Open Link
                </a>
              )}
              {item.paper.doi && (
                <a className="button secondary small" href={`https://doi.org/${item.paper.doi}`} target="_blank" rel="noreferrer">
                  <ExternalLink size={12} />
                  Open DOI
                </a>
              )}
              <button className="button secondary small" type="button" onClick={handleCopyBibtex}>
                <Copy size={12} />
                {copiedBibtex ? "Copied" : "Copy BibTeX"}
              </button>
            </div>
          </div>
        )}

        <div className="tag-row">
          {tags.slice(0, 5).map((tag, index) => (
            <span key={`${tag}-${index}`}>{tag}</span>
          ))}
        </div>
      </div>
    </article>
  );
}

export function JsonBlock({ data }: { data: unknown }) {
  return <pre className="json-block scrollbar-thin">{JSON.stringify(data, null, 2)}</pre>;
}
