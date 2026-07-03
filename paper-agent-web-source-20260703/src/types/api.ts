export type SearchMode = "live" | "mock" | "research";
export type JobStatus = "queued" | "running" | "succeeded" | "failed";
export type RelevanceLevel = "high" | "medium" | "low" | "irrelevant";

export interface RuntimeConnectionConfig {
  apiBaseUrl: string;
  apiKey: string;
  apiAuthHeader: string;
  apiAuthScheme: string;
  llmProvider: string;
  llmBaseUrl: string;
  llmApiKey: string;
  llmModel: string;
  defaultConfig: string;
  defaultSearchMode: SearchMode;
  requestTimeoutMs: number;
}

export interface BackendProbeResult {
  ok: boolean;
  url: string;
  status: number | null;
  contentType?: string;
  message: string;
}

export interface SearchRequest {
  query: string;
  mode: SearchMode;
  config: string;
  providers: string[];
  retrieval_only: boolean;
  use_local_index: boolean;
}

export interface SearchJob {
  job_id: string;
  status: JobStatus;
  stage: string;
  progress?: number;
  elapsed_seconds?: number;
  error?: string;
}

export interface QueryPlan {
  original_query: string;
  language: string;
  query_type: string;
  research_topic?: string | null;
  task?: string | null;
  methods: string[];
  datasets: string[];
  entities: string[];
  time_range?: Record<string, number | string> | null;
  venues: string[];
  must_have_constraints: string[];
  nice_to_have_constraints: string[];
  exclude_terms: string[];
  expected_output: string;
  uncertainty: string[];
}

export interface EvidenceItem {
  field: string;
  text: string;
}

export interface Paper {
  paper_id: string;
  title: string;
  abstract?: string | null;
  year?: number | null;
  venue?: string | null;
  authors: string[];
  doi?: string | null;
  arxiv_id?: string | null;
  url?: string | null;
  citation_count?: number | null;
  source: string;
  retrieval_path: string[];
  references: string[];
  citations: string[];
  metadata: Record<string, unknown>;
}

export interface SelectionResult {
  paper_id: string;
  relevance_level: RelevanceLevel;
  matched_constraints: string[];
  missing_constraints: string[];
  evidence: EvidenceItem[];
  reason: string;
  confidence: number;
  is_validated: boolean;
  validation_notes: string[];
  relevance_score?: number | null;
  constraint_score?: number | null;
  evidence_score?: number | null;
  uncertainty: string[];
}

export interface RankedPaper {
  paper: Paper;
  selection: SelectionResult;
  final_score: number;
  subscores: Record<string, number>;
  rank: number;
}

export interface ComponentMetric {
  component: string;
  elapsed_seconds: number;
  items_delta: number;
  api_calls_delta: number;
  api_calls_total: number;
  llm_calls_delta: number;
  llm_calls_total: number;
  token_estimate_delta: number;
  token_estimate_total: number;
  cache_hits_delta: number;
  cache_hits_total: number;
  errors_delta: number;
  errors_total: number;
  search_queries_delta: number;
  search_queries_total: number;
  retrieval_rounds_delta: number;
  retrieval_rounds_total: number;
}

export interface RunMetrics {
  search_queries_used: number;
  retrieval_rounds_used: number;
  candidate_pool_size: number;
  final_papers: number;
  evidence_summaries: number;
  llm_calls_used: number;
  llm_elapsed_seconds: number;
  api_calls_used: number;
  token_estimate: number;
  elapsed_seconds: number;
  cache_hits: number;
  errors: string[];
  component_metrics: ComponentMetric[];
}

export interface BenchmarkMetrics {
  hits: number;
  gold_total: number;
  output_total: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface EvalBudgetSummary {
  elapsed_seconds: number;
  api_calls: number;
  llm_calls: number;
  token_estimate: number;
  candidate_pool_size: number;
  final_papers: number;
}

export interface SearchProcessRound {
  round_index: number;
  search_goal: string;
  queries: string[];
  candidates_found: number;
  review_conclusion: string;
}

export interface WorkflowResult {
  original_query: string;
  query_plan: QueryPlan;
  search_process: SearchProcessRound[];
  highly_relevant_papers: RankedPaper[];
  partially_relevant_papers: RankedPaper[];
  supporting_papers: RankedPaper[];
  method_clusters: Array<Record<string, unknown>>;
  timeline: Array<Record<string, unknown>>;
  citation_graph: Record<string, unknown>;
  recommendation_reasoning: Array<Record<string, unknown>>;
  agent_self_report: Record<string, unknown>;
  run_metrics: RunMetrics;
  dynamic_k_chosen?: number | null;
  expected_f1_curve?: Record<number, number> | null;
  g_hat?: number | null;
  g_hat_scope?: number | null;
  g_hat_pool?: number | null;
  g_hat_visible?: number | null;
  low_confidence_uniform?: boolean;
  p_floor?: number | null;
  tie_break_reason?: string;
  second_pass_triggered?: boolean;
  benchmark_metrics?: BenchmarkMetrics | null;
}

export interface ProviderStatus {
  name: string;
  enabled: boolean;
  available: boolean;
  source: string;
  latency_ms: number;
  status: "healthy" | "degraded" | "offline";
  notes?: string;
}

export interface SystemStatus {
  status: "ok" | "degraded" | "error";
  message: string;
  mode: SearchMode;
  config: string;
  cache_hit_rate: number;
  local_index_ready: boolean;
  vector_index_ready: boolean;
  provider_count: {
    healthy: number;
    total: number;
  };
}

export interface DatabaseStoreStatus {
  path: string;
  exists?: boolean;
  size_mb: number;
  paper_count?: number;
  run_count?: number;
  table_count?: number;
  fts_tables?: string[];
}

export interface DatabaseStatus {
  pasa_local_fts: DatabaseStoreStatus;
  session_hub_index: DatabaseStoreStatus;
}

export interface EvalCaseRequest {
  dataset: string;
  case_index: number;
  config: string;
  mode: string;
  trace_gold: boolean;
}

export interface DiagnosticRow {
  paper_id: string;
  title: string;
  source: string;
  year?: number | null;
  citation_count?: number | null;
  relevance?: number | string | null;
  stage: string;
  score?: number | null;
  pass: boolean;
  reason: string;
}

export interface EvalCaseResult {
  job_id: string;
  dataset: string;
  case_index: number;
  query: string;
  gold: string[];
  eval_metrics?: BenchmarkMetrics | null;
  budget?: EvalBudgetSummary | null;
  progress: Array<{ stage: string; status: JobStatus; elapsed_seconds: number }>;
  candidate_pool: DiagnosticRow[];
  selection_candidates: DiagnosticRow[];
  ranked_papers: DiagnosticRow[];
  final_output: DiagnosticRow[];
  warnings: string[];
}

export interface GraphNode {
  id: string;
  type: "paper" | "method" | "dataset" | "author" | "topic";
  label: string;
  shortLabel?: string;
  x: number;
  y: number;
  size: number;
  clusterId?: string;
  year?: number;
  highlighted?: boolean;
  meta?: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: "reference" | "similar" | "citation-path" | "association";
  weight: number;
}

export interface GraphCluster {
  id: string;
  label: string;
  color: string;
  center: { x: number; y: number };
  radiusX: number;
  radiusY: number;
  nodeIds: string[];
}

export interface GraphResponse {
  job_id: string;
  query: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  clusters: GraphCluster[];
  selected_node_id?: string;
  stats: {
    papers: number;
    authors: number;
    methods: number;
    datasets: number;
    edges: number;
    clusters: number;
  };
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: "INFO" | "WARN" | "ERROR" | "DEBUG";
  stage: string;
  message: string;
  raw?: Record<string, unknown>;
}

export interface ResultsResponse {
  job_id: string;
  status: string;
  result: WorkflowResult | null;
  artifacts: {
    candidate_pool_size: number;
    selection_candidates_size: number;
    ranked_papers_size: number;
    final_output_size: number;
  };
}

export interface StageArtifactResponse {
  stage: string;
  job_id: string;
  data: Record<string, unknown>;
}
