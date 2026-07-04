import type { RuntimeConnectionConfig, SearchMode } from "../types/api";

const STORAGE_KEY = "paper-agent.runtime-config";
const DEFAULT_TIMEOUT_MS = 20000;
const DEFAULT_PIPELINE_CONFIG = "configs/default.yaml";

function cleanBaseUrl(value: string | undefined): string {
  return (value ?? "").trim().replace(/\/+$/, "");
}

function cleanMode(value: string | undefined): SearchMode {
  if (value === "live" || value === "mock" || value === "research" || value === "local") {
    return value;
  }
  return "research";
}

export function getDefaultRuntimeConnectionConfig(): RuntimeConnectionConfig {
  return {
    apiBaseUrl: cleanBaseUrl(import.meta.env.VITE_API_BASE_URL),
    apiKey: import.meta.env.VITE_API_KEY ?? "",
    apiAuthHeader: import.meta.env.VITE_API_AUTH_HEADER ?? "Authorization",
    apiAuthScheme: import.meta.env.VITE_API_AUTH_SCHEME ?? "Bearer",
    llmProvider: import.meta.env.VITE_LLM_PROVIDER ?? "",
    llmBaseUrl: cleanBaseUrl(import.meta.env.VITE_LLM_BASE_URL),
    llmApiKey: import.meta.env.VITE_LLM_API_KEY ?? "",
    llmModel: import.meta.env.VITE_LLM_MODEL ?? "",
    defaultConfig: import.meta.env.VITE_DEFAULT_PIPELINE_CONFIG ?? DEFAULT_PIPELINE_CONFIG,
    defaultSearchMode: cleanMode(import.meta.env.VITE_DEFAULT_SEARCH_MODE),
    requestTimeoutMs: Number(import.meta.env.VITE_API_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS),
  };
}

function sanitizeConfig(candidate: Partial<RuntimeConnectionConfig> | null | undefined): RuntimeConnectionConfig {
  const defaults = getDefaultRuntimeConnectionConfig();
  const timeout = Number(candidate?.requestTimeoutMs ?? defaults.requestTimeoutMs);

  return {
    apiBaseUrl: cleanBaseUrl(candidate?.apiBaseUrl ?? defaults.apiBaseUrl),
    apiKey: (candidate?.apiKey ?? defaults.apiKey).trim(),
    apiAuthHeader: (candidate?.apiAuthHeader ?? defaults.apiAuthHeader).trim() || defaults.apiAuthHeader,
    apiAuthScheme: (candidate?.apiAuthScheme ?? defaults.apiAuthScheme).trim(),
    llmProvider: (candidate?.llmProvider ?? defaults.llmProvider).trim(),
    llmBaseUrl: cleanBaseUrl(candidate?.llmBaseUrl ?? defaults.llmBaseUrl),
    llmApiKey: (candidate?.llmApiKey ?? defaults.llmApiKey).trim(),
    llmModel: (candidate?.llmModel ?? defaults.llmModel).trim(),
    defaultConfig: (candidate?.defaultConfig ?? defaults.defaultConfig).trim() || defaults.defaultConfig,
    defaultSearchMode: cleanMode(candidate?.defaultSearchMode ?? defaults.defaultSearchMode),
    requestTimeoutMs: Number.isFinite(timeout) ? Math.max(1000, timeout) : defaults.requestTimeoutMs,
  };
}

export function getRuntimeConnectionConfig(): RuntimeConnectionConfig {
  if (typeof window === "undefined") {
    return getDefaultRuntimeConnectionConfig();
  }

  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return getDefaultRuntimeConnectionConfig();
  }

  try {
    return sanitizeConfig(JSON.parse(raw) as Partial<RuntimeConnectionConfig>);
  } catch {
    return getDefaultRuntimeConnectionConfig();
  }
}

export function saveRuntimeConnectionConfig(config: RuntimeConnectionConfig): RuntimeConnectionConfig {
  const nextConfig = sanitizeConfig(config);
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextConfig));
  }
  return nextConfig;
}

export function resetRuntimeConnectionConfig(): RuntimeConnectionConfig {
  const defaults = getDefaultRuntimeConnectionConfig();
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(STORAGE_KEY);
  }
  return defaults;
}
