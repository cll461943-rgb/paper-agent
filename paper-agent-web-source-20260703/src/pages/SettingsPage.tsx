import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { RefreshCw, RotateCcw, Save, Wifi } from "lucide-react";
import { getConfigs, getProviders, getSystemStatus, probeBackendConnection } from "../lib/api";
import { getRuntimeConnectionConfig, resetRuntimeConnectionConfig, saveRuntimeConnectionConfig } from "../lib/runtimeConfig";
import type { BackendProbeResult, ProviderStatus, RuntimeConnectionConfig, SystemStatus } from "../types/api";
import { MetricCard, PageHeader, Panel, StatusPill } from "../components/Common";

export function SettingsPage() {
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConnectionConfig>(() => getRuntimeConnectionConfig());
  const [configs, setConfigs] = useState<string[]>([]);
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [probe, setProbe] = useState<BackendProbeResult | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const activeConfig = status?.config ?? runtimeConfig.defaultConfig;
  const activeMode = status?.mode ?? runtimeConfig.defaultSearchMode;

  async function loadRuntimeRegistry() {
    const [nextConfigs, nextProviders, nextStatus] = await Promise.all([getConfigs(), getProviders(), getSystemStatus()]);
    setConfigs(nextConfigs);
    setProviders(nextProviders);
    setStatus(nextStatus);
  }

  useEffect(() => {
    void loadRuntimeRegistry();
  }, []);

  async function handleSave(event: FormEvent) {
    event.preventDefault();
    setIsSaving(true);
    const nextConfig = saveRuntimeConnectionConfig(runtimeConfig);
    setRuntimeConfig(nextConfig);
    await loadRuntimeRegistry();
    setIsSaving(false);
  }

  async function handleReset() {
    const defaults = resetRuntimeConnectionConfig();
    setRuntimeConfig(defaults);
    setProbe(null);
    await loadRuntimeRegistry();
  }

  async function handleTest() {
    setIsTesting(true);
    const nextProbe = await probeBackendConnection();
    setProbe(nextProbe);
    if (nextProbe.ok) {
      await loadRuntimeRegistry();
    }
    setIsTesting(false);
  }

  return (
    <>
      <PageHeader eyebrow="Settings" title="Runtime and provider registry" description="Configure the backend API, auth headers, model gateway, and default pipeline behavior at runtime." />

      <div className="metric-grid">
        <MetricCard label="System" value={status?.status ?? "loading"} detail={status?.message ?? "checking"} />
        <MetricCard label="Mode" value={activeMode} detail={activeConfig} />
        <MetricCard label="API base" value={runtimeConfig.apiBaseUrl || "same-origin"} detail={`${runtimeConfig.requestTimeoutMs}ms timeout`} />
        <MetricCard label="LLM model" value={runtimeConfig.llmModel || "not set"} detail={runtimeConfig.llmProvider || "custom gateway"} />
      </div>

      <div className="settings-grid">
        <Panel title="Connection profile" meta="runtime editable">
          <form className="settings-form" onSubmit={handleSave}>
            <div className="settings-section">
              <div className="settings-section-head">
                <strong>Backend API</strong>
                <span>Subsequent `/api/*` requests use this base URL.</span>
              </div>
              <div className="field-grid">
                <label>
                  API base URL
                  <input
                    type="url"
                    placeholder="http://127.0.0.1:8000"
                    value={runtimeConfig.apiBaseUrl}
                    onChange={(event) => setRuntimeConfig((current) => ({ ...current, apiBaseUrl: event.target.value }))}
                  />
                </label>
                <label>
                  Request timeout (ms)
                  <input
                    type="number"
                    min={1000}
                    step={1000}
                    value={runtimeConfig.requestTimeoutMs}
                    onChange={(event) => setRuntimeConfig((current) => ({ ...current, requestTimeoutMs: Number(event.target.value) }))}
                  />
                </label>
                <label>
                  Auth header
                  <input
                    type="text"
                    value={runtimeConfig.apiAuthHeader}
                    onChange={(event) => setRuntimeConfig((current) => ({ ...current, apiAuthHeader: event.target.value }))}
                  />
                </label>
                <label>
                  Auth scheme
                  <input
                    type="text"
                    placeholder="Bearer"
                    value={runtimeConfig.apiAuthScheme}
                    onChange={(event) => setRuntimeConfig((current) => ({ ...current, apiAuthScheme: event.target.value }))}
                  />
                </label>
              </div>
              <label>
                API token
                <input
                  type="password"
                  placeholder="Optional backend token"
                  value={runtimeConfig.apiKey}
                  onChange={(event) => setRuntimeConfig((current) => ({ ...current, apiKey: event.target.value }))}
                />
              </label>
            </div>

            <div className="settings-section">
              <div className="settings-section-head">
                <strong>LLM override</strong>
                <span>Forwarded as request headers for backends that support runtime model routing.</span>
              </div>
              <div className="field-grid">
                <label>
                  Provider
                  <input
                    type="text"
                    placeholder="openai-compatible"
                    value={runtimeConfig.llmProvider}
                    onChange={(event) => setRuntimeConfig((current) => ({ ...current, llmProvider: event.target.value }))}
                  />
                </label>
                <label>
                  Model
                  <input
                    type="text"
                    placeholder="gpt-4.1-mini / deepseek-chat"
                    value={runtimeConfig.llmModel}
                    onChange={(event) => setRuntimeConfig((current) => ({ ...current, llmModel: event.target.value }))}
                  />
                </label>
              </div>
              <label>
                LLM base URL
                <input
                  type="url"
                  placeholder="https://api.openai.com/v1"
                  value={runtimeConfig.llmBaseUrl}
                  onChange={(event) => setRuntimeConfig((current) => ({ ...current, llmBaseUrl: event.target.value }))}
                />
              </label>
              <label>
                LLM API key
                <input
                  type="password"
                  placeholder="Optional runtime model key"
                  value={runtimeConfig.llmApiKey}
                  onChange={(event) => setRuntimeConfig((current) => ({ ...current, llmApiKey: event.target.value }))}
                />
              </label>
            </div>

            <div className="settings-section">
              <div className="settings-section-head">
                <strong>Workbench defaults</strong>
                <span>Used to prefill the workbench when a new run is created.</span>
              </div>
              <div className="field-grid">
                <label>
                  Default config
                  <select
                    value={runtimeConfig.defaultConfig}
                    onChange={(event) => setRuntimeConfig((current) => ({ ...current, defaultConfig: event.target.value }))}
                  >
                    {(configs.length ? configs : [runtimeConfig.defaultConfig]).map((config) => (
                      <option key={config} value={config}>
                        {config}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Default mode
                  <select
                    value={runtimeConfig.defaultSearchMode}
                    onChange={(event) => setRuntimeConfig((current) => ({ ...current, defaultSearchMode: event.target.value as RuntimeConnectionConfig["defaultSearchMode"] }))}
                  >
                    <option value="research">research</option>
                    <option value="live">live</option>
                    <option value="mock">mock</option>
                  </select>
                </label>
              </div>
            </div>

            <div className="settings-actions">
              <button className="button secondary" type="button" onClick={handleReset}>
                <RotateCcw size={16} />
                Reset defaults
              </button>
              <button className="button secondary" type="button" onClick={() => void handleTest()} disabled={isTesting}>
                <Wifi size={16} />
                {isTesting ? "Testing" : "Test connection"}
              </button>
              <button className="button primary" type="submit" disabled={isSaving}>
                <Save size={16} />
                {isSaving ? "Saving" : "Save profile"}
              </button>
            </div>
          </form>

          {probe ? (
            <div className={`probe-banner ${probe.ok ? "success" : "warning"}`}>
              <strong>{probe.ok ? "Backend reachable" : "Backend not ready"}</strong>
              <span>{probe.message}</span>
              <small>{probe.url}</small>
            </div>
          ) : null}
        </Panel>

        <Panel title="Runtime registry" meta={`${providers.length} sources`}>
          <div className="settings-toolbar">
            <StatusPill tone={probe?.ok ? "success" : "muted"} label={probe?.ok ? "remote api" : "fallback or unchecked"} />
            <button className="button secondary" type="button" onClick={() => void loadRuntimeRegistry()}>
              <RefreshCw size={16} />
              Refresh registry
            </button>
          </div>

          <div className="config-list">
            {configs.map((config) => (
              <div key={config}>
                <strong>{config}</strong>
                <StatusPill
                  tone={config === activeConfig ? "success" : "muted"}
                  label={config === activeConfig ? "active" : "available"}
                />
              </div>
            ))}
          </div>

          <div className="provider-status-list">
            {providers.map((provider) => (
              <div key={provider.name}>
                <div>
                  <strong>{provider.name}</strong>
                  <span>{provider.notes ?? provider.source}</span>
                </div>
                <StatusPill tone={provider.status === "healthy" ? "success" : "warning"} label={`${provider.status} / ${provider.latency_ms}ms`} />
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </>
  );
}
