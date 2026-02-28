import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8010";

function riskColor(label) {
  if (label === "Normal") return "#2d936c";
  if (label === "Suspicion faible") return "#a88f2a";
  if (label === "Suspicion moderee") return "#d9730d";
  if (label === "Suspicion elevee") return "#cf2f2f";
  return "#64748b";
}

function scorePercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function formatTimestamp(value) {
  const date = new Date((value || 0) * 1000);
  return date.toLocaleString("fr-FR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function waitingMinutes(value) {
  const diffSec = Math.max(0, Math.floor(Date.now() / 1000 - (value || 0)));
  return Math.floor(diffSec / 60);
}

function LoginView({ onLogin, error }) {
  const [username, setUsername] = useState("medecin");
  const [password, setPassword] = useState("med123");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    await onLogin(username, password);
    setLoading(false);
  }

  return (
    <div className="auth-page">
      <form className="auth-card" onSubmit={submit}>
        <h1>MedScan AI</h1>
        <p>Connexion securisee</p>
        <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Nom utilisateur" />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Mot de passe"
        />
        <button type="submit" disabled={loading}>{loading ? "Connexion..." : "Se connecter"}</button>
        <small>Compte demo: medecin / med123, admin / admin123</small>
        {error ? <div className="auth-error">{error}</div> : null}
      </form>
    </div>
  );
}

export default function App() {
  const [token, setToken] = useState(localStorage.getItem("medscan_token") || "");
  const [user, setUser] = useState(null);
  const [authError, setAuthError] = useState("");

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [heatmapEnabled, setHeatmapEnabled] = useState(true);
  const [patientId, setPatientId] = useState("");
  const [examType, setExamType] = useState("");
  const [clinicalNote, setClinicalNote] = useState("");
  const [result, setResult] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState("");
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");

  const [historyPatientFilter, setHistoryPatientFilter] = useState("");
  const [historyRiskFilter, setHistoryRiskFilter] = useState("");
  const [criticalOnly, setCriticalOnly] = useState(false);
  const [activeView, setActiveView] = useState("workspace");
  const [criticalQueue, setCriticalQueue] = useState([]);
  const [queueLoading, setQueueLoading] = useState(false);
  const [queueUpdatedAt, setQueueUpdatedAt] = useState(null);

  useEffect(() => {
    if (!token) return;
    void loadProfile();
    void refreshDashboard();
  }, [token]);

  useEffect(() => {
    if (!token || activeView !== "queue") return;
    void refreshCriticalQueue();
    const id = setInterval(() => {
      void refreshCriticalQueue(false);
    }, 10000);
    return () => clearInterval(id);
  }, [token, activeView]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function authHeaders() {
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  async function apiGet(path) {
    const res = await fetch(`${API_BASE_URL}${path}`, { headers: authHeaders() });
    if (res.status === 401) {
      handleLogout();
      throw new Error("Session expiree. Reconnecte-toi.");
    }
    return res;
  }

  async function onLogin(username, password) {
    setAuthError("");
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Connexion impossible");
      localStorage.setItem("medscan_token", data.access_token);
      setToken(data.access_token);
      setUser({ username: data.username, role: data.role });
    } catch (err) {
      setAuthError(err.message || "Erreur de connexion");
    }
  }

  async function loadProfile() {
    try {
      const response = await apiGet("/auth/me");
      if (!response.ok) return;
      setUser(await response.json());
    } catch {
      // handled by apiGet
    }
  }

  async function handleLogout() {
    try {
      if (token) {
        await fetch(`${API_BASE_URL}/auth/logout`, {
          method: "POST",
          headers: authHeaders()
        });
      }
    } catch {
      // noop
    } finally {
      localStorage.removeItem("medscan_token");
      setToken("");
      setUser(null);
      setHistory([]);
      setStats(null);
      setResult(null);
      setHeatmapUrl("");
      setCriticalQueue([]);
      setQueueUpdatedAt(null);
    }
  }

  async function refreshDashboard() {
    if (!token) return;
    try {
      const query = new URLSearchParams({ limit: "20" });
      if (historyPatientFilter.trim()) query.set("patient_id", historyPatientFilter.trim());
      if (historyRiskFilter) query.set("risk_label", historyRiskFilter);
      if (criticalOnly) query.set("critical_only", "true");

      const [historyRes, statsRes] = await Promise.all([
        apiGet(`/history?${query.toString()}`),
        apiGet("/stats")
      ]);

      if (historyRes.ok) {
        const historyData = await historyRes.json();
        setHistory(Array.isArray(historyData) ? historyData : []);
      }
      if (statsRes.ok) {
        setStats(await statsRes.json());
      }
    } catch (err) {
      setError(err.message || "Erreur API");
    }
  }

  async function refreshCriticalQueue(showLoader = true) {
    if (!token) return;
    try {
      if (showLoader) setQueueLoading(true);
      const response = await apiGet("/queue/critical?limit=50");
      if (response.ok) {
        const data = await response.json();
        setCriticalQueue(Array.isArray(data.items) ? data.items : []);
        setQueueUpdatedAt(data.updated_at || null);
      }
    } catch (err) {
      setError(err.message || "Erreur queue critique");
    } finally {
      if (showLoader) setQueueLoading(false);
    }
  }

  function onSelectFile(event) {
    const selected = event.target.files && event.target.files[0];
    if (!selected) return;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const localUrl = URL.createObjectURL(selected);
    setFile(selected);
    setPreviewUrl(localUrl);
    setResult(null);
    setHeatmapUrl("");
    setError("");
  }

  async function onAnalyze() {
    if (!file) {
      setError("Selectionne une image avant l'analyse.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      if (patientId.trim()) form.append("patient_id", patientId.trim());
      if (examType.trim()) form.append("exam_type", examType.trim());
      if (clinicalNote.trim()) form.append("clinical_note", clinicalNote.trim());

      const response = await fetch(
        `${API_BASE_URL}/analyze?heatmap=${heatmapEnabled ? "true" : "false"}`,
        { method: "POST", body: form, headers: authHeaders() }
      );
      if (response.status === 401) {
        handleLogout();
        throw new Error("Session expiree. Reconnecte-toi.");
      }
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Analyse impossible (${response.status}): ${text}`);
      }

      const data = await response.json();
      setResult(data);
      setHeatmapUrl(data.heatmap_base64_png ? `data:image/png;base64,${data.heatmap_base64_png}` : "");
      await refreshDashboard();
    } catch (err) {
      setError(err.message || "Erreur inconnue");
    } finally {
      setLoading(false);
    }
  }

  async function downloadPdf(analysisId) {
    try {
      const response = await fetch(`${API_BASE_URL}/report/${analysisId}`, {
        headers: authHeaders()
      });
      if (response.status === 401) {
        handleLogout();
        throw new Error("Session expiree. Reconnecte-toi.");
      }
      if (!response.ok) {
        throw new Error("Rapport PDF indisponible.");
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `medscan_report_${analysisId}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message || "Echec telechargement PDF");
    }
  }

  async function resolveCriticalCase(analysisId) {
    try {
      const response = await fetch(`${API_BASE_URL}/queue/critical/${analysisId}/resolve`, {
        method: "POST",
        headers: authHeaders()
      });
      if (response.status === 401) {
        handleLogout();
        throw new Error("Session expiree. Reconnecte-toi.");
      }
      if (!response.ok) throw new Error("Impossible de marquer ce cas comme traite.");
      await Promise.all([refreshCriticalQueue(), refreshDashboard()]);
    } catch (err) {
      setError(err.message || "Echec resolution cas critique");
    }
  }

  const distributionRows = useMemo(() => {
    if (!stats?.risk_distribution) return [];
    return Object.entries(stats.risk_distribution).map(([label, count]) => ({ label, count }));
  }, [stats]);

  if (!token) return <LoginView onLogin={onLogin} error={authError} />;

  return (
    <div className="page">
      <header className="hero">
        <div>
          <h1>MedScan AI</h1>
          <p>Detection precoce assistee par Intelligence Artificielle</p>
        </div>
        <div className="user-chip">
          <span>{user?.username || "user"} ({user?.role || "-"})</span>
          <button type="button" onClick={handleLogout}>Logout</button>
        </div>
      </header>

      <div className="view-switch">
        <button
          type="button"
          className={activeView === "workspace" ? "active" : ""}
          onClick={() => setActiveView("workspace")}
        >
          Workspace
        </button>
        <button
          type="button"
          className={activeView === "queue" ? "active" : ""}
          onClick={() => setActiveView("queue")}
        >
          Queue Critique
        </button>
      </div>

      <main className="layout">
        {activeView === "workspace" ? (
          <>
        <section className="card">
          <div className="card-head">
            <h2>Analyse d'image</h2>
            <span>Upload + preview</span>
          </div>
          <div className="controls">
            <label className="upload-btn">
              Upload image
              <input type="file" accept="image/*" onChange={onSelectFile} />
            </label>
            <button type="button" onClick={onAnalyze} disabled={loading}>{loading ? "Analyse..." : "Analyser"}</button>
            <label className="toggle">
              <input type="checkbox" checked={heatmapEnabled} onChange={(e) => setHeatmapEnabled(e.target.checked)} />
              Heatmap
            </label>
          </div>
          <div className="meta-grid">
            <input placeholder="Patient ID" value={patientId} onChange={(e) => setPatientId(e.target.value)} />
            <input placeholder="Type examen (X-Ray...)" value={examType} onChange={(e) => setExamType(e.target.value)} />
            <input placeholder="Note clinique" value={clinicalNote} onChange={(e) => setClinicalNote(e.target.value)} />
          </div>
          <div className="preview-box">{previewUrl ? <img src={previewUrl} alt="Preview" /> : <p>Aucune image selectionnee.</p>}</div>
        </section>

        <section className="card">
          <div className="card-head">
            <h2>Resultat IA</h2>
            <span>Score + recommandation + top classes</span>
          </div>
          {!result ? (
            <div className="empty">Lance une analyse pour voir le resultat.</div>
          ) : (
            <div className="result">
              <div className="risk-pill" style={{ background: `${riskColor(result.risk_label)}22`, color: riskColor(result.risk_label) }}>
                {result.risk_label}
              </div>
              <div className={`triage-pill triage-${result.triage_level || "routine"}`}>
                Triage: {result.triage_level || "routine"} {result.requires_urgent_review ? "(urgent)" : ""}
              </div>
              <div className="kv"><span>Score de risque</span><strong>{scorePercent(result.risk_score)}</strong></div>
              <div className="kv"><span>Confiance modele</span><strong>{scorePercent(result.confidence)}</strong></div>
              <div className="kv"><span>Classe predite</span><strong>{result.prediction_label || result.prediction_class}</strong></div>
              <p className="recommendation">{result.recommendation}</p>
              <p className="alert-text">{result.alert_message}</p>
              <button type="button" onClick={() => downloadPdf(result.analysis_id)}>Telecharger rapport PDF</button>
              {Array.isArray(result.top_predictions) ? (
                <div className="topk">
                  <h3>Top 3 predictions</h3>
                  {result.top_predictions.map((pred) => (
                    <div className="kv" key={`${pred.class_id}-${pred.label}`}>
                      <span>{pred.label}</span><strong>{scorePercent(pred.confidence)}</strong>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          )}
        </section>

        {heatmapUrl ? (
          <section className="card full">
            <div className="card-head"><h2>Heatmap</h2><span>Explainable AI</span></div>
            <div className="heatmap-box"><img src={heatmapUrl} alt="Heatmap" /></div>
          </section>
        ) : null}

        <section className="card">
          <div className="card-head">
            <h2>Historique</h2>
            <span>Filtres patient et niveau de risque</span>
          </div>
          <div className="meta-grid">
            <input
              placeholder="Filtre Patient ID"
              value={historyPatientFilter}
              onChange={(e) => setHistoryPatientFilter(e.target.value)}
            />
            <select value={historyRiskFilter} onChange={(e) => setHistoryRiskFilter(e.target.value)}>
              <option value="">Tous les risques</option>
              <option value="Normal">Normal</option>
              <option value="Suspicion faible">Suspicion faible</option>
              <option value="Suspicion moderee">Suspicion moderee</option>
              <option value="Suspicion elevee">Suspicion elevee</option>
            </select>
            <label className="toggle">
              <input type="checkbox" checked={criticalOnly} onChange={(e) => setCriticalOnly(e.target.checked)} />
              Critiques seulement
            </label>
            <button type="button" onClick={refreshDashboard}>Appliquer filtres</button>
          </div>
          <div className="history-list">
            {history.length === 0 ? (
              <div className="empty">Aucun historique.</div>
            ) : (
              history.map((item) => (
                <article key={item.analysis_id} className="history-item">
                  <div className="history-top">
                    <span className="risk-dot" style={{ backgroundColor: riskColor(item.risk_label) }} />
                    <strong>{item.risk_label}</strong>
                    <time>{formatTimestamp(item.timestamp)}</time>
                  </div>
                  <small>{item.prediction_label || item.prediction_class} | Confiance {scorePercent(item.confidence)}</small>
                  <small>{item.patient_id ? `Patient ${item.patient_id}` : "Patient N/A"} {item.exam_type ? `| ${item.exam_type}` : ""}</small>
                  <small>{item.alert_message}</small>
                  <button type="button" onClick={() => downloadPdf(item.analysis_id)}>PDF</button>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="card">
          <div className="card-head">
            <h2>Dashboard</h2>
            <span>Stats + alertes recentes</span>
          </div>
          {!stats ? (
            <div className="empty">Stats indisponibles.</div>
          ) : (
            <div className="stats">
              <div className="kv"><span>Total analyses</span><strong>{stats.total ?? 0}</strong></div>
              <div className="kv"><span>Confiance moyenne</span><strong>{scorePercent(stats.avg_confidence)}</strong></div>
              <div className="kv"><span>Cas critiques</span><strong>{stats.critical_count ?? 0}</strong></div>
              <div className="kv"><span>Taux critique</span><strong>{scorePercent(stats.critical_rate)}</strong></div>
              <h3>Distribution des risques</h3>
              {distributionRows.map((row) => {
                const total = stats.total || 1;
                const ratio = (row.count / total) * 100;
                return (
                  <div key={row.label} className="bar-row">
                    <div className="bar-label"><span>{row.label}</span><strong>{row.count}</strong></div>
                    <div className="bar-track"><div className="bar-fill" style={{ width: `${ratio}%`, background: riskColor(row.label) }} /></div>
                  </div>
                );
              })}
              <h3>Alertes recentes (Suspicion elevee)</h3>
              {Array.isArray(stats.recent_alerts) && stats.recent_alerts.length ? (
                stats.recent_alerts.map((alert) => (
                  <div key={alert.analysis_id} className="alert-row">
                    <span>{formatTimestamp(alert.timestamp)}</span>
                    <strong>{alert.patient_id || "Patient N/A"}</strong>
                    <span>{alert.exam_type || "-"}</span>
                    <strong>{scorePercent(alert.confidence)}</strong>
                  </div>
                ))
              ) : (
                <div className="empty">Aucune alerte recente.</div>
              )}
            </div>
          )}
        </section>
          </>
        ) : (
          <section className="card full">
            <div className="card-head">
              <h2>Queue Critique (temps reel)</h2>
              <span>
                {queueUpdatedAt
                  ? `Derniere mise a jour: ${formatTimestamp(queueUpdatedAt)}`
                  : "Aucune mise a jour"}
              </span>
            </div>
            <div className="controls">
              <button type="button" onClick={() => refreshCriticalQueue()}>Rafraichir</button>
              <span className="queue-counter">
                {queueLoading ? "Chargement..." : `${criticalQueue.length} cas ouverts`}
              </span>
            </div>
            <div className="history-list">
              {criticalQueue.length === 0 ? (
                <div className="empty">Aucun cas critique ouvert.</div>
              ) : (
                criticalQueue.map((item) => (
                  <article key={item.analysis_id} className="queue-item">
                    <div className="history-top">
                      <span className="risk-dot" style={{ backgroundColor: "#cf2f2f" }} />
                      <strong>{item.patient_id || "Patient N/A"}</strong>
                      <time>{formatTimestamp(item.timestamp)}</time>
                    </div>
                    <small>{item.prediction_label || item.prediction_class} | Confiance {scorePercent(item.confidence)}</small>
                    <small>{item.exam_type || "Examen N/A"} | Attente: {waitingMinutes(item.timestamp)} min</small>
                    <small>{item.alert_message}</small>
                    <div className="queue-actions">
                      <button type="button" onClick={() => downloadPdf(item.analysis_id)}>PDF</button>
                      <button type="button" onClick={() => resolveCriticalCase(item.analysis_id)}>Marquer traite</button>
                    </div>
                  </article>
                ))
              )}
            </div>
          </section>
        )}
      </main>

      {error ? <div className="error-banner">{error}</div> : null}
    </div>
  );
}
