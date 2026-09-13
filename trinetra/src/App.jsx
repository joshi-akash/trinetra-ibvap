import { useState, useEffect } from "react";

/* ---------- MOCK DATA (swap for real API later) ---------- */
const initialAlerts = [
  { id: 1, camera: "Cam 1", type: "Person Detected", confidence: 92, rule: "Perimeter Breach", time: "10:30 AM", status: "active" },
  { id: 2, camera: "Cam 2", type: "Vehicle Detected", confidence: 87, rule: "Unauthorized Vehicle", time: "10:35 AM", status: "active" },
  { id: 3, camera: "Cam 3", type: "Suspicious Activity", confidence: 78, rule: "Loitering", time: "10:41 AM", status: "passive" },
];

const cameras = [
  { id: 1, name: "Cam 1", location: "North Gate", trust: 98, status: "online" },
  { id: 2, name: "Cam 2", location: "East Fence", trust: 95, status: "online" },
  { id: 3, name: "Cam 3", location: "South Watch", trust: 88, status: "degraded" },
  { id: 4, name: "Cam 4", location: "West Ridge", trust: 99, status: "online" },
];

/* ---------- SHARED UI PIECES ---------- */
function StatusPill({ status }) {
  const styles = { online: "bg-green-600", degraded: "bg-yellow-600", offline: "bg-red-600" };
  return <span className={`px-2 py-0.5 rounded-full text-xs ${styles[status] || "bg-gray-600"}`}>{status}</span>;
}

function AlertBadge({ type }) {
  const colors = { "Person Detected": "bg-yellow-600", "Vehicle Detected": "bg-blue-600", "Suspicious Activity": "bg-red-600" };
  return <span className={`px-2 py-1 rounded text-xs font-medium ${colors[type] || "bg-gray-600"}`}>{type}</span>;
}

function TrustBadge({ trust }) {
  const color = trust >= 95 ? "text-green-400" : trust >= 85 ? "text-yellow-400" : "text-red-400";
  return <span className={`font-mono ${color}`}>{trust}%</span>;
}

/* ---------- LIVE VIEW ---------- */
function LiveView() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Live Camera View</h2>
      <div className="grid grid-cols-2 gap-4">
        {cameras.map((cam) => (
          <div key={cam.id} className="relative bg-gray-800 h-48 rounded-lg border border-gray-700 shadow-lg overflow-hidden">
            <div className="absolute top-2 left-2 flex items-center gap-1 text-xs bg-black/60 px-2 py-1 rounded">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span> LIVE
            </div>
            <div className="absolute top-2 right-2"><StatusPill status={cam.status} /></div>
            <div className="flex flex-col items-center justify-center h-full text-gray-400">
              <span className="text-3xl mb-1">📹</span>
              <span className="text-sm">{cam.name} — {cam.location}</span>
            </div>
            {/* simulated bounding box */}
            <div className="absolute border-2 border-green-400 rounded" style={{ top: "35%", left: "40%", width: "20%", height: "30%" }}>
              <span className="absolute -top-5 left-0 text-[10px] bg-green-400 text-black px-1 rounded">Person 91%</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ---------- ALERT FEED ---------- */
function AlertFeed({ alerts, onSelect, flash }) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Alert Feed</h2>
      <div className="space-y-3">
        {alerts.map((a) => (
          <div
            key={a.id}
            onClick={() => onSelect(a)}
            className={`flex items-center justify-between bg-gray-800 border border-gray-700 rounded-lg p-4 cursor-pointer hover:border-blue-500 transition ${
              flash === a.id ? "ring-2 ring-red-500 animate-pulse" : ""
            }`}
          >
            <div className="flex items-center gap-4">
              <div className="w-16 h-12 bg-gray-700 rounded flex items-center justify-center text-xl">🖼️</div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <AlertBadge type={a.type} />
                  <span className="text-xs text-gray-400">{a.camera} · {a.time}</span>
                </div>
                <div className="text-sm text-gray-300">Rule: {a.rule} · Confidence: {a.confidence}%</div>
              </div>
            </div>
            <div className="flex gap-2">
              <button className="text-xs bg-gray-700 px-3 py-1 rounded hover:bg-gray-600">View Clip</button>
              <button className="text-xs bg-blue-600 px-3 py-1 rounded hover:bg-blue-500">Export</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ---------- ALERT DETAIL MODAL ---------- */
function AlertDetail({ alert, onClose }) {
  if (!alert) return null;
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 w-96">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-bold">Alert Detail</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-white">✕</button>
        </div>
        <div className="w-full h-32 bg-gray-800 rounded mb-4 flex items-center justify-center text-3xl">🎞️</div>
        <div className="text-sm space-y-1 text-gray-300">
          <div><span className="text-gray-500">Camera:</span> {alert.camera}</div>
          <div><span className="text-gray-500">Type:</span> <AlertBadge type={alert.type} /></div>
          <div><span className="text-gray-500">Rule Fired:</span> {alert.rule}</div>
          <div><span className="text-gray-500">Confidence:</span> {alert.confidence}%</div>
          <div><span className="text-gray-500">Time:</span> {alert.time}</div>
          <div><span className="text-gray-500">Status:</span> {alert.status}</div>
        </div>
      </div>
    </div>
  );
}

/* ---------- GEO-FENCE EDITOR ---------- */
function GeoFenceEditor() {
  const [points, setPoints] = useState([]);
  const [saved, setSaved] = useState(false);

  const handleClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setPoints([...points, { x, y }]);
    setSaved(false);
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Geo-Fence Editor</h2>
      <p className="text-sm text-gray-400 mb-3">Click on the feed to place points (minimum 3 to save a zone).</p>
      <div onClick={handleClick} className="relative bg-gray-800 border border-gray-700 rounded-lg h-72 w-full cursor-crosshair overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center text-gray-500">Cam 1 — North Gate</div>
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {points.length >= 3 && (
            <polygon points={points.map((p) => `${p.x},${p.y}`).join(" ")} fill="rgba(59,130,246,0.3)" stroke="#3b82f6" strokeWidth="2" />
          )}
          {points.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r="4" fill="#3b82f6" />
          ))}
        </svg>
      </div>
      <div className="mt-3 flex gap-2">
        <button onClick={() => { setPoints([]); setSaved(false); }} className="text-xs bg-gray-700 px-3 py-1.5 rounded hover:bg-gray-600">Reset</button>
        <button
          disabled={points.length < 3}
          onClick={() => setSaved(true)}
          className={`text-xs px-3 py-1.5 rounded ${points.length < 3 ? "bg-gray-700 text-gray-500 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-500"}`}
        >
          Save Zone
        </button>
        {saved && <span className="text-xs text-green-400 self-center">✓ Zone saved ({points.length} vertices)</span>}
      </div>
    </div>
  );
}

/* ---------- FORENSIC SEARCH ---------- */
function ForensicSearch({ alerts }) {
  const [camera, setCamera] = useState("");
  const [entityType, setEntityType] = useState("");
  const [results, setResults] = useState(null);

  const runSearch = () => {
    let filtered = alerts;
    if (camera) filtered = filtered.filter((a) => a.camera === camera);
    if (entityType) filtered = filtered.filter((a) => a.type === entityType);
    setResults(filtered);
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Forensic Search</h2>
      <div className="flex flex-wrap gap-3 mb-4">
        <select value={camera} onChange={(e) => setCamera(e.target.value)} className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
          <option value="">All Cameras</option>
          {cameras.map((c) => <option key={c.id} value={c.name}>{c.name}</option>)}
        </select>
        <select value={entityType} onChange={(e) => setEntityType(e.target.value)} className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
          <option value="">All Types</option>
          <option>Person Detected</option>
          <option>Vehicle Detected</option>
          <option>Suspicious Activity</option>
        </select>
        <input type="date" className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm" />
        <button onClick={runSearch} className="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded text-sm">Search</button>
      </div>
      {results && (
        <div className="space-y-2">
          {results.length === 0 && <p className="text-gray-500 text-sm">No matching records.</p>}
          {results.map((r) => (
            <div key={r.id} className="bg-gray-800 border border-gray-700 rounded p-3 flex justify-between text-sm">
              <span>{r.camera} — <AlertBadge type={r.type} /></span>
              <span className="text-gray-400">{r.time}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------- CAMERA MANAGEMENT ---------- */
function CameraManagement() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Camera Management</h2>
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-gray-700 text-gray-400">
            <th className="p-2">Camera</th><th className="p-2">Location</th><th className="p-2">Trust Score</th><th className="p-2">Status</th><th className="p-2"></th>
          </tr>
        </thead>
        <tbody>
          {cameras.map((c) => (
            <tr key={c.id} className="border-b border-gray-800">
              <td className="p-2">{c.name}</td>
              <td className="p-2">{c.location}</td>
              <td className="p-2"><TrustBadge trust={c.trust} /></td>
              <td className="p-2"><StatusPill status={c.status} /></td>
              <td className="p-2"><button className="text-xs bg-gray-700 px-3 py-1 rounded hover:bg-gray-600">Calibrate</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- REVIEW QUEUE ---------- */
function ReviewQueue({ alerts, onFalseFlag }) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Review Queue</h2>
      <div className="space-y-2">
        {alerts.map((a) => (
          <div key={a.id} className="flex justify-between items-center bg-gray-800 border border-gray-700 rounded p-3 text-sm">
            <span><AlertBadge type={a.type} /> {a.camera} · {a.time}</span>
            <button onClick={() => onFalseFlag(a.id)} className="text-xs bg-red-700 hover:bg-red-600 px-3 py-1 rounded">Mark False Flag</button>
          </div>
        ))}
        {alerts.length === 0 && <p className="text-gray-500 text-sm">Queue is empty.</p>}
      </div>
    </div>
  );
}

/* ---------- EXPORT LOG ---------- */
function ExportLog({ history }) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Export / HQ Log</h2>
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-gray-700 text-gray-400">
            <th className="p-2">Alert ID</th><th className="p-2">Exported By</th><th className="p-2">Time</th><th className="p-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {history.map((h, i) => (
            <tr key={i} className="border-b border-gray-800">
              <td className="p-2">#{h.alertId}</td><td className="p-2">{h.exportedBy}</td><td className="p-2">{h.time}</td>
              <td className="p-2"><StatusPill status={h.status === "sent" ? "online" : "degraded"} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- SETTINGS ---------- */
function Settings() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Settings</h2>
      <div className="space-y-3 max-w-sm text-sm">
        <div><label className="text-gray-400 block mb-1">Role</label>
          <select className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"><option>Commander</option><option>Operator</option></select>
        </div>
        <div><label className="text-gray-400 block mb-1">Alert Sound</label>
          <select className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"><option>Siren</option><option>Silent</option></select>
        </div>
      </div>
    </div>
  );
}

/* ---------- MAIN APP ---------- */
export default function App() {
  const [page, setPage] = useState("live");
  const [alerts, setAlerts] = useState(initialAlerts);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [flash, setFlash] = useState(null);
  const [online, setOnline] = useState(true);
  const [exportHistory, setExportHistory] = useState([
    { alertId: 1, exportedBy: "Cdr. Sharma", time: "10:32 AM", status: "sent" },
  ]);

  // simulate a new "live" alert every 15s — remove once real WebSocket exists
  useEffect(() => {
    const interval = setInterval(() => {
      const id = Date.now();
      const newAlert = { id, camera: "Cam 4", type: "Person Detected", confidence: 90, rule: "Perimeter Breach", time: new Date().toLocaleTimeString(), status: "active" };
      setAlerts((prev) => [newAlert, ...prev]);
      setFlash(id);
      setTimeout(() => setFlash(null), 2000);
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleFalseFlag = (id) => setAlerts((prev) => prev.filter((a) => a.id !== id));

  const navItems = [
    ["live", "Live View"], ["alerts", "Alert Feed"], ["geofence", "Geo-Fence"],
    ["search", "Forensic Search"], ["cameras", "Cameras"], ["review", "Review Queue"],
    ["export", "Export Log"], ["settings", "Settings"],
  ];

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      <div className="w-52 bg-gray-950 p-4 space-y-1">
        <h1 className="text-xl font-bold mb-6">TRINETRA</h1>
        {navItems.map(([key, label]) => (
          <button key={key} onClick={() => setPage(key)}
            className={`block w-full text-left p-2 rounded text-sm ${page === key ? "bg-blue-600" : "hover:bg-gray-800"}`}>
            {label}
          </button>
        ))}
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        <div className="flex justify-between items-center mb-6 pb-4 border-b border-gray-700">
          <span className="text-sm text-gray-400">Border Out Post — Sector 7</span>
          <button onClick={() => setOnline(!online)} className="text-sm">
            {online ? <span className="text-green-400">● System Online</span> : <span className="text-red-400">● WAN Disconnected (Local Ops Active)</span>}
          </button>
        </div>

        {page === "live" && <LiveView />}
        {page === "alerts" && <AlertFeed alerts={alerts} onSelect={setSelectedAlert} flash={flash} />}
        {page === "geofence" && <GeoFenceEditor />}
        {page === "search" && <ForensicSearch alerts={alerts} />}
        {page === "cameras" && <CameraManagement />}
        {page === "review" && <ReviewQueue alerts={alerts.filter((a) => a.status === "passive")} onFalseFlag={handleFalseFlag} />}
        {page === "export" && <ExportLog history={exportHistory} />}
        {page === "settings" && <Settings />}
      </div>

      <AlertDetail alert={selectedAlert} onClose={() => setSelectedAlert(null)} />
    </div>
  );
}