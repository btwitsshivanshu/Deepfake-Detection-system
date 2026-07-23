import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { analyticsApi } from "../services/api.js";
import { useAuth } from "../context/AuthContext.jsx";

function Stat({ label, value, accent }) {
  return (
    <div className="card">
      <p className="text-sm text-slate-500">{label}</p>
      <p className={`mt-1 text-3xl font-bold ${accent || "text-slate-800"}`}>{value}</p>
    </div>
  );
}

export default function Dashboard() {
  const { user } = useAuth();
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    analyticsApi.summary().then((r) => setSummary(r.data)).catch(() => {});
  }, []);

  return (
    <div>
      <h1 className="mb-6 text-2xl font-bold">Welcome back, {user?.username} 👋</h1>

      <div className="mb-8 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Total Predictions" value={summary?.total_predictions ?? "—"} />
        <Stat label="Real" value={summary?.real_count ?? "—"} accent="text-green-600" />
        <Stat label="Fake" value={summary?.fake_count ?? "—"} accent="text-red-600" />
        <Stat
          label="Avg Confidence"
          value={summary ? `${Math.round(summary.avg_confidence * 100)}%` : "—"}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Link to="/detect/image" className="card transition hover:shadow-md">
          <span className="text-2xl">🖼️</span>
          <h3 className="mt-2 font-semibold">Detect Image</h3>
          <p className="text-sm text-slate-500">Upload an image to check authenticity.</p>
        </Link>
        <Link to="/detect/video" className="card transition hover:shadow-md">
          <span className="text-2xl">🎬</span>
          <h3 className="mt-2 font-semibold">Detect Video</h3>
          <p className="text-sm text-slate-500">Analyze a video frame by frame.</p>
        </Link>
        <Link to="/analytics" className="card transition hover:shadow-md">
          <span className="text-2xl">📊</span>
          <h3 className="mt-2 font-semibold">View Analytics</h3>
          <p className="text-sm text-slate-500">Explore your detection insights.</p>
        </Link>
      </div>
    </div>
  );
}
