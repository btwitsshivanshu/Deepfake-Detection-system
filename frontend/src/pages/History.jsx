import { useEffect, useState } from "react";
import { historyApi } from "../services/api.js";

export default function History() {
  const [items, setItems] = useState([]);
  const [verdict, setVerdict] = useState("");
  const [mediaType, setMediaType] = useState("");
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    historyApi
      .list({ verdict: verdict || undefined, media_type: mediaType || undefined })
      .then((r) => setItems(r.data))
      .finally(() => setLoading(false));
  };

  useEffect(load, [verdict, mediaType]);

  return (
    <div>
      <h1 className="mb-6 text-2xl font-bold">Prediction History</h1>

      <div className="mb-4 flex gap-3">
        <select value={verdict} onChange={(e) => setVerdict(e.target.value)} className="input max-w-[160px]">
          <option value="">All verdicts</option>
          <option value="real">Real</option>
          <option value="fake">Fake</option>
          <option value="uncertain">Uncertain</option>
        </select>
        <select value={mediaType} onChange={(e) => setMediaType(e.target.value)} className="input max-w-[160px]">
          <option value="">All media</option>
          <option value="image">Image</option>
          <option value="video">Video</option>
        </select>
      </div>

      <div className="card overflow-x-auto p-0">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-slate-200 bg-slate-50 text-slate-500">
            <tr>
              <th className="px-4 py-3">ID</th>
              <th className="px-4 py-3">Media</th>
              <th className="px-4 py-3">Verdict</th>
              <th className="px-4 py-3">Confidence</th>
              <th className="px-4 py-3">Frames</th>
              <th className="px-4 py-3">Date</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={6} className="px-4 py-6 text-center text-slate-400">Loading…</td></tr>
            ) : items.length === 0 ? (
              <tr><td colSpan={6} className="px-4 py-6 text-center text-slate-400">No predictions yet.</td></tr>
            ) : (
              items.map((p) => (
                <tr key={p.id} className="border-b border-slate-100">
                  <td className="px-4 py-3">#{p.id}</td>
                  <td className="px-4 py-3 capitalize">{p.media_type}</td>
                  <td className="px-4 py-3">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                      p.verdict === "fake" ? "bg-red-100 text-red-700"
                        : p.verdict === "uncertain" ? "bg-amber-100 text-amber-700"
                        : "bg-green-100 text-green-700"
                    }`}>{p.verdict.toUpperCase()}</span>
                  </td>
                  <td className="px-4 py-3">{Math.round(p.confidence * 100)}%</td>
                  <td className="px-4 py-3">{p.frames_analyzed}</td>
                  <td className="px-4 py-3 text-slate-500">
                    {new Date(p.created_at).toLocaleString()}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
