import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { analyticsApi } from "../services/api.js";

const COLORS = ["#22c55e", "#ef4444"];

export default function Analytics() {
  const [data, setData] = useState(null);

  useEffect(() => {
    analyticsApi.summary().then((r) => setData(r.data)).catch(() => {});
  }, []);

  if (!data) return <p className="text-slate-400">Loading analytics…</p>;

  const verdictData = [
    { name: "Real", value: data.real_count },
    { name: "Fake", value: data.fake_count },
  ];
  const mediaData = [
    { name: "Image", value: data.image_count },
    { name: "Video", value: data.video_count },
  ];
  const modelData = Object.entries(data.model_usage).map(([name, value]) => ({ name, value }));
  const dailyData = Object.entries(data.daily_counts).map(([date, count]) => ({ date, count }));

  return (
    <div>
      <h1 className="mb-6 text-2xl font-bold">Analytics</h1>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card">
          <h3 className="mb-4 font-semibold">Real vs Fake</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={verdictData} dataKey="value" nameKey="name" outerRadius={90} label>
                {verdictData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Pie>
              <Tooltip /><Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="mb-4 font-semibold">Media Type</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={mediaData}>
              <XAxis dataKey="name" /><YAxis allowDecimals={false} />
              <Tooltip /><Bar dataKey="value" fill="#6366f1" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="mb-4 font-semibold">Model Usage</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={modelData}>
              <XAxis dataKey="name" /><YAxis allowDecimals={false} />
              <Tooltip /><Bar dataKey="value" fill="#0ea5e9" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="mb-4 font-semibold">Daily Activity</h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={dailyData}>
              <XAxis dataKey="date" /><YAxis allowDecimals={false} />
              <Tooltip /><Line type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
