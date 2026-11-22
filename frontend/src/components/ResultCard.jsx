/**
 * Reusable detection result card: shows verdict, confidence bar and
 * the Grad-CAM heatmap overlay.
 */
export default function ResultCard({ result }) {
  if (!result) return null;
  const STYLE = {
    fake: { badge: "bg-red-100 text-red-700", bar: "bg-red-500", label: "FAKE" },
    real: { badge: "bg-green-100 text-green-700", bar: "bg-green-500", label: "REAL" },
    uncertain: { badge: "bg-amber-100 text-amber-700", bar: "bg-amber-500", label: "UNCERTAIN" },
  };
  const s = STYLE[result.verdict] || STYLE.uncertain;
  const pct = Math.round(result.confidence * 100);

  return (
    <div className="card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold">Result</h3>
        <span
          className={`rounded-full px-3 py-1 text-sm font-semibold ${s.badge}`}
        >
          {s.label}
        </span>
      </div>

      {result.verdict === "uncertain" && (
        <p className="mb-4 rounded-lg bg-amber-50 p-3 text-sm text-amber-800">
          The model could not decide confidently (fake probability{" "}
          {Math.round(result.fake_probability * 100)}%). This often happens with image
          types it was not trained on — treat the result with caution.
        </p>
      )}

      <div className="mb-4">
        <div className="mb-1 flex justify-between text-sm text-slate-500">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className={`h-full ${s.bar}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {result.screenshot_crop && (
        <p className="mb-4 rounded-lg bg-sky-50 p-3 text-sm text-sky-800">
          Screenshot detected: only the photo inside it was analysed (status bar, text and
          buttons were ignored).
        </p>
      )}

      {result.fake_type && (
        <p className="mb-4 text-sm">
          <span className="text-slate-500">Likely type: </span>
          <span className="font-semibold">{result.fake_type}</span>
        </p>
      )}

      {result.detector_scores && (
        <div className="mb-4 space-y-2">
          <p className="text-sm font-medium text-slate-600">Detector scores (fake probability)</p>
          {[
            ["xception", "GAN-face detector"],
            ["sbi", "Face-swap detector"],
            ["clip", "AI-image detector"],
          ].map(([key, label]) => {
            const v = Math.round((result.detector_scores[key] ?? 0) * 100);
            return (
              <div key={key}>
                <div className="flex justify-between text-xs text-slate-500">
                  <span>{label}</span><span>{v}%</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                  <div className={`h-full ${v >= 50 ? "bg-red-400" : "bg-slate-400"}`} style={{ width: `${v}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div><dt className="text-slate-500">Model</dt><dd className="font-medium">{result.model_used}</dd></div>
        <div><dt className="text-slate-500">Media</dt><dd className="font-medium">{result.media_type}</dd></div>
        <div><dt className="text-slate-500">Frames analyzed</dt><dd className="font-medium">{result.frames_analyzed}</dd></div>
        <div><dt className="text-slate-500">Inference</dt><dd className="font-medium">{result.inference_ms} ms</dd></div>
      </dl>

      {result.heatmap_url && (
        <div className="mt-5">
          <p className="mb-2 text-sm font-medium text-slate-600">
            Grad-CAM — manipulated regions
          </p>
          <img
            src={result.heatmap_url}
            alt="Grad-CAM heatmap"
            className="w-full max-w-xs rounded-lg border border-slate-200"
          />
        </div>
      )}
    </div>
  );
}
