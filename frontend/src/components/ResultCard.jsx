/**
 * Reusable detection result card: shows verdict, confidence bar and
 * the Grad-CAM heatmap overlay.
 */
export default function ResultCard({ result }) {
  if (!result) return null;
  const isFake = result.verdict === "fake";
  const pct = Math.round(result.confidence * 100);

  return (
    <div className="card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold">Result</h3>
        <span
          className={`rounded-full px-3 py-1 text-sm font-semibold ${
            isFake ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
          }`}
        >
          {isFake ? "FAKE" : "REAL"}
        </span>
      </div>

      <div className="mb-4">
        <div className="mb-1 flex justify-between text-sm text-slate-500">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className={`h-full ${isFake ? "bg-red-500" : "bg-green-500"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

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
