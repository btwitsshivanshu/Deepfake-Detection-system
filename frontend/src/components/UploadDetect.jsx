import { useState } from "react";
import ResultCard from "./ResultCard.jsx";

/**
 * Shared upload + detect widget used by Image and Video pages.
 * Props:
 *   accept   - file input accept string
 *   onDetect - async (formData) => response.data
 *   title    - heading
 */
export default function UploadDetect({ accept, onDetect, title }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [model, setModel] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    setResult(null);
    setError("");
    setPreview(f ? URL.createObjectURL(f) : null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      if (model) fd.append("model", model);
      const data = await onDetect(fd);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Detection failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <form onSubmit={handleSubmit} className="card">
        <h2 className="mb-4 text-xl font-semibold">{title}</h2>

        <label className="label">Select file</label>
        <input type="file" accept={accept} onChange={handleFile} className="input mb-4" />

        <label className="label">Model</label>
        <select value={model} onChange={(e) => setModel(e.target.value)} className="input mb-4">
          <option value="">Default (Xception)</option>
          <option value="resnet50">ResNet50</option>
          <option value="efficientnet_b0">EfficientNet-B0</option>
          <option value="xception">XceptionNet</option>
        </select>

        {preview && accept.includes("image") && (
          <img src={preview} alt="preview" className="mb-4 w-full max-w-xs rounded-lg border" />
        )}
        {preview && accept.includes("video") && (
          <video src={preview} controls className="mb-4 w-full max-w-xs rounded-lg border" />
        )}

        {error && <p className="mb-3 text-sm text-red-600">{error}</p>}

        <button type="submit" disabled={!file || loading} className="btn-primary w-full">
          {loading ? "Analyzing…" : "Detect"}
        </button>
      </form>

      <ResultCard result={result} />
    </div>
  );
}
