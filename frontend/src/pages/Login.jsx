import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      await login(email, password);
      navigate("/");
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <form onSubmit={handleSubmit} className="card w-full max-w-md">
        <h1 className="mb-1 text-center text-2xl font-bold text-brand">🛡️ DeepFake Detector</h1>
        <p className="mb-6 text-center text-sm text-slate-500">Sign in to your account</p>

        <label className="label">Email</label>
        <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} className="input mb-4" />

        <label className="label">Password</label>
        <input type="password" required value={password} onChange={(e) => setPassword(e.target.value)} className="input mb-4" />

        {error && <p className="mb-3 text-sm text-red-600">{error}</p>}

        <button type="submit" disabled={loading} className="btn-primary w-full">
          {loading ? "Signing in…" : "Sign In"}
        </button>

        <p className="mt-4 text-center text-sm text-slate-500">
          No account?{" "}
          <Link to="/register" className="font-medium text-brand">Register</Link>
        </p>
      </form>
    </div>
  );
}
