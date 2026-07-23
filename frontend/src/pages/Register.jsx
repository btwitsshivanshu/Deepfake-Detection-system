import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const update = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      await register(form.username, form.email, form.password);
      navigate("/");
    } catch (err) {
      setError(err.response?.data?.detail || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <form onSubmit={handleSubmit} className="card w-full max-w-md">
        <h1 className="mb-1 text-center text-2xl font-bold text-brand">Create account</h1>
        <p className="mb-6 text-center text-sm text-slate-500">Join the DeepFake Detector</p>

        <label className="label">Username</label>
        <input required minLength={3} value={form.username} onChange={update("username")} className="input mb-4" />

        <label className="label">Email</label>
        <input type="email" required value={form.email} onChange={update("email")} className="input mb-4" />

        <label className="label">Password</label>
        <input type="password" required minLength={6} value={form.password} onChange={update("password")} className="input mb-4" />

        {error && <p className="mb-3 text-sm text-red-600">{error}</p>}

        <button type="submit" disabled={loading} className="btn-primary w-full">
          {loading ? "Creating…" : "Register"}
        </button>

        <p className="mt-4 text-center text-sm text-slate-500">
          Already have an account?{" "}
          <Link to="/login" className="font-medium text-brand">Sign in</Link>
        </p>
      </form>
    </div>
  );
}
