import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

const links = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/detect/image", label: "Image Detection" },
  { to: "/detect/video", label: "Video Detection" },
  { to: "/history", label: "History" },
  { to: "/analytics", label: "Analytics" },
];

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2 font-bold text-brand">
            <span className="text-xl">🛡️</span> AuthentiScan
          </div>
          <nav className="hidden gap-1 md:flex">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.end}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm font-medium ${
                    isActive ? "bg-brand/10 text-brand" : "text-slate-600 hover:bg-slate-100"
                  }`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-3">
            <span className="hidden text-sm text-slate-500 sm:inline">
              {user?.username}
            </span>
            <button onClick={handleLogout} className="btn bg-slate-100 hover:bg-slate-200 text-sm">
              Logout
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8">
        <Outlet />
      </main>
    </div>
  );
}
