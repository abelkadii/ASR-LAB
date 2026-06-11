import { useEffect, useState } from "react";
import { ApiProvider, Workspace } from "./api";
import { ResearchHeader } from "./components/ResearchHeader";
import { ResearchWorkspace } from "./components/ResearchWorkspace";

export type ThemeMode = "dark" | "light";

export default function App() {
  const [activeWorkspace, setActiveWorkspace] = useState<Workspace>("home");
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const saved = window.localStorage.getItem("speechrecon-theme");
    return saved === "light" ? "light" : "dark";
  });

  useEffect(() => {
    window.localStorage.setItem("speechrecon-theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.classList.toggle("theme-light", theme === "light");
    document.documentElement.classList.toggle("theme-dark", theme === "dark");
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  return (
    <ApiProvider>
      <div className="app-shell size-full flex flex-col bg-[#0a0a0b] text-gray-100">
        <ResearchHeader activeWorkspace={activeWorkspace} onWorkspaceChange={setActiveWorkspace} theme={theme} onThemeChange={setTheme} />
        <ResearchWorkspace workspace={activeWorkspace} onWorkspaceChange={setActiveWorkspace} />
      </div>
    </ApiProvider>
  );
}
