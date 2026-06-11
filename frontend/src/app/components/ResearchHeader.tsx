import { Activity, Database, FlaskConical, Mic2, Moon, Sun } from 'lucide-react';
import { useApi, Workspace } from '../api';
import { ThemeMode } from '../App';

interface ResearchHeaderProps {
  activeWorkspace: string;
  onWorkspaceChange: (workspace: Workspace) => void;
  theme: ThemeMode;
  onThemeChange: (theme: ThemeMode) => void;
}

export function ResearchHeader({ activeWorkspace, onWorkspaceChange, theme, onThemeChange }: ResearchHeaderProps) {
  const { status } = useApi();
  const workspaces = [
    { id: 'home', label: 'Home' },
    { id: 'workbench', label: 'Workbench' },
    { id: 'models', label: 'Models' },
    { id: 'dataset', label: 'Datasets' },
    { id: 'label', label: 'Label' },
    { id: 'analysis', label: 'Analysis' },
  ] as const;
  const { models, datasets, selectedModel, selectedDataset, selectedSample } = useApi();
  const modelName = models.find((model) => model.id === selectedModel)?.name || selectedModel || 'No model';
  const datasetName = datasets.find((dataset) => dataset.id === selectedDataset)?.name || selectedDataset || 'No dataset';
  const sessionShell =
    theme === 'light'
      ? 'border-slate-300 bg-white text-slate-700 shadow-sm'
      : 'border-gray-800 bg-[#0f0f11] text-gray-300';
  const sessionBadge =
    theme === 'light'
      ? 'bg-slate-100 text-slate-700'
      : 'bg-gray-900/90 text-gray-300';
  const sessionMuted = theme === 'light' ? 'text-slate-500' : 'text-gray-500';
  const sessionTitle = theme === 'light' ? 'text-slate-700' : 'text-gray-400';
  const toggleShell = theme === 'light' ? 'border-slate-300 bg-white shadow-sm' : 'border-gray-800 bg-[#0f0f11]';

  return (
    <header className="h-12 bg-[#121214] border-b border-gray-800 flex items-center px-4 gap-6">
      <div className="flex items-center gap-2.5">
        <Activity className="w-5 h-5 text-emerald-500" strokeWidth={2.5} />
        <div className="flex flex-col -space-y-0.5">
          <span className="text-sm font-semibold tracking-tight">DARIJA ASR</span>
          <span className="text-[10px] text-gray-500 uppercase tracking-wider">Research Workbench</span>
        </div>
      </div>

      <div className="h-6 w-px bg-gray-800" />

      <nav className="flex items-center gap-1">
        {workspaces.map((ws) => (
          <button
            key={ws.id}
            onClick={() => onWorkspaceChange(ws.id)}
            className={`px-3 h-7 text-xs font-medium rounded transition-colors ${
              activeWorkspace === ws.id
                ? 'bg-emerald-500/20 text-emerald-400'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
            }`}
          >
            {ws.label}
          </button>
        ))}
      </nav>

      <div className="ml-auto flex items-center gap-3 text-xs min-w-0">
        <div className={`hidden xl:flex items-center gap-2 rounded border px-2.5 h-8 max-w-[520px] ${sessionShell}`}>
          <FlaskConical className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span className={`font-semibold ${sessionTitle}`}>Session</span>
          <span
            className={`flex min-w-0 max-w-32 items-center gap-1 rounded px-2 py-1 ${sessionBadge}`}
            title={`Model: ${modelName}`}
          >
            <Mic2 className={`w-3 h-3 shrink-0 ${sessionMuted}`} />
            <span className={sessionMuted}>M:</span>
            <span className="truncate">{modelName}</span>
          </span>
          <span
            className={`flex min-w-0 max-w-32 items-center gap-1 rounded px-2 py-1 ${sessionBadge}`}
            title={`Dataset: ${datasetName}`}
          >
            <Database className={`w-3 h-3 shrink-0 ${sessionMuted}`} />
            <span className={sessionMuted}>D:</span>
            <span className="truncate">{datasetName}</span>
          </span>
          <span
            className="rounded bg-emerald-500/10 px-2 py-1 font-mono text-emerald-400 shrink-0"
            title={selectedSample ? `Sample #${String(selectedSample.index + 1).padStart(4, '0')}` : 'No sample selected'}
          >
            S: {selectedSample ? `#${String(selectedSample.index + 1).padStart(4, '0')}` : 'none'}
          </span>
        </div>
        <div className={`flex h-7 items-center rounded border p-0.5 ${toggleShell}`}>
          <button
            type="button"
            onClick={() => onThemeChange('dark')}
            aria-label="Use dark mode"
            className={`h-6 w-7 rounded flex items-center justify-center transition-colors ${
              theme === 'dark' ? 'bg-emerald-500/20 text-emerald-400' : 'text-gray-500 hover:text-gray-200'
            }`}
          >
            <Moon className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => onThemeChange('light')}
            aria-label="Use light mode"
            className={`h-6 w-7 rounded flex items-center justify-center transition-colors ${
              theme === 'light' ? 'bg-emerald-500/20 text-emerald-600' : 'text-gray-500 hover:text-gray-200'
            }`}
          >
            <Sun className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          <span className="text-gray-400 max-w-[360px] truncate">{status || 'Local Mode'}</span>
        </div>
        <div className="text-gray-500">v1.0.0</div>
      </div>
    </header>
  );
}
