import { BarChart3, Database, FlaskConical, Layers3, Mic2, PencilLine } from 'lucide-react';
import { Workspace, useApi } from '../../api';

export function HomeWorkspace({ onWorkspaceChange }: { onWorkspaceChange: (workspace: Workspace) => void }) {
  const { models, datasets, selectedModel, selectedDataset, selectedSample } = useApi();
  const selectedModelName = models.find((model) => model.id === selectedModel)?.name || selectedModel || 'Choose a model';
  const selectedDatasetName = datasets.find((dataset) => dataset.id === selectedDataset)?.name || selectedDataset || 'Choose a dataset';

  const cards = [
    {
      title: 'Run Transcription',
      body: 'Record, upload, or use the selected dataset sample with the active model.',
      icon: <Mic2 className="w-5 h-5" />,
      workspace: 'workbench' as Workspace,
      tone: 'emerald',
    },
    {
      title: 'Inspect Models',
      body: 'Browse model descriptions, runtime requirements, and benchmark statistics.',
      icon: <Layers3 className="w-5 h-5" />,
      workspace: 'models' as Workspace,
      tone: 'blue',
    },
    {
      title: 'Browse Datasets',
      body: 'Inspect public datasets, load clips, and select the active sample.',
      icon: <Database className="w-5 h-5" />,
      workspace: 'dataset' as Workspace,
      tone: 'cyan',
    },
    {
      title: 'Create Dataset',
      body: 'Save recordings, correct transcripts, and build editable local datasets.',
      icon: <PencilLine className="w-5 h-5" />,
      workspace: 'label' as Workspace,
      tone: 'amber',
    },
    {
      title: 'Run Analysis',
      body: 'Compare multiple models and datasets with cached benchmark matrices.',
      icon: <BarChart3 className="w-5 h-5" />,
      workspace: 'analysis' as Workspace,
      tone: 'violet',
    },
  ];

  return (
    <div className="h-full overflow-y-auto bg-[#0a0a0b]">
      <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        <section className="border border-gray-800 bg-[#121214] rounded-lg p-6">
          <div className="flex items-start gap-4">
            <div className="w-11 h-11 rounded bg-emerald-500/15 text-emerald-400 flex items-center justify-center shrink-0">
              <FlaskConical className="w-6 h-6" />
            </div>
            <div className="min-w-0">
              <h1 className="text-2xl font-semibold text-gray-100 tracking-tight">Darija ASR Research Workbench</h1>
              <p className="mt-2 text-sm text-gray-400 leading-relaxed max-w-3xl">
                A local lab for Moroccan Darija speech recognition: inspect models and datasets as registries,
                run transcription in the Workbench, build custom labeled datasets, and compare systems in Analysis.
              </p>
            </div>
          </div>
        </section>

        <section className="grid grid-cols-3 gap-3">
          <SessionCard label="Active Model" value={selectedModelName} />
          <SessionCard label="Active Dataset" value={selectedDatasetName} />
          <SessionCard label="Active Sample" value={selectedSample ? `#${String(selectedSample.index + 1).padStart(4, '0')}` : 'No sample selected'} />
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {cards.map((card) => (
            <button
              key={card.title}
              onClick={() => onWorkspaceChange(card.workspace)}
              className="text-left border border-gray-800 bg-[#121214] hover:bg-gray-800/40 rounded-lg p-4 transition-colors min-h-36"
            >
              <div className={`w-9 h-9 rounded flex items-center justify-center mb-4 ${toneClass(card.tone)}`}>
                {card.icon}
              </div>
              <h2 className="text-base font-medium text-gray-100">{card.title}</h2>
              <p className="mt-2 text-sm text-gray-500 leading-relaxed">{card.body}</p>
            </button>
          ))}
        </section>

        <section className="border border-gray-800 bg-[#101012] rounded-lg p-4">
          <h2 className="text-sm font-medium text-gray-200 mb-3">Application Structure</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-gray-400">
            <Rule title="Models and Datasets are registries" body="They are for inspection and selection, not for running transcription." />
            <Rule title="Workbench is execution" body="All recording, uploading, sample transcription, and direct ASR runs happen there." />
            <Rule title="Label is dataset building" body="Use it to create corrected examples and editable local assets." />
            <Rule title="Analysis is benchmarking" body="Use it for model-by-dataset comparison, cached results, and report-ready metrics." />
          </div>
        </section>
      </div>
    </div>
  );
}

function SessionCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-gray-800 bg-[#121214] rounded-lg p-3 min-w-0">
      <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">{label}</div>
      <div className="text-sm text-gray-100 truncate">{value}</div>
    </div>
  );
}

function Rule({ title, body }: { title: string; body: string }) {
  return (
    <div className="border border-gray-800 bg-[#0a0a0b] rounded p-3">
      <div className="text-gray-200 font-medium">{title}</div>
      <div className="text-xs text-gray-500 mt-1 leading-relaxed">{body}</div>
    </div>
  );
}

function toneClass(tone: string) {
  if (tone === 'blue') return 'bg-blue-500/15 text-blue-400';
  if (tone === 'cyan') return 'bg-cyan-500/15 text-cyan-400';
  if (tone === 'amber') return 'bg-amber-500/15 text-amber-400';
  if (tone === 'violet') return 'bg-violet-500/15 text-violet-400';
  return 'bg-emerald-500/15 text-emerald-400';
}
