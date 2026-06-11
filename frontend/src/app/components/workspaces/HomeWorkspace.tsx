import { BarChart3, Database, FlaskConical, Layers3, Mic2, PencilLine, Play, Search } from 'lucide-react';
import { Workspace, useApi } from '../../api';

const FALLBACK_BENCHMARK_RESULTS = 110;
const FALLBACK_CACHE_RESULTS = 800;
const FALLBACK_ANALYSIS_RUNS = 8;

export function HomeWorkspace({ onWorkspaceChange }: { onWorkspaceChange: (workspace: Workspace) => void }) {
  const { models, datasets, selectedModel, selectedDataset, selectedSample } = useApi();
  const selectedModelName = models.find((model) => model.id === selectedModel)?.name || selectedModel || 'Choose a model';
  const selectedDatasetName = datasets.find((dataset) => dataset.id === selectedDataset)?.name || selectedDataset || 'Choose a dataset';
  const sampleLabel = selectedSample ? `#${String(selectedSample.index + 1).padStart(4, '0')}` : 'No sample selected';
  const nextStep = selectedSample
    ? {
        title: 'Run transcription on the selected sample.',
        action: 'Open Workbench',
        workspace: 'workbench' as Workspace,
      }
    : {
        title: 'Load a dataset sample and run your first transcription.',
        action: 'Browse Datasets',
        workspace: 'dataset' as Workspace,
      };

  const workflows = [
    {
      title: 'Transcribe Audio',
      subtitle: 'Record -> Predict -> Evaluate',
      body: 'Record microphone audio, upload a file, or use a selected dataset sample, then run ASR with the active model and inspect WER, CER, latency, and transcript differences.',
      button: 'Open Workbench',
      workspace: 'workbench' as Workspace,
      icon: <Mic2 className="w-5 h-5" />,
      tone: 'emerald',
    },
    {
      title: 'Explore Datasets',
      subtitle: 'Inspect -> Preview -> Select',
      body: 'Browse public and custom datasets, preview audio clips, inspect transcripts, and select samples for transcription or benchmarking.',
      button: 'Browse Datasets',
      workspace: 'dataset' as Workspace,
      icon: <Database className="w-5 h-5" />,
      tone: 'blue',
    },
    {
      title: 'Build Datasets',
      subtitle: 'Label -> Correct -> Save',
      body: 'Upload or record audio, write human transcripts, add metadata, manage accepted spelling variants, and save corrected examples into editable local datasets.',
      button: 'Open Label Studio',
      workspace: 'label' as Workspace,
      icon: <PencilLine className="w-5 h-5" />,
      tone: 'amber',
    },
    {
      title: 'Benchmark Systems',
      subtitle: 'Compare -> Cache -> Report',
      body: 'Run multi-model, multi-dataset analysis matrices with sample caps, optional noise settings, cached inference, saved runs, and report-ready benchmark metrics.',
      button: 'Run Analysis',
      workspace: 'analysis' as Workspace,
      icon: <BarChart3 className="w-5 h-5" />,
      tone: 'violet',
    },
  ];

  return (
    <div className="h-full overflow-y-auto bg-[#0a0a0b]">
      <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        <section className="border border-gray-800 bg-[#121214] rounded-lg p-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
            <div className="flex items-start gap-4 min-w-0">
              <div className="w-12 h-12 rounded bg-emerald-500/15 text-emerald-400 flex items-center justify-center shrink-0">
                <FlaskConical className="w-7 h-7" />
              </div>
              <div className="min-w-0">
                <div className="text-[10px] uppercase tracking-wider text-emerald-400 font-semibold">Research platform</div>
                <h1 className="mt-1 text-3xl font-semibold text-gray-100 tracking-tight">Darija Automatic Speech Recognition Workbench</h1>
                <p className="mt-2 text-sm font-medium text-gray-300">Research • Benchmarking • Dataset Construction</p>
                <p className="mt-3 text-sm text-gray-400 leading-relaxed max-w-3xl">
                  A local research laboratory for Moroccan Darija speech recognition: evaluate ASR models, inspect public datasets,
                  build custom labeled corpora, and compare systems through reproducible benchmark runs.
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 lg:justify-end">
              <HeroButton primary label="Run Transcription" icon={<Play className="w-4 h-4" />} onClick={() => onWorkspaceChange('workbench')} />
              <HeroButton label="Browse Models" icon={<Layers3 className="w-4 h-4" />} onClick={() => onWorkspaceChange('models')} />
              <HeroButton label="Start Analysis" icon={<BarChart3 className="w-4 h-4" />} onClick={() => onWorkspaceChange('analysis')} />
            </div>
          </div>
        </section>

        <section className="space-y-3">
          <SectionTitle title="Research Platform Snapshot" />
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <StatCard value={String(models.length || 11)} label="Models" />
            <StatCard value={String(datasets.length || 10)} label="Datasets" />
            <StatCard value={String(FALLBACK_BENCHMARK_RESULTS)} label="Benchmark Results" />
            <StatCard value={String(FALLBACK_CACHE_RESULTS)} label="Cached Sample Results" />
            <StatCard value={String(FALLBACK_ANALYSIS_RUNS)} label="Analysis Runs" />
          </div>
        </section>

        <section className="space-y-3">
          <SectionTitle title="Active Context" />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <ContextCard
              label="Active Model"
              value={selectedModelName}
              action="Change model"
              icon={<Layers3 className="w-4 h-4" />}
              onClick={() => onWorkspaceChange('models')}
            />
            <ContextCard
              label="Active Dataset"
              value={selectedDatasetName}
              action="Browse samples"
              icon={<Database className="w-4 h-4" />}
              onClick={() => onWorkspaceChange('dataset')}
            />
            <ContextCard
              label="Active Sample"
              value={sampleLabel}
              action={selectedSample ? 'Change sample' : 'Select sample'}
              icon={<Search className="w-4 h-4" />}
              onClick={() => onWorkspaceChange('dataset')}
            />
          </div>
        </section>

        <section className="space-y-3">
          <SectionTitle title="Core Workflows" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {workflows.map((workflow) => (
              <WorkflowCard
                key={workflow.title}
                {...workflow}
                onClick={() => onWorkspaceChange(workflow.workspace)}
              />
            ))}
          </div>
        </section>

        <section className="border border-gray-800 bg-[#101012] rounded-lg p-5 space-y-4">
          <SectionTitle title="How the Workbench Fits Together" />
          <div className="rounded border border-gray-800 bg-[#0a0a0b] p-4 font-mono text-xs text-gray-300 leading-relaxed overflow-x-auto">
            <pre>{`Models Registry ----.
                    +--> Workbench Execution ----> Label / Save Dataset
Datasets Registry --'

Models + Datasets ----> Analysis Matrix ----> Cached Results + Saved Runs`}</pre>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Rule title="Models and datasets are registries." body="They are used for inspection, selection, and benchmark configuration." />
            <Rule title="Workbench is execution." body="Direct transcription happens here using the active model and input." />
            <Rule title="Label is dataset construction." body="Recorded, uploaded, or copied samples become corrected training examples." />
            <Rule title="Analysis is benchmarking." body="Model-dataset matrices generate cached, reusable, report-ready metrics." />
          </div>
        </section>

        <section className="border border-emerald-500/25 bg-emerald-500/10 rounded-lg p-5">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-emerald-400 font-semibold">Recommended next step</div>
              <h2 className="mt-1 text-lg font-semibold text-gray-100">{nextStep.title}</h2>
            </div>
            <button
              onClick={() => onWorkspaceChange(nextStep.workspace)}
              className="h-10 px-4 rounded bg-emerald-500 text-white text-sm font-medium hover:bg-emerald-600 transition-colors flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4" />
              {nextStep.action}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}

function HeroButton({ label, icon, onClick, primary }: { label: string; icon: JSX.Element; onClick: () => void; primary?: boolean }) {
  return (
    <button
      onClick={onClick}
      className={`h-10 px-3 rounded text-sm font-medium transition-colors flex items-center gap-2 ${
        primary
          ? 'bg-emerald-500 text-white hover:bg-emerald-600'
          : 'border border-gray-700 bg-[#0f0f11] text-gray-300 hover:bg-gray-800'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function SectionTitle({ title }: { title: string }) {
  return <h2 className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold">{title}</h2>;
}

function StatCard({ value, label }: { value: string; label: string }) {
  return (
    <div className="border border-gray-800 bg-[#121214] rounded-lg p-4 min-w-0">
      <div className="text-2xl font-semibold text-gray-100 font-mono">{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-gray-500 mt-1">{label}</div>
    </div>
  );
}

function ContextCard({ label, value, action, icon, onClick }: { label: string; value: string; action: string; icon: JSX.Element; onClick: () => void }) {
  return (
    <div className="border border-gray-800 bg-[#121214] rounded-lg p-4 min-w-0">
      <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">{label}</div>
      <div className="text-sm text-gray-100 truncate" title={value}>{value}</div>
      <button onClick={onClick} className="mt-3 h-8 px-2 rounded bg-gray-800 text-xs text-gray-300 hover:bg-gray-700 transition-colors flex items-center gap-1.5">
        {icon}
        {action}
      </button>
    </div>
  );
}

function WorkflowCard({
  title,
  subtitle,
  body,
  button,
  icon,
  tone,
  onClick,
}: {
  title: string;
  subtitle: string;
  body: string;
  button: string;
  icon: JSX.Element;
  tone: string;
  onClick: () => void;
}) {
  return (
    <div className="border border-gray-800 bg-[#121214] rounded-lg p-5 min-h-56 flex flex-col">
      <div className={`w-10 h-10 rounded flex items-center justify-center mb-4 ${toneClass(tone)}`}>
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-gray-100">{title}</h3>
      <div className="text-xs text-gray-500 mt-1">{subtitle}</div>
      <p className="mt-3 text-sm text-gray-400 leading-relaxed flex-1">{body}</p>
      <button onClick={onClick} className="mt-4 h-9 px-3 rounded border border-gray-700 bg-[#0f0f11] text-sm text-gray-300 hover:bg-gray-800 transition-colors self-start">
        {button}
      </button>
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
  if (tone === 'amber') return 'bg-amber-500/15 text-amber-400';
  if (tone === 'violet') return 'bg-violet-500/15 text-violet-400';
  return 'bg-emerald-500/15 text-emerald-400';
}
