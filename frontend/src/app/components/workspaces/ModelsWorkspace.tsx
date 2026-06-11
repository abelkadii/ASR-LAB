import { useMemo, useState } from 'react';
import { CheckCircle2, Gauge, Layers3, Search, Zap } from 'lucide-react';
import { ModelInfo, Workspace, useApi } from '../../api';
import registryBenchmark from '../../../../../data/registry_benchmark_results.json';

type BenchmarkSnapshot = {
  wer: string;
  cer: string;
  speed: string;
  memory: string;
  datasets: string;
  samples: string;
};

type ScoreBuckets = NonNullable<ModelInfo['benchmark']>['distribution'];
type BenchmarkRow = {
  model_id?: string;
  dataset_id?: string;
  status?: string;
  count?: number;
  avg_wer?: number | null;
  avg_cer?: number | null;
  avg_elapsed_seconds?: number | null;
  cache_hits?: number;
  cache_misses?: number;
  error?: string | null;
};

function architectureFor(model: ModelInfo) {
  if (model.framework === 'ctc-transformers') return 'CTC Transformers';
  if (model.framework === 'speechbrain') return 'SpeechBrain CTC';
  if (model.framework === 'peft+transformers') return 'PEFT Whisper Adapter';
  return 'Transformers';
}

function modelStatus(model: ModelInfo) {
  if (model.framework === 'peft+transformers') return 'adapter';
  if ((model.min_vram_gb || 0) >= 3) return 'heavy';
  return 'available';
}

function shortName(name: string) {
  return name
    .replace(/\s*\([^)]*\)/g, '')
    .replace('Moroccan Darija', 'Darija')
    .replace('Fine-Tuned', 'FT')
    .trim();
}

function benchmarkFor(model: ModelInfo): BenchmarkSnapshot {
  const memory = `${model.min_vram_gb || 0} GB`;
  const stats = model.benchmark;
  if (stats && stats.samples > 0) {
    return {
      wer: formatPercent(stats.avg_wer),
      cer: formatPercent(stats.avg_cer),
      speed: formatSpeed(stats.avg_elapsed_seconds),
      memory,
      datasets: String(stats.runs || '-'),
      samples: stats.samples.toLocaleString(),
    };
  }
  return {
    wer: 'Not run',
    cer: 'Not run',
    speed: 'Pending',
    memory,
    datasets: '-',
    samples: '-',
  };
}

function benchmarkDetailsFor(model: ModelInfo): NonNullable<ModelInfo['benchmark']> | null {
  const existing = model.benchmark || null;
  if (existing?.dataset_scores?.length && existing.distribution && existing.rank) return existing;

  const raw = registryBenchmark as {
    results?: BenchmarkRow[];
    model_stats?: Record<string, NonNullable<ModelInfo['benchmark']>>;
  };
  const stats = raw.model_stats?.[model.id] || existing;
  if (!stats) return existing;

  const allRows = raw.results || [];
  const rows = allRows
    .filter((row) => row.model_id === model.id && row.status !== 'failed' && !row.error && (row.count || 0) > 0 && typeof row.avg_wer === 'number')
    .sort((a, b) => Number(a.avg_wer) - Number(b.avg_wer));

  const rankedModels = Object.entries(raw.model_stats || {})
    .filter(([, value]) => (value.samples || 0) > 0 && typeof value.avg_wer === 'number')
    .sort(([, a], [, b]) => Number(a.avg_wer) - Number(b.avg_wer));

  const distribution = { excellent: 0, good: 0, fair: 0, poor: 0, failed: 0 };
  for (const row of rows) {
    const wer = Number(row.avg_wer);
    if (wer < 0.35) distribution.excellent += 1;
    else if (wer < 0.55) distribution.good += 1;
    else if (wer < 0.8) distribution.fair += 1;
    else distribution.poor += 1;
  }
  distribution.failed = allRows.filter((row) => row.model_id === model.id && (row.status === 'failed' || row.error)).length;

  return {
    ...stats,
    ...existing,
    rank: rankedModels.findIndex(([id]) => id === model.id) + 1 || null,
    rank_total: rankedModels.length,
    best: existing?.best || stats.best || rowToScore(rows[0]),
    worst: existing?.worst || stats.worst || rowToScore(rows[rows.length - 1]),
    distribution: existing?.distribution || distribution,
    dataset_scores: existing?.dataset_scores?.length
      ? existing.dataset_scores
      : rows.map((row) => ({
          dataset_id: row.dataset_id,
          count: row.count,
          avg_wer: row.avg_wer,
          avg_cer: row.avg_cer,
          avg_elapsed_seconds: row.avg_elapsed_seconds,
          cache_hits: row.cache_hits || 0,
          cache_misses: row.cache_misses || 0,
        })),
  };
}

function rowToScore(row?: BenchmarkRow) {
  if (!row) return null;
  return {
    model_id: row.model_id,
    dataset_id: row.dataset_id,
    avg_wer: row.avg_wer,
    avg_cer: row.avg_cer,
  };
}

export function ModelsWorkspace({ onWorkspaceChange }: { onWorkspaceChange: (workspace: Workspace) => void }) {
  const { models, selectedModel, setSelectedModel, setStatus } = useApi();
  const [query, setQuery] = useState('');
  const [frameworkFilter, setFrameworkFilter] = useState('all');
  const [vramFilter, setVramFilter] = useState('all');
  const [benchmarkFilter, setBenchmarkFilter] = useState('all');

  const frameworks = useMemo(
    () => Array.from(new Set(models.map((model) => model.framework))).sort(),
    [models],
  );

  const filteredModels = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return models.filter((model) => {
      const text = `${model.name} ${model.id} ${model.framework}`.toLowerCase();
      const matchesQuery = !normalizedQuery || text.includes(normalizedQuery);
      const matchesFramework = frameworkFilter === 'all' || model.framework === frameworkFilter;
      const vram = model.min_vram_gb || 0;
      const hasBenchmark = Boolean(model.benchmark && model.benchmark.samples > 0);
      const matchesVram =
        vramFilter === 'all' ||
        (vramFilter === 'low' && vram <= 1) ||
        (vramFilter === 'mid' && vram > 1 && vram <= 3) ||
        (vramFilter === 'high' && vram > 3);
      const matchesBenchmark =
        benchmarkFilter === 'all' ||
        (benchmarkFilter === 'available' && hasBenchmark) ||
        (benchmarkFilter === 'missing' && !hasBenchmark);
      return matchesQuery && matchesFramework && matchesVram && matchesBenchmark;
    });
  }, [benchmarkFilter, frameworkFilter, models, query, vramFilter]);

  const selected = models.find((model) => model.id === selectedModel) || filteredModels[0] || models[0];
  const selectedDetails = selected ? benchmarkDetailsFor(selected) : null;
  const selectedBenchmark = selected ? benchmarkFor(selected) : null;

  function activateModel(model: ModelInfo) {
    setSelectedModel(model.id);
    setStatus(`Active model: ${model.name}. Open Workbench to transcribe.`);
  }

  function openBenchmarkForModel(model: ModelInfo) {
    setSelectedModel(model.id);
    setStatus(`Benchmark setup opened with selected model: ${model.name}.`);
    onWorkspaceChange('analysis');
  }

  function openWorkbenchForModel(model: ModelInfo) {
    setSelectedModel(model.id);
    setStatus(`Workbench opened with selected model: ${model.name}.`);
    onWorkspaceChange('workbench');
  }

  return (
    <div className="h-full grid grid-cols-[320px_1fr] divide-x divide-gray-800 bg-[#0a0a0b]">
      <aside className="flex flex-col overflow-hidden bg-[#0f0f11]">
        <div className="p-4 border-b border-gray-800 space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="text-sm font-medium text-gray-100">Model Registry</h2>
              <p className="text-xs text-gray-500 mt-1">{filteredModels.length} of {models.length} model(s)</p>
            </div>
            <span className="rounded bg-emerald-500/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-emerald-400">
              Registry
            </span>
          </div>

          <label className="relative block">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search models..."
              className="h-9 w-full rounded border border-gray-800 bg-[#0a0a0b] pl-9 pr-3 text-sm text-gray-100 outline-none transition-colors placeholder:text-gray-600 focus:border-emerald-500"
            />
          </label>

          <div className="grid grid-cols-1 gap-2">
            <select
              value={frameworkFilter}
              onChange={(event) => setFrameworkFilter(event.target.value)}
              className="h-8 rounded border border-gray-800 bg-[#0a0a0b] px-2 text-xs text-gray-300 outline-none focus:border-emerald-500"
            >
              <option value="all">All frameworks</option>
              {frameworks.map((framework) => (
                <option key={framework} value={framework}>{framework}</option>
              ))}
            </select>
            <div className="grid grid-cols-2 gap-2">
              <select
                value={vramFilter}
                onChange={(event) => setVramFilter(event.target.value)}
                className="h-8 rounded border border-gray-800 bg-[#0a0a0b] px-2 text-xs text-gray-300 outline-none focus:border-emerald-500"
              >
                <option value="all">All VRAM</option>
                <option value="low">1 GB or less</option>
                <option value="mid">1-3 GB</option>
                <option value="high">Over 3 GB</option>
              </select>
              <select
                value={benchmarkFilter}
                onChange={(event) => setBenchmarkFilter(event.target.value)}
                className="h-8 rounded border border-gray-800 bg-[#0a0a0b] px-2 text-xs text-gray-300 outline-none focus:border-emerald-500"
              >
                <option value="all">All benchmark</option>
                <option value="available">Benchmark ready</option>
                <option value="missing">Awaiting run</option>
              </select>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {filteredModels.length === 0 ? (
            <div className="p-4 text-xs text-gray-500">No models match the current filters.</div>
          ) : filteredModels.map((model) => {
            const active = selectedModel === model.id;
            return (
              <button
                key={model.id}
                onClick={() => setSelectedModel(model.id)}
                className={`w-full border-b border-gray-800/50 px-4 py-4 text-left transition-colors hover:bg-gray-800/30 ${
                  active ? 'bg-emerald-500/10' : ''
                }`}
                title={model.name}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-[15px] font-semibold text-gray-100">{shortName(model.name)}</div>
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-gray-500">
                      <span>{model.min_vram_gb || 0} GB</span>
                      <span className="h-1 w-1 rounded-full bg-gray-700" />
                      <span>{architectureFor(model)}</span>
                    </div>
                  </div>
                  {active ? (
                    <span className="flex shrink-0 items-center gap-1 rounded bg-emerald-500/15 px-2 py-1 text-[10px] font-semibold text-emerald-400">
                      <CheckCircle2 className="h-3 w-3" />
                      Active
                    </span>
                  ) : (
                    <span className={`shrink-0 rounded px-2 py-1 text-[10px] font-semibold ${statusClass(modelStatus(model))}`}>
                      {modelStatus(model)}
                    </span>
                  )}
                </div>
                <div className="mt-3 flex items-center justify-end">
                  <span className="text-[11px] text-gray-600">{model.benchmark?.samples ? `${model.benchmark.samples} samples` : 'Benchmark pending'}</span>
                </div>
              </button>
            );
          })}
        </div>
      </aside>

      <main className="overflow-y-auto p-5 space-y-4">
        {selected && selectedBenchmark ? (
          <>
            <section className="rounded-lg border border-gray-800 bg-[#121214] p-5">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <h1 className="truncate text-2xl font-semibold text-gray-100">{selected.name}</h1>
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-gray-500">
                    <span>{architectureFor(selected)}</span>
                    <span className="h-1 w-1 rounded-full bg-gray-700" />
                    <span>{selected.min_vram_gb || 0} GB VRAM</span>
                    <span className="h-1 w-1 rounded-full bg-gray-700" />
                    <span>{selected.benchmark?.samples ? `${selected.benchmark.samples.toLocaleString()} benchmark samples` : 'Awaiting benchmark'}</span>
                  </div>
                  <p className="mt-1 break-all font-mono text-xs text-gray-600">{selected.id}</p>
                </div>
                <button
                  onClick={() => activateModel(selected)}
                  className={`flex h-9 items-center gap-2 rounded px-3 text-sm font-medium transition-colors ${
                    selectedModel === selected.id
                      ? 'bg-emerald-500/15 text-emerald-300'
                      : 'bg-emerald-500 text-white hover:bg-emerald-600'
                  }`}
                >
                  {selectedModel === selected.id ? <CheckCircle2 className="h-4 w-4" /> : <Zap className="h-4 w-4" />}
                  {selectedModel === selected.id ? 'Active Model' : 'Use In Workbench'}
                </button>
              </div>

              <div className="mt-5 grid grid-cols-4 gap-3">
                <HeroMetric label="WER" value={selectedBenchmark.wer} tone="emerald" />
                <HeroMetric label="CER" value={selectedBenchmark.cer} tone="blue" />
                <HeroMetric label="Speed" value={selectedBenchmark.speed} tone="amber" />
                <HeroMetric label="Memory" value={selectedBenchmark.memory} tone="gray" />
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3 text-xs md:grid-cols-4">
                <Info label="Runs" value={selectedBenchmark.datasets} />
                <Info label="Samples" value={selectedBenchmark.samples} />
                <Info label="Rank" value={formatRank(selectedDetails)} />
                <Info label="Cache" value={formatCache(selected.benchmark)} />
              </div>
            </section>

            <section className="grid grid-cols-[1fr_360px] gap-4">
              <div className="rounded-lg border border-gray-800 bg-[#121214] p-5">
                <div className="mb-4 flex items-center gap-2">
                  <Gauge className="h-4 w-4 text-emerald-400" />
                  <h2 className="text-sm font-medium text-gray-100">Benchmark Performance</h2>
                </div>
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3">
                    <Info label="Benchmark Rank" value={formatRank(selectedDetails)} />
                    <Info label="Best Run" value={formatPercent(selectedDetails?.best?.avg_wer)} />
                    <Info label="Worst Run" value={formatPercent(selectedDetails?.worst?.avg_wer)} />
                  </div>
                  <Info label="25%-75% WER" value={formatIqr(selectedDetails?.dataset_scores)} />
                  <ScoreDistribution distribution={selectedDetails?.distribution} />
                </div>
              </div>

              <div className="rounded-lg border border-gray-800 bg-[#121214] p-5">
                <div className="mb-4 flex items-center gap-2">
                  <Layers3 className="h-4 w-4 text-blue-400" />
                  <h2 className="text-sm font-medium text-gray-100">Model Capabilities</h2>
                </div>
                <div className="space-y-4 text-sm">
                  <Capability label="Transcription" enabled />
                  <Capability label="Batch analysis" enabled />
                  <Capability label="Benchmarking" enabled />
                  <div className="grid grid-cols-2 gap-2 pt-2">
                    <Info label="Input" value="16 kHz audio" />
                    <Info label="Output" value="Text" />
                    <Info label="Runtime" value={architectureFor(selected)} />
                    <Info label="Status" value={modelStatus(selected)} />
                  </div>
                </div>
              </div>
            </section>

            <section className="rounded-lg border border-emerald-500/25 bg-emerald-500/10 p-5">
              <div className="flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <div className="text-xs font-semibold uppercase tracking-wider text-emerald-400">Model Actions</div>
                  <h2 className="mt-1 truncate text-lg font-semibold text-gray-100">{selected.name}</h2>
                  <p className="mt-1 text-sm text-gray-400">
                    Use this model for immediate transcription or open Analysis to benchmark it against datasets.
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    onClick={() => openWorkbenchForModel(selected)}
                    className="h-10 rounded border border-gray-700 bg-[#0f0f11] px-4 text-sm font-medium text-gray-200 transition-colors hover:bg-gray-800 flex items-center gap-2"
                  >
                    <Zap className="h-4 w-4 text-emerald-400" />
                    Transcribe
                  </button>
                  <button
                    onClick={() => openBenchmarkForModel(selected)}
                    className="h-10 rounded bg-emerald-500 px-4 text-sm font-medium text-white transition-colors hover:bg-emerald-600 flex items-center gap-2"
                  >
                    <Gauge className="h-4 w-4" />
                    Benchmark
                  </button>
                </div>
              </div>
            </section>

          </>
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-gray-500">No model registry is available yet.</div>
        )}
      </main>
    </div>
  );
}

function statusClass(status: string) {
  if (status === 'adapter') return 'bg-blue-500/20 text-blue-400';
  if (status === 'heavy') return 'bg-amber-500/20 text-amber-400';
  return 'bg-emerald-500/20 text-emerald-400';
}

function HeroMetric({ label, value, tone }: { label: string; value: string; tone: 'emerald' | 'blue' | 'amber' | 'gray' }) {
  const toneClass = {
    emerald: 'text-emerald-300 border-emerald-500/25 bg-emerald-500/10',
    blue: 'text-blue-300 border-blue-500/25 bg-blue-500/10',
    amber: 'text-amber-300 border-amber-500/25 bg-amber-500/10',
    gray: 'text-gray-200 border-gray-700 bg-[#0f0f11]',
  }[tone];
  return (
    <div className={`rounded-lg border p-4 ${toneClass}`}>
      <div className="text-[11px] font-semibold uppercase tracking-wider opacity-75">{label}</div>
      <div className="mt-2 text-2xl font-semibold">{value}</div>
    </div>
  );
}

function Capability({ label, enabled }: { label: string; enabled: boolean }) {
  return (
    <div className="flex items-center justify-between rounded border border-gray-800 bg-[#0f0f11] px-3 py-2">
      <span className="text-gray-300">{label}</span>
      <span className={enabled ? 'text-emerald-400' : 'text-gray-600'}>{enabled ? 'Supported' : 'Unavailable'}</span>
    </div>
  );
}

function ScoreDistribution({ distribution }: { distribution?: ScoreBuckets }) {
  const buckets = [
    { key: 'excellent', label: '<35% WER', color: 'bg-emerald-500' },
    { key: 'good', label: '35-55%', color: 'bg-blue-500' },
    { key: 'fair', label: '55-80%', color: 'bg-amber-500' },
    { key: 'poor', label: '80%+', color: 'bg-red-500' },
  ] as const;
  const total = buckets.reduce((sum, bucket) => sum + Number(distribution?.[bucket.key] || 0), 0);
  return (
    <div className="rounded border border-gray-800 bg-[#0f0f11] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="text-xs font-medium uppercase tracking-wider text-gray-500">Benchmark Score Distribution</div>
        <div className="text-xs text-gray-500">{total ? 'WER spread' : 'Not available'}</div>
      </div>
      <div className="flex h-3 overflow-hidden rounded bg-gray-800">
        {buckets.map((bucket) => {
          const count = Number(distribution?.[bucket.key] || 0);
          if (!count || !total) return null;
          return <div key={bucket.key} className={bucket.color} style={{ width: `${(count / total) * 100}%` }} />;
        })}
      </div>
      <div className="mt-3 grid grid-cols-4 gap-2">
        {buckets.map((bucket) => (
          <div key={bucket.key} className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span className={`h-2 w-2 shrink-0 rounded-full ${bucket.color}`} />
              <span className="truncate text-[10px] text-gray-500">{bucket.label}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded border border-gray-800 p-3">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className="truncate text-gray-100" title={value}>{value}</div>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-gray-800 bg-[#121214] p-2">
      <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className="mt-1 text-xs text-gray-200">{value}</div>
    </div>
  );
}

function formatPercent(value?: number | null) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'Not run';
  return `${(value * 100).toFixed(1)}%`;
}

function formatSpeed(value?: number | null) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'Pending';
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  return `${value.toFixed(2)} s`;
}

function formatCache(stats?: ModelInfo['benchmark']) {
  if (!stats) return '-';
  const hits = stats.cache_hits || 0;
  const misses = stats.cache_misses || 0;
  return `${hits}/${hits + misses}`;
}

function formatRank(stats?: ModelInfo['benchmark']) {
  if (!stats?.rank || !stats.rank_total) return 'Not ranked';
  return `#${stats.rank} of ${stats.rank_total}`;
}

function formatIqr(rows?: NonNullable<ModelInfo['benchmark']>['dataset_scores']) {
  const values = (rows || [])
    .map((row) => row.avg_wer)
    .filter((value): value is number => typeof value === 'number' && !Number.isNaN(value))
    .sort((a, b) => a - b);
  if (!values.length) return 'Not available';
  const q1 = percentile(values, 0.25);
  const q3 = percentile(values, 0.75);
  return `${formatPercent(q1)} - ${formatPercent(q3)}`;
}

function percentile(values: number[], ratio: number) {
  if (values.length === 1) return values[0];
  const index = (values.length - 1) * ratio;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return values[lower];
  return values[lower] + (values[upper] - values[lower]) * (index - lower);
}
