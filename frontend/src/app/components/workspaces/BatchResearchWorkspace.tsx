import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Clock3, Database, Layers3, Loader2, Play, Shuffle } from 'lucide-react';
import { DatasetInfo, formatPercent, ModelInfo, Workspace, requestJson, useApi } from '../../api';

type AnalysisRun = {
  model_id: string;
  model_name: string;
  dataset_id: string;
  dataset_name: string;
  status?: 'ok' | 'failed';
  error?: string;
  summary: {
    count: number;
    avg_wer: number;
    avg_cer: number;
    avg_duration_seconds: number;
    avg_elapsed_seconds: number;
    elapsed_seconds: number;
    cache_hits?: number;
    cache_misses?: number;
    error?: string;
    hardest_words?: Array<{ word: string; seen: number; accuracy: number }>;
    word_substitutions?: Array<{ reference: string; hypothesis: string; count: number }>;
  };
  results: Array<{
    model_id: string;
    dataset_id: string;
    index: number;
    reference: string;
    hypothesis: string;
    wer: number;
    cer: number;
    duration_seconds: number;
    elapsed_seconds: number;
    cached_elapsed_seconds?: number | null;
    cache_hit?: boolean;
    rtf?: number;
  }>;
};

type AnalysisResponse = {
  success: boolean;
  settings: Record<string, unknown>;
  summary: {
    count: number;
    runs: number;
    avg_wer: number;
    avg_cer: number;
    avg_elapsed_seconds: number;
    elapsed_seconds: number;
    cache_hits?: number;
    cache_misses?: number;
    successful_runs?: number;
    failed_runs?: number;
    rankings: Array<{ model_id: string; dataset_id: string; status?: string; error?: string; avg_wer: number; avg_cer: number; count: number; avg_elapsed_seconds: number }>;
  };
  runs: AnalysisRun[];
  results: AnalysisRun['results'];
  run_id?: string;
  created_at?: string;
};

type SavedAnalysisRun = {
  run_id: string;
  created_at: string;
  model_ids: string[];
  dataset_ids: string[];
  settings: Record<string, unknown>;
  summary: AnalysisResponse['summary'];
};

export function BatchResearchWorkspace({ onWorkspaceChange }: { onWorkspaceChange: (workspace: Workspace) => void }) {
  const { models, datasets, selectedModel, selectedDataset, setStatus } = useApi();
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [sampleCount, setSampleCount] = useState('5');
  const [samplePool, setSamplePool] = useState('50');
  const [mixDatasets, setMixDatasets] = useState(false);
  const [addNoise, setAddNoise] = useState(false);
  const [snrDb, setSnrDb] = useState('20');
  const [busy, setBusy] = useState(false);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [savedRuns, setSavedRuns] = useState<SavedAnalysisRun[]>([]);

  useEffect(() => {
    if (models.length && selectedModels.length === 0) setSelectedModels([selectedModel || models[0].id]);
  }, [models, selectedModel, selectedModels.length]);

  useEffect(() => {
    if (datasets.length && selectedDatasets.length === 0) {
      setSelectedDatasets([selectedDataset || datasets.find((dataset) => dataset.id !== 'recorded-audio')?.id || datasets[0].id]);
    }
  }, [datasets, selectedDataset, selectedDatasets.length]);

  useEffect(() => {
    void loadSavedRuns();
  }, []);

  const plannedRuns = mixDatasets ? selectedModels.length : selectedModels.length * selectedDatasets.length;
  const plannedSamples = plannedRuns * Number(sampleCount || 0) * (mixDatasets ? Math.max(1, selectedDatasets.length) : 1);
  async function runAnalysis() {
    if (!selectedModels.length || !selectedDatasets.length) {
      setStatus('Choose at least one model and one dataset.');
      return;
    }
    setBusy(true);
    setStatus(`Running ${plannedRuns} analysis run(s)...`);
    try {
      const data = await requestJson<AnalysisResponse>('/analyze-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_ids: selectedModels,
          dataset_ids: selectedDatasets,
          limit: Number(sampleCount || 5),
          sample_pool: Number(samplePool || 50),
          mix_datasets: mixDatasets,
          add_noise: addNoise,
          snr_db: Number(snrDb || 20),
        }),
      });
      setAnalysis(data);
      await loadSavedRuns();
      setStatus(`Analysis complete: ${data.summary.runs} run(s), ${data.summary.count} sample result(s), ${data.summary.cache_hits || 0} cache hit(s).`);
    } catch (error) {
      setStatus(`Analysis failed: ${(error as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  async function loadSavedRuns() {
    try {
      const data = await requestJson<{ success: boolean; runs: SavedAnalysisRun[] }>('/analysis-runs?limit=20');
      setSavedRuns(data.runs || []);
    } catch {
      setSavedRuns([]);
    }
  }

  async function openSavedRun(runId: string) {
    setHistoryBusy(true);
    try {
      const data = await requestJson<AnalysisResponse>(`/analysis-runs/${runId}`);
      setAnalysis(data);
      setStatus(`Loaded saved analysis ${runId}.`);
    } catch (error) {
      setStatus(`Could not load saved analysis: ${(error as Error).message}`);
    } finally {
      setHistoryBusy(false);
    }
  }

  const chartRankings = analysis?.summary.rankings.filter((row) => row.status !== 'failed' && row.count > 0) || [];
  const rankingData = chartRankings.map((row, index) => ({
    name: `${shortName(row.model_id)} / ${shortName(row.dataset_id)}`,
    wer: Number((row.avg_wer * 100).toFixed(1)),
    cer: Number((row.avg_cer * 100).toFixed(1)),
    rank: index + 1,
  }));

  const runTrend = analysis?.runs.filter((run) => run.status !== 'failed' && run.summary.count > 0).map((run, index) => ({
    run: `R${index + 1}`,
    wer: Number((run.summary.avg_wer * 100).toFixed(1)),
    cer: Number((run.summary.avg_cer * 100).toFixed(1)),
    seconds: Number(run.summary.elapsed_seconds.toFixed(1)),
  })) || [];

  const selectedRun = analysis?.runs[0];
  const comparisonDatasets = useMemo(() => {
    if (!analysis) return [];
    return Array.from(new Set(analysis.runs.map((run) => run.dataset_id)));
  }, [analysis]);
  const comparisonModels = useMemo(() => {
    if (!analysis) return [];
    return Array.from(new Set(analysis.runs.map((run) => run.model_id)));
  }, [analysis]);
  const runLookup = useMemo(() => {
    const lookup = new Map<string, AnalysisRun>();
    analysis?.runs.forEach((run) => lookup.set(`${run.model_id}::${run.dataset_id}`, run));
    return lookup;
  }, [analysis]);

  return (
    <div className="h-full grid grid-cols-[380px_1fr] divide-x divide-gray-800">
      <aside className="flex flex-col overflow-hidden bg-[#0f0f11]">
        <div className="p-3 border-b border-gray-800">
          <h2 className="text-sm font-medium text-gray-200">Analysis Runs</h2>
          <p className="text-xs text-gray-500 mt-1">Multi-model and multi-dataset experiment matrix</p>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-4">
          <SelectorGroup title="Models" icon={<Layers3 className="w-3.5 h-3.5 text-emerald-400" />}>
            {models.map((model) => (
              <CheckRow
                key={model.id}
                checked={selectedModels.includes(model.id)}
                label={model.name}
                sublabel={model.framework}
                onChange={() => setSelectedModels(toggleValue(selectedModels, model.id))}
              />
            ))}
          </SelectorGroup>

          <SelectorGroup title="Datasets" icon={<Database className="w-3.5 h-3.5 text-blue-400" />}>
            {datasets.map((dataset) => (
              <CheckRow
                key={dataset.id}
                checked={selectedDatasets.includes(dataset.id)}
                label={dataset.name}
                sublabel={dataset.id}
                onChange={() => setSelectedDatasets(toggleValue(selectedDatasets, dataset.id))}
              />
            ))}
          </SelectorGroup>

          <div className="grid grid-cols-2 gap-2">
            <NumberField label="Samples / Dataset" value={sampleCount} onChange={setSampleCount} min={1} max={100} />
            <NumberField label="Sample Pool" value={samplePool} onChange={setSamplePool} min={1} max={300} />
          </div>

          <div className="space-y-2 border-t border-gray-800 pt-3">
            <label className="flex items-center gap-2 text-xs text-gray-400">
              <input type="checkbox" checked={mixDatasets} onChange={(event) => setMixDatasets(event.target.checked)} className="accent-emerald-500" />
              <Shuffle className="w-3.5 h-3.5" />
              Mix selected datasets into one run per model
            </label>
            <label className="flex items-center gap-2 text-xs text-gray-400">
              <input type="checkbox" checked={addNoise} onChange={(event) => setAddNoise(event.target.checked)} className="accent-emerald-500" />
              Add noise
            </label>
            {addNoise && <NumberField label="SNR dB" value={snrDb} onChange={setSnrDb} min={1} max={60} />}
          </div>

          <div className="grid grid-cols-3 gap-2">
            <SummaryCard label="Runs" value={String(plannedRuns)} />
            <SummaryCard label="Samples" value={String(plannedSamples)} />
            <SummaryCard label="Models" value={String(selectedModels.length)} />
          </div>

          <button
            onClick={() => void runAnalysis()}
            disabled={busy}
            className="w-full h-10 rounded flex items-center justify-center gap-2 text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-600 transition-colors disabled:opacity-50"
          >
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {busy ? 'Running Matrix...' : 'Run Analysis Matrix'}
          </button>

          <SelectorGroup title="Saved Runs" icon={<Clock3 className="w-3.5 h-3.5 text-amber-400" />}>
            {savedRuns.length === 0 ? (
              <div className="p-3 text-xs text-gray-500">No saved analysis runs yet.</div>
            ) : savedRuns.map((run) => (
              <button
                key={run.run_id}
                type="button"
                onClick={() => void openSavedRun(run.run_id)}
                disabled={historyBusy}
                className="w-full text-left p-2.5 border-b border-gray-800/60 last:border-b-0 hover:bg-gray-800/30 disabled:opacity-60"
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="text-xs text-gray-200 truncate">{run.summary?.runs || 0} run(s), {run.summary?.count || 0} samples</span>
                  <span className="text-[10px] text-emerald-400 font-mono">{run.summary?.cache_hits || 0} hit</span>
                </span>
                <span className="block text-[10px] text-gray-500 truncate">{formatRunDate(run.created_at)} - {shortName((run.model_ids || [])[0] || 'model')}</span>
              </button>
            ))}
          </SelectorGroup>
        </div>
      </aside>

      <main className="flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {!analysis ? (
            <div className="h-full flex items-center justify-center p-6 text-center text-sm text-gray-500">
              Select models and datasets, then run an analysis matrix.
            </div>
          ) : (
            <>
              <div className="grid grid-cols-8 gap-3">
                <SummaryCard label="Runs" value={String(analysis.summary.runs)} />
                <SummaryCard label="Samples" value={String(analysis.summary.count)} />
                <SummaryCard label="Average WER" value={formatPercent(analysis.summary.avg_wer)} color="text-amber-400" />
                <SummaryCard label="Average CER" value={formatPercent(analysis.summary.avg_cer)} color="text-emerald-400" />
                <SummaryCard label="Elapsed" value={`${analysis.summary.elapsed_seconds}s`} />
                <SummaryCard label="Cache Hits" value={String(analysis.summary.cache_hits || 0)} color="text-emerald-400" />
                <SummaryCard label="Cache Misses" value={String(analysis.summary.cache_misses || 0)} color="text-blue-300" />
                <SummaryCard label="Failed" value={String(analysis.summary.failed_runs || 0)} color={(analysis.summary.failed_runs || 0) ? 'text-red-300' : 'text-gray-200'} />
              </div>

              <ChartPanel title="Final Model x Dataset Comparison">
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[760px] text-xs">
                    <thead className="text-gray-500">
                      <tr className="border-b border-gray-800">
                        <th className="py-2 pr-3 text-left sticky left-0 bg-[#121214] z-10">Model</th>
                        {comparisonDatasets.map((datasetId) => (
                          <th key={datasetId} className="py-2 px-2 text-left min-w-44">{datasetName(datasets, datasetId)}</th>
                        ))}
                        <th className="py-2 pl-3 text-right">Avg WER</th>
                        <th className="py-2 pl-3 text-right">Avg CER</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonModels.map((modelId) => {
                        const modelRuns = comparisonDatasets
                          .map((datasetId) => runLookup.get(`${modelId}::${datasetId}`))
                          .filter((run): run is AnalysisRun => Boolean(run) && run.status !== 'failed' && run.summary.count > 0);
                        const avgWer = average(modelRuns.map((run) => run.summary.avg_wer));
                        const avgCer = average(modelRuns.map((run) => run.summary.avg_cer));
                        return (
                          <tr key={modelId} className="border-b border-gray-800/60 text-gray-300 align-top">
                            <td className="py-2 pr-3 sticky left-0 bg-[#121214] z-10">
                              <div className="font-medium text-gray-100">{modelName(models, modelId)}</div>
                              <div className="text-[10px] text-gray-500 font-mono truncate max-w-52">{modelId}</div>
                            </td>
                            {comparisonDatasets.map((datasetId) => {
                              const run = runLookup.get(`${modelId}::${datasetId}`);
                              return (
                                <td key={`${modelId}-${datasetId}`} className="py-2 px-2">
                                  {run ? (
                                    <ComparisonCell run={run} />
                                  ) : (
                                    <span className="text-gray-600">not run</span>
                                  )}
                                </td>
                              );
                            })}
                            <td className="py-2 pl-3 text-right font-mono text-amber-300">{formatPercent(avgWer)}</td>
                            <td className="py-2 pl-3 text-right font-mono text-emerald-300">{formatPercent(avgCer)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </ChartPanel>

              <div className="grid grid-cols-[1.2fr_1fr] gap-4">
                <ChartPanel title="Run Ranking">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={rankingData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                      <XAxis dataKey="name" tick={{ fill: 'var(--chart-tick)', fontSize: 10 }} stroke="var(--chart-axis)" interval={0} angle={-12} height={56} />
                      <YAxis tick={{ fill: 'var(--chart-tick)', fontSize: 11 }} stroke="var(--chart-axis)" />
                      <Tooltip contentStyle={{ backgroundColor: 'var(--chart-tooltip-bg)', border: '1px solid var(--chart-tooltip-border)', borderRadius: '6px', color: 'var(--chart-tooltip-text)' }} />
                      <Bar dataKey="wer" radius={[4, 4, 0, 0]} name="WER %">
                        {rankingData.map((_, index) => <Cell key={index} fill={index === 0 ? '#10b981' : '#f59e0b'} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </ChartPanel>

                <ChartPanel title="Run Trend">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={runTrend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                      <XAxis dataKey="run" tick={{ fill: 'var(--chart-tick)', fontSize: 11 }} stroke="var(--chart-axis)" />
                      <YAxis tick={{ fill: 'var(--chart-tick)', fontSize: 11 }} stroke="var(--chart-axis)" />
                      <Tooltip contentStyle={{ backgroundColor: 'var(--chart-tooltip-bg)', border: '1px solid var(--chart-tooltip-border)', borderRadius: '6px', color: 'var(--chart-tooltip-text)' }} />
                      <Line type="monotone" dataKey="wer" stroke="#f59e0b" strokeWidth={2} name="WER %" />
                      <Line type="monotone" dataKey="cer" stroke="#10b981" strokeWidth={2} name="CER %" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartPanel>
              </div>

              <section className="space-y-3">
                <div>
                  <h3 className="text-sm font-medium text-gray-200">Comparison Subwindows</h3>
                  <p className="text-xs text-gray-500 mt-1">Each panel is one model tested against one dataset, with its own sample-level evidence.</p>
                </div>
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  {analysis.runs.map((run, index) => (
                    <RunComparisonPanel
                      key={`${run.model_id}-${run.dataset_id}-${index}`}
                      run={run}
                      rank={analysis.summary.rankings.findIndex((row) => row.model_id === run.model_id && row.dataset_id === run.dataset_id) + 1}
                    />
                  ))}
                </div>
              </section>

              {selectedRun && (
                <div className="grid grid-cols-2 gap-4">
                  <WordList title="Hardest Words In First Run" rows={selectedRun.summary.hardest_words || []} />
                  <SubstitutionList rows={selectedRun.summary.word_substitutions || []} />
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

function toggleValue(values: string[], value: string) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function shortName(value: string) {
  if (value === 'mixed') return 'mixed';
  return value.split('/').pop()?.replace(/^(whisper-|models--)/, '').slice(0, 18) || value;
}

function formatRunDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'saved run';
  return date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function runCacheLabel(run?: AnalysisRun) {
  if (!run) return '-';
  const hits = run.summary.cache_hits || 0;
  const misses = run.summary.cache_misses || 0;
  return `${hits}/${hits + misses}`;
}

function average(values: number[]) {
  const usable = values.filter((value) => Number.isFinite(value));
  if (!usable.length) return 0;
  return usable.reduce((sum, value) => sum + value, 0) / usable.length;
}

function ComparisonCell({ run }: { run: AnalysisRun }) {
  if (run.status === 'failed') {
    return (
      <div className="space-y-1">
        <div className="text-red-300 font-mono">failed</div>
        <div className="line-clamp-2 text-[10px] text-gray-500">{run.error || run.summary.error || 'Model run failed'}</div>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-3">
        <span className={scoreClass(run.summary.avg_wer)}>WER {formatPercent(run.summary.avg_wer)}</span>
        <span className="text-gray-500 font-mono">{run.summary.count} samples</span>
      </div>
      <div className="flex items-center justify-between gap-3 text-[10px]">
        <span className="text-emerald-300 font-mono">CER {formatPercent(run.summary.avg_cer)}</span>
        <span className="text-gray-500 font-mono">cache {runCacheLabel(run)}</span>
      </div>
    </div>
  );
}

function RunComparisonPanel({ run, rank }: { run: AnalysisRun; rank: number }) {
  const previewRows = run.results.slice(0, 5);
  const failed = run.status === 'failed';
  return (
    <section className="bg-[#121214] border border-gray-800 rounded-lg overflow-hidden">
      <div className="p-3 border-b border-gray-800 bg-[#101014]">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h4 className="text-sm font-medium text-gray-100 truncate">{run.model_name}</h4>
            <p className="text-xs text-gray-500 truncate">{run.dataset_name}</p>
          </div>
          <span className={`shrink-0 rounded px-2 py-1 text-[10px] font-mono ${failed ? 'bg-red-500/15 text-red-300' : 'bg-emerald-500/15 text-emerald-300'}`}>
            {failed ? 'failed' : `#${rank || '-'}`}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-2 p-3 border-b border-gray-800">
        <MiniMetric label="WER" value={failed ? '-' : formatPercent(run.summary.avg_wer)} color={failed ? 'text-red-300' : scoreClass(run.summary.avg_wer)} />
        <MiniMetric label="CER" value={failed ? '-' : formatPercent(run.summary.avg_cer)} color={failed ? 'text-red-300' : 'text-emerald-300'} />
        <MiniMetric label="Samples" value={String(run.summary.count)} />
        <MiniMetric label="Time" value={`${run.summary.avg_elapsed_seconds.toFixed(2)}s`} />
        <MiniMetric label="Cache" value={runCacheLabel(run)} />
      </div>

      {failed ? (
        <div className="p-3 text-xs text-red-200/90">
          <div className="mb-1 font-medium text-red-300">Model run error</div>
          <div className="max-h-28 overflow-y-auto rounded border border-red-500/20 bg-red-500/5 p-2 font-mono text-[11px] text-red-100/80">
            {run.error || run.summary.error || 'Unknown error'}
          </div>
        </div>
      ) : (
      <div className="max-h-72 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="text-gray-500 sticky top-0 bg-[#121214]">
            <tr className="border-b border-gray-800">
              <th className="py-2 pl-3 text-left">Sample</th>
              <th className="py-2 text-right">WER</th>
              <th className="py-2 text-right">CER</th>
              <th className="py-2 pr-3 text-right">Source</th>
            </tr>
          </thead>
          <tbody>
            {previewRows.map((row) => (
              <tr key={`${row.dataset_id}-${row.index}`} className="border-b border-gray-800/60 text-gray-300">
                <td className="py-2 pl-3">
                  <div className="font-mono text-gray-400">#{String(row.index).padStart(4, '0')}</div>
                  <div className="mt-1 line-clamp-1 text-gray-200" dir="auto">{row.reference || 'No reference'}</div>
                  <div className="line-clamp-1 text-gray-500" dir="auto">{row.hypothesis || 'No hypothesis'}</div>
                </td>
                <td className="py-2 text-right font-mono">{formatPercent(row.wer)}</td>
                <td className="py-2 text-right font-mono">{formatPercent(row.cer)}</td>
                <td className="py-2 pr-3 text-right font-mono text-[10px] text-gray-500">{row.cache_hit ? 'cache' : 'run'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}
    </section>
  );
}

function MiniMetric({ label, value, color = 'text-gray-200' }: { label: string; value: string; color?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] text-gray-500 truncate">{label}</div>
      <div className={`text-sm font-mono truncate ${color}`}>{value}</div>
    </div>
  );
}

function scoreClass(value: number) {
  if (value <= 0.25) return 'text-emerald-300 font-mono';
  if (value <= 0.55) return 'text-amber-300 font-mono';
  return 'text-red-300 font-mono';
}

function modelName(models: ModelInfo[], id: string) {
  return models.find((model) => model.id === id)?.name || id;
}

function datasetName(datasets: DatasetInfo[], id: string) {
  if (id === 'mixed') return 'Mixed datasets';
  return datasets.find((dataset) => dataset.id === id)?.name || id;
}

function SelectorGroup({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-gray-500">{icon}{title}</div>
      <div className="max-h-52 overflow-y-auto border border-gray-800 rounded bg-[#0d0d0f]">{children}</div>
    </div>
  );
}

function CheckRow({ checked, label, sublabel, onChange }: { checked: boolean; label: string; sublabel: string; onChange: () => void }) {
  return (
    <label className="flex items-start gap-2 p-2.5 border-b border-gray-800/60 last:border-b-0 hover:bg-gray-800/30 cursor-pointer">
      <input type="checkbox" checked={checked} onChange={onChange} className="mt-0.5 accent-emerald-500" />
      <span className="min-w-0">
        <span className="block text-xs text-gray-200 truncate">{label}</span>
        <span className="block text-[10px] text-gray-500 truncate">{sublabel}</span>
      </span>
    </label>
  );
}

function NumberField({ label, value, onChange, min, max }: { label: string; value: string; onChange: (value: string) => void; min: number; max: number }) {
  return (
    <label className="block">
      <span className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 block">{label}</span>
      <input type="number" min={min} max={max} value={value} onChange={(event) => onChange(event.target.value)} className="w-full bg-[#1a1a1d] border border-gray-700 rounded px-2.5 h-8 text-sm text-gray-200" />
    </label>
  );
}

function SummaryCard({ label, value, color = 'text-gray-200' }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-[#121214] border border-gray-800 rounded-lg p-2.5 min-w-0">
      <div className="text-[10px] text-gray-500 mb-0.5 truncate">{label}</div>
      <div className={`text-lg font-mono truncate ${color}`}>{value}</div>
    </div>
  );
}

function ChartPanel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="bg-[#121214] border border-gray-800 rounded-lg p-4 space-y-3">
      <h3 className="text-sm font-medium text-gray-200">{title}</h3>
      {children}
    </section>
  );
}

function WordList({ title, rows }: { title: string; rows: Array<{ word: string; seen: number; accuracy: number }> }) {
  return (
    <ChartPanel title={title}>
      <div className="space-y-2 max-h-72 overflow-y-auto">
        {rows.slice(0, 10).map((item, index) => (
          <div key={`${item.word}-${index}`} className="flex items-center justify-between p-2 bg-[#0f0f11] rounded border border-gray-800">
            <span className="text-sm text-gray-200" dir="auto">{item.word}</span>
            <span className="text-xs text-gray-500">{item.seen} seen / {formatPercent(item.accuracy)}</span>
          </div>
        ))}
      </div>
    </ChartPanel>
  );
}

function SubstitutionList({ rows }: { rows: Array<{ reference: string; hypothesis: string; count: number }> }) {
  return (
    <ChartPanel title="Common Substitutions">
      <div className="space-y-2 max-h-72 overflow-y-auto">
        {rows.slice(0, 10).map((item, index) => (
          <div key={`${item.reference}-${item.hypothesis}-${index}`} className="flex items-center justify-between p-2 bg-[#0f0f11] rounded border border-gray-800">
            <span className="text-sm text-gray-300 truncate" dir="auto">{item.reference} -&gt; <span className="text-amber-400">{item.hypothesis}</span></span>
            <span className="text-xs text-gray-500 font-mono">{item.count}x</span>
          </div>
        ))}
      </div>
    </ChartPanel>
  );
}
