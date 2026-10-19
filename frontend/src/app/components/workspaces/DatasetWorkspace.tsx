import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Copy,
  Database,
  Download,
  Loader2,
  MoreVertical,
  Music2,
  Play,
  Plus,
  RefreshCw,
  Tag,
  X,
} from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { DatasetInfo, SampleInfo, Workspace, requestJson, useApi } from '../../api';
import { AudioPlayer } from '../AudioPlayer';
import { DetailedWaveform } from '../DetailedWaveform';
import { Spectrogram } from '../Spectrogram';

interface DatasetWorkspaceProps {
  onWorkspaceChange: (workspace: Workspace) => void;
}

export function DatasetWorkspace({ onWorkspaceChange }: DatasetWorkspaceProps) {
  const {
    datasets,
    selectedDataset,
    setSelectedDataset,
    sampleLimit,
    setSampleLimit,
    samples,
    selectedSample,
    setSelectedSample,
    loadSamples,
    setStatus,
    refreshRegistries,
  } = useApi();

  const [isLoading, setIsLoading] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newDatasetName, setNewDatasetName] = useState('');
  const [newDatasetNotes, setNewDatasetNotes] = useState('');
  const [showAdvancedInfo, setShowAdvancedInfo] = useState(false);
  const [openSampleMenu, setOpenSampleMenu] = useState<string | null>(null);
  const [copySample, setCopySample] = useState<SampleInfo | null>(null);
  const [copyTargetId, setCopyTargetId] = useState('');
  const [isCopying, setIsCopying] = useState(false);

  const selectedDatasetInfo = datasets.find((dataset) => dataset.id === selectedDataset);
  const customDatasets = datasets.filter(isCustomDataset);
  const publicDatasets = datasets.filter((dataset) => !isCustomDataset(dataset));
  const loadedContext = totalRows(selectedDatasetInfo);

  const referenceLengthData = useMemo(() => {
    const buckets = [
      { range: '0-4w', count: 0 },
      { range: '5-8w', count: 0 },
      { range: '9-12w', count: 0 },
      { range: '13-20w', count: 0 },
      { range: '21w+', count: 0 },
    ];

    for (const sample of samples) {
      const words = wordCount(sample.reference);
      if (words <= 4) buckets[0].count += 1;
      else if (words <= 8) buckets[1].count += 1;
      else if (words <= 12) buckets[2].count += 1;
      else if (words <= 20) buckets[3].count += 1;
      else buckets[4].count += 1;
    }

    return buckets;
  }, [samples]);

  async function loadSelected(limit = sampleLimit) {
    if (!selectedDataset) return;
    setIsLoading(true);
    try {
      await loadSamples(selectedDataset, limit);
    } finally {
      setIsLoading(false);
    }
  }

  function setActiveSample(sample: SampleInfo) {
    setSelectedSample(sample);
    setStatus(`Opening sample #${sample.index + 1} from ${sample.dataset_id} in Workbench.`);
    onWorkspaceChange('workbench');
  }

  async function createCustomDataset() {
    if (!newDatasetName.trim()) {
      setStatus('Enter a custom dataset name first.');
      return;
    }
    const data = await requestJson<{ success: boolean; dataset: DatasetInfo }>('/custom-datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newDatasetName.trim(), notes: newDatasetNotes.trim() }),
    });
    await refreshRegistries();
    setSelectedDataset(data.dataset.id);
    setCopyTargetId(data.dataset.id);
    setNewDatasetName('');
    setNewDatasetNotes('');
    setShowCreateModal(false);
    setStatus(`Created custom dataset: ${data.dataset.name}`);
  }

  async function copyToDataset(sample = copySample, targetId = copyTargetId) {
    if (!sample || !targetId) {
      setStatus('Choose a custom dataset before copying the sample.');
      return;
    }
    setIsCopying(true);
    try {
      await requestJson('/label-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: sample.dataset_id,
          sample_index: sample.index,
          sample_limit: sample.sample_limit,
          label: sample.reference || '',
          notes: `Copied from ${sample.dataset_id} #${sample.index + 1}`,
          target_dataset_id: targetId,
        }),
      });
      await refreshRegistries();
      setStatus(`Copied sample #${sample.index + 1} to ${datasetName(datasets, targetId)}.`);
      setCopySample(null);
      setOpenSampleMenu(null);
    } finally {
      setIsCopying(false);
    }
  }

  function openCopy(sample: SampleInfo) {
    setCopySample(sample);
    setCopyTargetId(copyTargetId || customDatasets[0]?.id || '');
    setOpenSampleMenu(null);
  }

  function openLabel(sample: SampleInfo) {
    setActiveSample(sample);
    onWorkspaceChange('label');
  }

  function openBenchmarkForDataset(dataset?: DatasetInfo) {
    if (!dataset) {
      setStatus('Select a dataset before opening benchmark setup.');
      return;
    }
    setSelectedDataset(dataset.id);
    setStatus(`Benchmark setup opened with selected dataset: ${dataset.name}.`);
    onWorkspaceChange('analysis');
  }

  return (
    <div className="h-full grid grid-cols-[300px_minmax(360px,460px)_1fr] divide-x divide-gray-800 bg-[#0a0a0b]">
      <aside className="flex flex-col overflow-hidden bg-[#0f0f11]">
        <div className="h-12 border-b border-gray-800 px-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-medium text-gray-200">Datasets</h2>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="h-7 px-2 rounded bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25 transition-colors text-xs font-medium flex items-center gap-1.5"
          >
            <Plus className="w-3.5 h-3.5" />
            New
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-4">
          <RegistrySection title="Custom Datasets" empty="No custom datasets yet.">
            {customDatasets.map((dataset) => (
              <DatasetButton
                key={dataset.id}
                dataset={dataset}
                active={selectedDataset === dataset.id}
                onClick={() => setSelectedDataset(dataset.id)}
              />
            ))}
          </RegistrySection>

          <RegistrySection title="Public Datasets" empty="No public datasets registered.">
            {publicDatasets.map((dataset) => (
              <DatasetButton
                key={dataset.id}
                dataset={dataset}
                active={selectedDataset === dataset.id}
                onClick={() => setSelectedDataset(dataset.id)}
              />
            ))}
          </RegistrySection>
        </div>
      </aside>

      <section className="flex flex-col overflow-hidden bg-[#101012]">
        <div className="border-b border-gray-800 p-3 space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 className="text-base font-medium text-gray-200 truncate">
                {selectedDatasetInfo?.name || 'Select a dataset'}
              </h2>
              <p className="text-[11px] text-gray-500 font-mono truncate">
                {selectedDatasetInfo?.hf_path || selectedDatasetInfo?.id || 'No dataset selected'}
              </p>
            </div>
            <DatasetKindBadge dataset={selectedDatasetInfo} />
          </div>

          <div className="grid grid-cols-[1fr_auto_auto] gap-2">
            <input
              aria-label="Sample limit"
              type="number"
              min={1}
              max={200}
              value={sampleLimit}
              onChange={(event) => setSampleLimit(Number(event.target.value || 50))}
              className="bg-[#1a1a1d] border border-gray-700 rounded px-2.5 h-8 text-sm text-gray-200"
            />
            <button
              onClick={() => void loadSelected()}
              disabled={!selectedDataset || isLoading}
              className="h-8 px-3 bg-emerald-500/20 text-emerald-400 rounded text-xs font-medium hover:bg-emerald-500/30 disabled:opacity-50 transition-colors flex items-center gap-1.5"
            >
              {isLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
              Load
            </button>
            <button
              onClick={() => void loadSelected(200)}
              disabled={!selectedDataset || isLoading}
              className="h-8 px-3 bg-gray-800 text-gray-300 rounded text-xs font-medium hover:bg-gray-700 disabled:opacity-50 transition-colors"
            >
              Max 200
            </button>
          </div>
        </div>

        <div className="h-10 border-b border-gray-800 px-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Music2 className="w-4 h-4 text-gray-500" />
            <h3 className="text-sm font-medium text-gray-200">Samples</h3>
          </div>
          <span className="text-[10px] text-gray-500">
            {loadedContext ? `Showing ${samples.length} / ${loadedContext.toLocaleString()}` : `${samples.length} loaded`}
          </span>
        </div>

        <div className="flex-1 overflow-y-auto">
          {samples.length === 0 ? (
            <div className="h-full flex items-center justify-center p-6 text-center">
              <div className="max-w-64">
                <div className="text-sm text-gray-300 mb-1">No samples loaded</div>
                <div className="text-xs text-gray-500 leading-relaxed">
                  Choose a dataset and load a small batch to inspect audio, transcripts, and copy candidates.
                </div>
              </div>
            </div>
          ) : (
            samples.map((sample) => {
              const active = selectedSample?.index === sample.index && selectedSample?.dataset_id === sample.dataset_id;
              const key = `${sample.dataset_id}-${sample.index}`;
              return (
                <SampleCard
                  key={key}
                  sample={sample}
                  active={active}
                  menuOpen={openSampleMenu === key}
                  onPreview={() => setSelectedSample(sample)}
                  onSetActive={() => setActiveSample(sample)}
                  onToggleMenu={() => setOpenSampleMenu(openSampleMenu === key ? null : key)}
                  onCopy={() => openCopy(sample)}
                  onLabel={() => openLabel(sample)}
                />
              );
            })
          )}
        </div>
      </section>

      <main className="flex flex-col overflow-hidden">
        <div className="h-12 bg-[#121214] border-b border-gray-800 flex items-center px-4 gap-3">
          <span className="text-xs text-gray-500">Active Sample</span>
          <span className="text-sm font-mono text-emerald-400">
            {selectedSample ? `#${String(selectedSample.index + 1).padStart(4, '0')}` : 'None'}
          </span>
          <div className="h-4 w-px bg-gray-800" />
          <span className="text-xs text-gray-400 truncate">Dataset: {selectedDatasetInfo?.name || selectedDataset || '-'}</span>
          <div className="ml-auto flex items-center gap-2">
            {selectedSample && (
              <>
                <button
                  onClick={() => setActiveSample(selectedSample)}
                  className="h-7 px-2 text-xs text-emerald-400 hover:text-emerald-300 hover:bg-emerald-500/10 rounded transition-colors flex items-center gap-1"
                >
                  <CheckCircle2 className="w-3 h-3" />
                  Transcribe Sample
                </button>
                <button
                  onClick={() => openCopy(selectedSample)}
                  className="h-7 px-2 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded transition-colors flex items-center gap-1"
                >
                  <Copy className="w-3 h-3" />
                  Copy To Dataset
                </button>
              </>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <section className="bg-[#121214] border border-gray-800 rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-200">Audio Preview</h3>
              {selectedSample && <DatasetKindBadge dataset={selectedDatasetInfo} />}
            </div>

            <AudioPlayer
              src={selectedSample?.audio_url}
              title={selectedSample ? `Sample #${String(selectedSample.index + 1).padStart(4, '0')}` : 'Select a sample to preview'}
              subtitle={selectedDatasetInfo?.name || selectedDataset || undefined}
              downloadName={`${selectedSample?.dataset_id || 'sample'}-${selectedSample?.index ?? 0}.wav`}
            />
            <DetailedWaveform audioUrl={selectedSample?.audio_url} height={130} />
            <Spectrogram audioUrl={selectedSample?.audio_url} height={140} />

            <div className="bg-[#0f0f11] rounded border border-gray-800 p-4">
              <div className="mb-2 text-xs uppercase tracking-wider text-gray-500">Transcript</div>
              <p className="text-lg leading-relaxed text-gray-100" dir="auto">
                {selectedSample?.reference || 'Select a sample to inspect its transcript.'}
              </p>
            </div>
          </section>

          <section className="rounded-lg border border-blue-500/25 bg-blue-500/10 p-4">
            <div className="flex items-center justify-between gap-4">
              <div className="min-w-0">
                <div className="text-xs font-semibold uppercase tracking-wider text-blue-300">Test Dataset</div>
                <h2 className="mt-1 truncate text-lg font-semibold text-gray-100">
                  {selectedDatasetInfo?.name || 'Select a dataset'}
                </h2>
                <p className="mt-1 text-sm text-gray-400">
                  Open Analysis with this dataset already selected, then choose one or more models to benchmark on it.
                </p>
              </div>
              <button
                onClick={() => openBenchmarkForDataset(selectedDatasetInfo)}
                disabled={!selectedDatasetInfo}
                className="h-10 shrink-0 rounded bg-blue-500 px-4 text-sm font-medium text-white transition-colors hover:bg-blue-600 disabled:opacity-50 flex items-center gap-2"
              >
                <Play className="h-4 w-4" />
                Benchmark This Dataset
              </button>
            </div>
          </section>

          <section className="bg-[#121214] border border-gray-800 rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-base text-gray-200 font-medium">Dataset Overview</h2>
                <p className="text-xs text-gray-500 mt-1 truncate">
                  {selectedDatasetInfo?.hf_path || selectedDatasetInfo?.id || 'No dataset selected'}
                </p>
              </div>
              <button
                onClick={() => setShowAdvancedInfo(!showAdvancedInfo)}
                className="h-8 px-2 rounded text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors flex items-center gap-1.5"
              >
                {showAdvancedInfo ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                {showAdvancedInfo ? 'Show Less' : 'Show More'}
              </button>
            </div>

            <div className="grid grid-cols-5 gap-3">
              <DatasetStat label="Rows" value={formatCount(selectedDatasetInfo?.sample_count)} />
              <DatasetStat label="Size" value={formatSize(selectedDatasetInfo)} />
              <DatasetStat label="Language" value={selectedDatasetInfo?.language || 'Darija'} />
              <DatasetStat label="Task" value={selectedDatasetInfo?.task || 'ASR'} />
              <DatasetStat label="Source" value={displaySource(selectedDatasetInfo)} />
            </div>

            <div className="grid grid-cols-4 gap-3">
              <DatasetStat label="Benchmark WER" value={formatBenchmarkPercent(selectedDatasetInfo?.benchmark?.avg_wer)} />
              <DatasetStat label="Benchmark CER" value={formatBenchmarkPercent(selectedDatasetInfo?.benchmark?.avg_cer)} />
              <DatasetStat label="Avg Speed" value={formatBenchmarkSpeed(selectedDatasetInfo?.benchmark?.avg_elapsed_seconds)} />
              <DatasetStat label="Bench Samples" value={formatCount(selectedDatasetInfo?.benchmark?.samples)} />
            </div>

            {showAdvancedInfo && (
              <>
                <div className="grid grid-cols-4 gap-3">
                  <DatasetStat label="Loaded" value={`${samples.length.toLocaleString()}`} />
                  <DatasetStat label="Cached" value={formatLocalCache(selectedDatasetInfo)} />
                  <DatasetStat label="Downloads" value={formatNumber(selectedDatasetInfo?.downloads)} />
                  <DatasetStat label="Files" value={formatNumber(selectedDatasetInfo?.repo_file_count)} />
                </div>

                <div className="grid grid-cols-3 gap-3 text-xs">
                  <InfoRow label="Sample Rate" value={formatRate(selectedDatasetInfo)} />
                  <InfoRow label="Splits" value={formatAvailableSplits(selectedDatasetInfo)} />
                  <InfoRow label="Split Rows" value={formatSplitRows(selectedDatasetInfo)} />
                  <InfoRow label="License" value={formatLicense(selectedDatasetInfo)} />
                  <InfoRow label="Audio Column" value={selectedDatasetInfo?.audio_column || 'audio'} />
                  <InfoRow label="Transcript Column" value={selectedDatasetInfo?.transcript_column || 'reference'} />
                  <InfoRow label="Collected" value={formatCollected(selectedDatasetInfo?.metadata_collected_at)} />
                  <InfoRow label="Metadata" value={selectedDatasetInfo?.metadata_error ? 'Partial catalog' : 'Catalog synced'} />
                  <InfoRow label="Purpose" value={isCustomDataset(selectedDatasetInfo) ? 'Editable custom dataset' : 'Read-only public source'} />
                  <InfoRow label="Best Model" value={selectedDatasetInfo?.benchmark?.best?.model_id || '-'} />
                  <InfoRow label="Worst Model" value={selectedDatasetInfo?.benchmark?.worst?.model_id || '-'} />
                  <InfoRow label="Benchmark Cache" value={formatBenchmarkCache(selectedDatasetInfo?.benchmark)} />
                </div>

                {selectedDatasetInfo?.notes && (
                  <div className="bg-[#0f0f11] border border-gray-800 rounded p-3 text-xs text-gray-400 leading-relaxed">
                    {selectedDatasetInfo.notes}
                  </div>
                )}
              </>
            )}
          </section>

          <section className="bg-[#121214] border border-gray-800 rounded-lg p-4 space-y-3">
            <h3 className="text-sm font-medium text-gray-200">Transcript Lengths</h3>
            <ResponsiveContainer width="100%" height={170}>
              <BarChart data={referenceLengthData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                <XAxis dataKey="range" tick={{ fill: 'var(--chart-tick)', fontSize: 11 }} stroke="var(--chart-axis)" />
                <YAxis allowDecimals={false} tick={{ fill: 'var(--chart-tick)', fontSize: 11 }} stroke="var(--chart-axis)" />
                <Tooltip contentStyle={{ backgroundColor: 'var(--chart-tooltip-bg)', border: '1px solid var(--chart-tooltip-border)', borderRadius: '6px', color: 'var(--chart-tooltip-text)' }} />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </section>
        </div>
      </main>

      {showCreateModal && (
        <Modal title="Create Custom Dataset" onClose={() => setShowCreateModal(false)}>
          <div className="space-y-3">
            <label className="block">
              <span className="text-xs text-gray-400">Name</span>
              <input
                value={newDatasetName}
                onChange={(event) => setNewDatasetName(event.target.value)}
                placeholder="Darija Training Set v1"
                className="mt-1 h-9 w-full rounded border border-gray-700 bg-[#0a0a0b] px-3 text-sm text-gray-100 outline-none focus:border-emerald-500"
              />
            </label>
            <label className="block">
              <span className="text-xs text-gray-400">Description</span>
              <textarea
                value={newDatasetNotes}
                onChange={(event) => setNewDatasetNotes(event.target.value)}
                placeholder="Domain, speakers, source policy..."
                rows={4}
                className="mt-1 w-full rounded border border-gray-700 bg-[#0a0a0b] px-3 py-2 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-emerald-500"
              />
            </label>
            <button
              onClick={() => void createCustomDataset()}
              className="h-9 w-full rounded bg-emerald-500 text-sm font-medium text-white hover:bg-emerald-600 transition-colors"
            >
              Create
            </button>
          </div>
        </Modal>
      )}

      {copySample && (
        <Modal title="Copy Sample To Dataset" onClose={() => setCopySample(null)}>
          <div className="space-y-3">
            <div className="rounded border border-gray-800 bg-[#0f0f11] p-3">
              <div className="text-xs text-gray-500">Sample</div>
              <div className="mt-1 font-mono text-sm text-emerald-400">#{String(copySample.index + 1).padStart(4, '0')}</div>
              <p className="mt-2 line-clamp-3 text-sm text-gray-300" dir="auto">{copySample.reference || 'No transcript available'}</p>
            </div>
            <label className="block">
              <span className="text-xs text-gray-400">Custom Dataset</span>
              <select
                value={copyTargetId}
                onChange={(event) => setCopyTargetId(event.target.value)}
                className="mt-1 h-9 w-full rounded border border-gray-700 bg-[#0a0a0b] px-3 text-sm text-gray-100 outline-none focus:border-emerald-500"
              >
                <option value="">Choose target...</option>
                {customDatasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                ))}
              </select>
            </label>
            {customDatasets.length === 0 && (
              <button
                onClick={() => {
                  setCopySample(null);
                  setShowCreateModal(true);
                }}
                className="h-9 w-full rounded border border-emerald-500/30 bg-emerald-500/10 text-sm font-medium text-emerald-400 hover:bg-emerald-500/20 transition-colors"
              >
                Create Custom Dataset First
              </button>
            )}
            <button
              onClick={() => void copyToDataset()}
              disabled={!copyTargetId || isCopying}
              className="h-9 w-full rounded bg-emerald-500 text-sm font-medium text-white hover:bg-emerald-600 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
              {isCopying && <Loader2 className="w-4 h-4 animate-spin" />}
              Copy Sample
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function RegistrySection({ title, empty, children }: { title: string; empty: string; children: ReactNode }) {
  const hasChildren = Array.isArray(children) ? children.length > 0 : Boolean(children);
  return (
    <section className="space-y-2">
      <div className="px-1 text-[10px] font-semibold uppercase tracking-wider text-gray-500">{title}</div>
      {hasChildren ? children : <div className="rounded border border-dashed border-gray-800 p-3 text-xs text-gray-600">{empty}</div>}
    </section>
  );
}

function DatasetButton({ dataset, active, onClick }: { dataset: DatasetInfo; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-lg border p-3 transition-colors ${
        active ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-[#121214] border-gray-800 hover:bg-gray-800/40'
      }`}
      title={dataset.name}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-sm text-gray-200 font-medium truncate">{dataset.name}</div>
          <div className="text-[10px] text-gray-500 font-mono truncate">{dataset.id}</div>
        </div>
        <DatasetKindBadge dataset={dataset} />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2">
        <MiniInfo label="Rows" value={formatCount(dataset.sample_count)} />
        <MiniInfo label="Size" value={formatSize(dataset)} />
        <MiniInfo label="Source" value={displaySource(dataset)} />
      </div>
    </button>
  );
}

function SampleCard({
  sample,
  active,
  menuOpen,
  onPreview,
  onSetActive,
  onToggleMenu,
  onCopy,
  onLabel,
}: {
  sample: SampleInfo;
  active: boolean;
  menuOpen: boolean;
  onPreview: () => void;
  onSetActive: () => void;
  onToggleMenu: () => void;
  onCopy: () => void;
  onLabel: () => void;
}) {
  return (
    <div className={`relative border-b border-gray-800/50 px-3 py-3 transition-colors hover:bg-gray-800/30 ${active ? 'bg-gray-800/60' : ''}`}>
      <button onClick={onPreview} className="w-full text-left">
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <span className="text-gray-500 font-mono text-xs">#{String(sample.index + 1).padStart(4, '0')}</span>
            {active && (
              <span className="flex items-center gap-1 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-400">
                <CheckCircle2 className="w-3 h-3" />
                Active Sample
              </span>
            )}
          </div>
          <span className="text-gray-500 font-mono text-xs">{sample.sampling_rate || 16000}Hz</span>
        </div>
        <p className="text-sm text-gray-300 leading-relaxed line-clamp-2" dir="auto">
          {sample.reference || 'Reference unavailable'}
        </p>
      </button>
      <div className="mt-3 flex items-center justify-between">
        <button
          onClick={onSetActive}
          className="h-7 px-2.5 rounded bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25 transition-colors text-xs font-medium flex items-center gap-1.5"
        >
          <CheckCircle2 className="w-3 h-3" />
          Use In Session
        </button>
        <button
          onClick={onToggleMenu}
          className="h-7 w-7 rounded text-gray-500 hover:text-gray-200 hover:bg-gray-800 transition-colors flex items-center justify-center"
          aria-label="Sample actions"
        >
          <MoreVertical className="w-4 h-4" />
        </button>
      </div>
      {menuOpen && (
        <div className="absolute right-3 top-10 z-20 w-44 overflow-hidden rounded border border-gray-800 bg-[#121214] shadow-xl">
          <MenuButton icon={<CheckCircle2 className="w-3.5 h-3.5" />} label="Use In Session" onClick={onSetActive} />
          <MenuButton icon={<Copy className="w-3.5 h-3.5" />} label="Copy To Dataset" onClick={onCopy} />
          <MenuButton icon={<Tag className="w-3.5 h-3.5" />} label="Open In Label" onClick={onLabel} />
          <a
            href={sample.audio_url}
            download={`${sample.dataset_id}-${sample.index}.wav`}
            className="flex h-8 items-center gap-2 px-3 text-xs text-gray-300 hover:bg-gray-800"
          >
            <Download className="w-3.5 h-3.5" />
            Download Audio
          </a>
        </div>
      )}
    </div>
  );
}

function MenuButton({ icon, label, onClick }: { icon: ReactNode; label: string; onClick: () => void }) {
  return (
    <button onClick={onClick} className="flex h-8 w-full items-center gap-2 px-3 text-left text-xs text-gray-300 hover:bg-gray-800">
      {icon}
      {label}
    </button>
  );
}

function Modal({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-md rounded-lg border border-gray-800 bg-[#121214] shadow-2xl">
        <div className="flex h-12 items-center justify-between border-b border-gray-800 px-4">
          <h2 className="text-sm font-semibold text-gray-100">{title}</h2>
          <button onClick={onClose} className="h-7 w-7 rounded text-gray-500 hover:bg-gray-800 hover:text-gray-200 flex items-center justify-center">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

function wordCount(value: string) {
  return value.trim().split(/\s+/).filter(Boolean).length;
}

function isCustomDataset(dataset?: DatasetInfo) {
  return Boolean(dataset && (dataset.id.startsWith('custom-') || dataset.source_type === 'local' || dataset.id === 'recorded-audio'));
}

function totalRows(dataset?: DatasetInfo) {
  if (typeof dataset?.sample_count === 'number') return dataset.sample_count;
  return null;
}

function datasetName(datasets: DatasetInfo[], id: string) {
  return datasets.find((dataset) => dataset.id === id)?.name || id;
}

function formatCount(value?: number | null) {
  return typeof value === 'number' ? value.toLocaleString() : 'Not listed';
}

function formatNumber(value?: number | null) {
  return typeof value === 'number' ? value.toLocaleString() : '-';
}

function formatSize(dataset?: DatasetInfo) {
  if (typeof dataset?.size_gb === 'number') {
    return `${dataset.size_gb.toFixed(dataset.size_gb < 1 ? 3 : 2)} GB`;
  }
  return dataset?.source_type === 'local' ? '0 GB' : 'Catalog only';
}

function formatLocalCache(dataset?: DatasetInfo) {
  if (typeof dataset?.local_size_gb === 'number') {
    return `${dataset.local_size_gb.toFixed(dataset.local_size_gb < 1 ? 3 : 2)} GB`;
  }
  return '0 GB';
}

function formatRate(dataset?: DatasetInfo) {
  if (dataset?.sampling_rate) return `${dataset.sampling_rate}Hz`;
  return 'Native';
}

function formatAvailableSplits(dataset?: DatasetInfo) {
  return dataset?.available_splits?.length ? dataset.available_splits.join(', ') : dataset?.split || 'Local';
}

function formatSplitRows(dataset?: DatasetInfo) {
  const rows = dataset?.split_rows;
  if (!rows || Object.keys(rows).length === 0) return 'Unavailable';
  return Object.entries(rows)
    .map(([split, count]) => `${split}: ${count.toLocaleString()}`)
    .join(' / ');
}

function formatCollected(value?: string) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toLocaleDateString();
}

function formatLicense(dataset?: DatasetInfo) {
  if (!dataset?.license || dataset.license === 'Unknown') return dataset?.source_type === 'local' ? 'User provided' : 'Dataset card';
  return dataset.license;
}

function formatBenchmarkPercent(value?: number | null) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'Not run';
  return `${(value * 100).toFixed(1)}%`;
}

function formatBenchmarkSpeed(value?: number | null) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'Pending';
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  return `${value.toFixed(2)} s`;
}

function formatBenchmarkCache(stats?: DatasetInfo['benchmark']) {
  if (!stats) return '-';
  const hits = stats.cache_hits || 0;
  const misses = stats.cache_misses || 0;
  return `${hits}/${hits + misses}`;
}

function displaySource(dataset?: DatasetInfo) {
  const value = (dataset?.source_type || '').toLowerCase();
  if (value === 'huggingface') return 'Hugging Face';
  if (value === 'local') return 'Local';
  return value ? value : 'Local';
}

function MiniInfo({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[9px] uppercase text-gray-600">{label}</div>
      <div className="text-[11px] text-gray-300 truncate" title={value}>{value}</div>
    </div>
  );
}

function DatasetStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[#0f0f11] rounded border border-gray-800 p-3 min-w-0">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-sm text-gray-200 font-medium truncate" title={value}>{value}</div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[#0f0f11] border border-gray-800 rounded p-2 min-w-0">
      <div className="text-[10px] text-gray-500 mb-0.5">{label}</div>
      <div className="text-gray-200 truncate" title={value}>{value}</div>
    </div>
  );
}

function DatasetKindBadge({ dataset }: { dataset?: DatasetInfo }) {
  const custom = isCustomDataset(dataset);
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap ${custom ? 'bg-blue-500/15 text-blue-400' : 'bg-emerald-500/15 text-emerald-400'}`}>
      {custom ? 'Custom' : 'Public'}
    </span>
  );
}
