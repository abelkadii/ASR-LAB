import { createContext, ReactNode, useContext, useEffect, useState } from "react";
import collectedDatasetMetadata from "../../../backend/dataset_metadata.json";

export type Workspace = "home" | "workbench" | "models" | "dataset" | "label" | "analysis";

export interface ModelInfo {
  id: string;
  name: string;
  framework: string;
  min_vram_gb: number;
  benchmark?: BenchmarkStats | null;
}

export interface BenchmarkStats {
  runs: number;
  samples: number;
  avg_wer?: number | null;
  avg_cer?: number | null;
  avg_elapsed_seconds?: number | null;
  cache_hits?: number;
  cache_misses?: number;
  rank?: number | null;
  rank_total?: number | null;
  distribution?: {
    excellent?: number;
    good?: number;
    fair?: number;
    poor?: number;
    failed?: number;
  };
  dataset_scores?: Array<{
    dataset_id?: string;
    count?: number;
    avg_wer?: number | null;
    avg_cer?: number | null;
    avg_elapsed_seconds?: number | null;
    cache_hits?: number;
    cache_misses?: number;
  }>;
  best?: {
    model_id?: string;
    dataset_id?: string;
    avg_wer?: number | null;
    avg_cer?: number | null;
  } | null;
  worst?: {
    model_id?: string;
    dataset_id?: string;
    avg_wer?: number | null;
    avg_cer?: number | null;
  } | null;
}

export interface DatasetInfo {
  id: string;
  name: string;
  hf_path?: string;
  split?: string;
  audio_column?: string;
  transcript_column?: string;
  language?: string;
  task?: string;
  source_type?: string;
  license?: string;
  sample_count?: number | null;
  size_gb?: number | null;
  size_bytes?: number | null;
  hours?: number | null;
  sampling_rate?: number | null;
  status?: string;
  notes?: string;
  available?: boolean;
  downloads?: number | null;
  likes?: number | null;
  repo_file_count?: number | null;
  local_size_gb?: number | null;
  available_splits?: string[];
  split_rows?: Record<string, number>;
  metadata_collected_at?: string;
  metadata_error?: string;
  benchmark?: BenchmarkStats | null;
}

export interface SampleInfo {
  index: number;
  source_index?: number;
  sampling_rate: number;
  reference: string;
  created_at?: string;
  audio_url: string;
  dataset_id: string;
  sample_limit: number;
}

export interface DiffRow {
  type: "equal" | "accepted" | "substitution" | "insertion" | "deletion";
  reference: string;
  hypothesis: string;
}

export interface Metrics {
  wer: number;
  raw_wer?: number;
  adjusted_wer?: number;
  cer: number;
  reference: string;
  hypothesis: string;
  word_diff: DiffRow[];
  char_diff: DiffRow[];
  word_errors: DiffRow[];
  char_summary?: Record<string, number>;
  accepted_variant_count?: number;
}

interface ApiContextValue {
  models: ModelInfo[];
  datasets: DatasetInfo[];
  selectedModel: string;
  setSelectedModel: (value: string) => void;
  selectedDataset: string;
  setSelectedDataset: (value: string) => void;
  sampleLimit: number;
  setSampleLimit: (value: number) => void;
  samples: SampleInfo[];
  selectedSample: SampleInfo | null;
  setSelectedSample: (sample: SampleInfo | null) => void;
  status: string;
  setStatus: (value: string) => void;
  loadSamples: (datasetId?: string, limit?: number) => Promise<void>;
  refreshRegistries: () => Promise<void>;
}

const ApiContext = createContext<ApiContextValue | null>(null);
const datasetMetadata = collectedDatasetMetadata as {
  datasets?: Record<string, Partial<DatasetInfo> & {
    repo_size_gb?: number | null;
    repo_size_bytes?: number | null;
    local_size_bytes?: number | null;
    local_size_gb?: number | null;
    collected_at?: string;
  }>;
};

export async function requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  const data = type.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok || data?.success === false) {
    throw new Error(data?.error || data || response.statusText);
  }
  return data as T;
}

export function ApiProvider({ children }: { children: ReactNode }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedDataset, setSelectedDataset] = useState("");
  const [sampleLimit, setSampleLimit] = useState(50);
  const [samples, setSamples] = useState<SampleInfo[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleInfo | null>(null);
  const [status, setStatus] = useState("Connecting to Flask backend...");

  useEffect(() => {
    void refreshRegistries();
  }, []);

  function chooseDataset(value: string) {
    setSelectedDataset(value);
    setSamples([]);
    setSelectedSample(null);
  }

  async function refreshRegistries() {
    setStatus("Connecting to Flask backend...");

    const [modelResult, datasetResult] = await Promise.allSettled([
      requestJson<{ models: ModelInfo[]; default: string }>("/models"),
      requestJson<{ datasets: DatasetInfo[] }>("/datasets"),
    ]);

    if (modelResult.status === "fulfilled") {
      const modelData = modelResult.value;
      setModels(modelData.models || []);
      setSelectedModel((current) => current || modelData.default || modelData.models?.[0]?.id || "");
    } else {
      setModels([]);
    }

    if (datasetResult.status === "fulfilled") {
      const datasetData = datasetResult.value;
      const loadedDatasets = mergeDatasetMetadata(datasetData.datasets || []);
      setDatasets(loadedDatasets);
      setSelectedDataset((current) => {
        if (current && loadedDatasets.some((dataset) => dataset.id === current)) return current;
        return loadedDatasets.find((dataset) => dataset.id !== "recorded-audio")?.id || loadedDatasets[0]?.id || "";
      });
    } else {
      const fallbackDatasets = mergeDatasetMetadata([]);
      setDatasets(fallbackDatasets);
      setSelectedDataset((current) => {
        if (current && fallbackDatasets.some((dataset) => dataset.id === current)) return current;
        return fallbackDatasets.find((dataset) => dataset.id !== "recorded-audio")?.id || fallbackDatasets[0]?.id || "";
      });
    }

    const errors = [
      modelResult.status === "rejected" ? `models: ${modelResult.reason?.message || modelResult.reason}` : "",
      datasetResult.status === "rejected" ? `datasets: ${datasetResult.reason?.message || datasetResult.reason}` : "",
    ].filter(Boolean);

    setStatus(errors.length ? `Backend issue loading ${errors.join("; ")}` : "Backend connected");
  }

  async function loadSamples(datasetId = selectedDataset, limit = sampleLimit) {
    if (!datasetId) return;
    setStatus(`Loading samples from ${datasetId}...`);
    const data = await requestJson<{ samples: Omit<SampleInfo, "dataset_id" | "sample_limit">[] }>(
      `/dataset-samples?dataset_id=${encodeURIComponent(datasetId)}&limit=${encodeURIComponent(limit)}`,
    );
    const loaded = (data.samples || []).map((sample) => ({
      ...sample,
      dataset_id: datasetId,
      sample_limit: limit,
    }));
    setSamples(loaded);
    setSelectedSample(loaded[0] || null);
    setStatus(`Loaded ${loaded.length} sample(s)`);
  }

  return (
    <ApiContext.Provider
      value={{
        models,
        datasets,
        selectedModel,
        setSelectedModel,
        selectedDataset,
        setSelectedDataset: chooseDataset,
        sampleLimit,
        setSampleLimit,
        samples,
        selectedSample,
        setSelectedSample,
        status,
        setStatus,
        loadSamples,
        refreshRegistries,
      }}
    >
      {children}
    </ApiContext.Provider>
  );
}

function mergeDatasetMetadata(datasets: DatasetInfo[]) {
  const metadataById = datasetMetadata.datasets || {};
  const seen = new Set<string>();
  const merged = datasets.map((dataset) => {
    seen.add(dataset.id);
    return mergeDataset(dataset, metadataById[dataset.id]);
  });

  for (const [id, metadata] of Object.entries(metadataById)) {
    if (!seen.has(id)) {
      merged.push(mergeDataset({ id, name: metadata.name || id }, metadata));
    }
  }

  return merged;
}

function mergeDataset(dataset: DatasetInfo, metadata?: Partial<DatasetInfo> & {
  repo_size_gb?: number | null;
  repo_size_bytes?: number | null;
  local_size_bytes?: number | null;
  local_size_gb?: number | null;
  collected_at?: string;
}) {
  if (!metadata) return dataset;
  const isRemote = Boolean(metadata.hf_path || dataset.hf_path);
  return {
    ...metadata,
    ...dataset,
    hf_path: usefulString(dataset.hf_path) || metadata.hf_path,
    split: usefulString(dataset.split) || metadata.split,
    audio_column: usefulString(dataset.audio_column) || metadata.audio_column || "audio",
    transcript_column: usefulString(dataset.transcript_column) || metadata.transcript_column || "reference",
    source_type: usefulString(dataset.source_type) || (isRemote ? "huggingface" : "local"),
    status: usefulString(dataset.status) || "ready",
    language: usefulString(dataset.language) || (isRemote ? "Moroccan Darija" : "User provided"),
    task: usefulString(dataset.task) || "ASR",
    sample_count: numberOr(dataset.sample_count, metadata.sample_count),
    size_gb: numberOr(dataset.size_gb, metadata.size_gb, metadata.repo_size_gb),
    size_bytes: numberOr(dataset.size_bytes, metadata.size_bytes, metadata.repo_size_bytes),
    local_size_gb: numberOr(dataset.local_size_gb, metadata.local_size_gb),
    downloads: numberOr(dataset.downloads, metadata.downloads),
    likes: numberOr(dataset.likes, metadata.likes),
    repo_file_count: numberOr(dataset.repo_file_count, metadata.repo_file_count),
    available_splits: dataset.available_splits || metadata.available_splits,
    split_rows: dataset.split_rows || metadata.split_rows,
    metadata_collected_at: dataset.metadata_collected_at || metadata.metadata_collected_at || metadata.collected_at,
    metadata_error: dataset.metadata_error || metadata.metadata_error,
    license: usefulString(dataset.license) || usefulString(metadata.license) || (isRemote ? "Dataset card" : "User provided"),
  } satisfies DatasetInfo;
}

function usefulString(value?: string | null) {
  if (!value || value === "Unknown") return undefined;
  return value;
}

function numberOr(...values: Array<number | null | undefined>) {
  return values.find((value) => typeof value === "number") ?? null;
}

export function useApi() {
  const context = useContext(ApiContext);
  if (!context) throw new Error("useApi must be used inside ApiProvider");
  return context;
}

export function formatPercent(value?: number) {
  return `${((value || 0) * 100).toFixed(1)}%`;
}
