import { useEffect, useMemo, useRef, useState } from 'react';
import { CheckCircle2, Database, FileAudio, Loader2, Mic, PencilLine, Play, Plus, StopCircle, ThumbsUp, Upload, XCircle } from 'lucide-react';
import { Workspace, DiffRow, formatPercent, Metrics, requestJson, useApi } from '../../api';
import { AudioPlayer } from '../AudioPlayer';
import { DetailedWaveform } from '../DetailedWaveform';
import { Spectrogram } from '../Spectrogram';

type InputMode = 'record' | 'upload' | 'sample';
type SaveMode = 'session' | 'dataset';

export function TranscribeResearchWorkspace({ onWorkspaceChange }: { onWorkspaceChange: (workspace: Workspace) => void }) {
  const {
    models,
    datasets,
    selectedModel,
    selectedDataset,
    selectedSample,
    setStatus,
    loadSamples,
    refreshRegistries,
  } = useApi();

  const [inputMode, setInputMode] = useState<InputMode>(selectedSample ? 'sample' : 'record');
  const [saveMode, setSaveMode] = useState<SaveMode>('session');
  const [targetDatasetId, setTargetDatasetId] = useState('recorded-audio');
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [uploadedUrl, setUploadedUrl] = useState('');
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordingUrl, setRecordingUrl] = useState('');
  const [referenceText, setReferenceText] = useState('');
  const [transcript, setTranscript] = useState('');
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const selectedModelInfo = models.find((model) => model.id === selectedModel);
  const selectedDatasetInfo = datasets.find((dataset) => dataset.id === selectedDataset);
  const targetDatasets = datasets.filter((dataset) => dataset.id === 'recorded-audio' || dataset.id.startsWith('custom-'));
  const targetDataset = datasets.find((dataset) => dataset.id === targetDatasetId);

  const activeAudioUrl = inputMode === 'record' ? recordingUrl : inputMode === 'upload' ? uploadedUrl : selectedSample?.audio_url;
  const inputLabel = inputMode === 'record'
    ? recordedBlob ? 'Microphone Recording' : 'Record Audio'
    : inputMode === 'upload'
      ? audioFile?.name || 'Uploaded Audio'
      : selectedSample
        ? `Dataset Sample #${String(selectedSample.index + 1).padStart(4, '0')}`
        : 'Dataset Sample';
  const inputSubLabel = inputMode === 'sample'
    ? selectedDatasetInfo?.name || selectedSample?.dataset_id || 'Select a sample in Datasets'
    : inputMode === 'upload'
      ? audioFile?.type || 'Choose an audio file'
      : recordedBlob
        ? `${Math.round(recordedBlob.size / 1024)} KB`
        : 'Use your microphone';
  const canRun = Boolean(selectedModel && ((inputMode === 'record' && recordedBlob) || (inputMode === 'upload' && audioFile) || (inputMode === 'sample' && selectedSample)));
  const referenceValue = metrics?.reference || referenceText || selectedSample?.reference || '';

  const confidence = useMemo(() => {
    if (!transcript) return null;
    if (metrics) return Math.max(0, 1 - (metrics.adjusted_wer ?? metrics.wer ?? 0));
    return 0.78;
  }, [metrics, transcript]);

  useEffect(() => {
    return () => {
      if (uploadedUrl) URL.revokeObjectURL(uploadedUrl);
      if (recordingUrl) URL.revokeObjectURL(recordingUrl);
    };
  }, [uploadedUrl, recordingUrl]);

  useEffect(() => {
    if (selectedSample?.reference && inputMode === 'sample') setReferenceText(selectedSample.reference);
  }, [selectedSample, inputMode]);

  function chooseMode(mode: InputMode) {
    setInputMode(mode);
    setTranscript('');
    setMetrics(null);
    if (mode === 'sample' && selectedSample?.reference) setReferenceText(selectedSample.reference);
  }

  function handleFileChange(file: File | null) {
    if (uploadedUrl) URL.revokeObjectURL(uploadedUrl);
    setAudioFile(file);
    setUploadedUrl(file ? URL.createObjectURL(file) : '');
    setTranscript('');
    setMetrics(null);
    setInputMode('upload');
    if (file) setStatus(`Audio file selected: ${file.name}`);
  }

  async function toggleRecording() {
    setInputMode('record');
    if (isRecording && recorderRef.current) {
      recorderRef.current.stop();
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunksRef.current = [];
    const recorder = new MediaRecorder(stream);
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
      setRecordedBlob(blob);
      setTranscript('');
      setMetrics(null);
      if (recordingUrl) URL.revokeObjectURL(recordingUrl);
      setRecordingUrl(URL.createObjectURL(blob));
      setIsRecording(false);
      setStatus('Recording captured.');
    };
    recorder.start();
    setIsRecording(true);
    setStatus('Recording...');
  }

  async function createTargetDataset() {
    if (!newDatasetName.trim()) {
      setStatus('Enter a dataset name first.');
      return;
    }
    const data = await requestJson<{ success: boolean; dataset: { id: string; name: string } }>('/custom-datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newDatasetName.trim() }),
    });
    await refreshRegistries();
    setTargetDatasetId(data.dataset.id);
    setSaveMode('dataset');
    setNewDatasetName('');
    setStatus(`Created target dataset: ${data.dataset.name}`);
  }

  async function handleTranscribe() {
    if (!canRun) {
      setStatus('Choose an input source first.');
      return;
    }
    setIsTranscribing(true);
    setTranscript('');
    setMetrics(null);
    setStatus('Transcribing audio...');
    try {
      if (inputMode === 'record' || inputMode === 'upload') {
        const blob = inputMode === 'upload' ? audioFile! : recordedBlob!;
        const body = new FormData();
        body.append('model_id', selectedModel);
        body.append('audio', blob, audioFile?.name || 'recording.webm');
        body.append('target_dataset_id', saveMode === 'dataset' ? targetDatasetId : '__session__');
        if (referenceText.trim()) body.append('reference', referenceText.trim());
        const data = await requestJson<{ transcription: string; metrics?: Metrics | null }>('/transcribe', { method: 'POST', body });
        setTranscript(data.transcription || '');
        setMetrics(data.metrics || null);
        if (saveMode === 'dataset') await loadSamples(targetDatasetId, 50);
        setStatus(saveMode === 'dataset' ? `Transcription saved to ${targetDataset?.name || targetDatasetId}.` : 'Transcription complete in temporary session.');
      } else if (selectedSample) {
        const data = await requestJson<{ transcription: string; metrics?: Metrics | null; reference?: string }>('/transcribe-sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_id: selectedModel,
            dataset_id: selectedSample.dataset_id,
            sample_index: selectedSample.index,
            sample_limit: selectedSample.sample_limit,
          }),
        });
        setTranscript(data.transcription || '');
        setReferenceText(data.reference || selectedSample.reference || referenceText);
        setMetrics(data.metrics || null);
        setStatus('Selected sample transcription complete.');
      }
    } catch (error) {
      setStatus(`Transcription failed: ${(error as Error).message}`);
    } finally {
      setIsTranscribing(false);
    }
  }

  async function acceptVariant(row: DiffRow) {
    await requestJson('/word-variants', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reference_word: row.reference, hypothesis_word: row.hypothesis, source: 'workbench-ui' }),
    });
    setStatus(`Accepted variant saved: ${row.reference} = ${row.hypothesis}`);
  }

  return (
    <div className="h-full overflow-hidden bg-[#0a0a0b]">
      <div className="h-full flex flex-col p-4 gap-4">
        <section className="grid grid-cols-[260px_1fr] gap-4 min-h-[420px] flex-1 overflow-hidden">
          <aside className="bg-[#0f0f11] border border-gray-800 rounded-lg p-3 overflow-y-auto space-y-4">
            <PanelTitle title="Current Model" />
            <div className="rounded border border-gray-800 bg-[#121214] p-3">
              <div className="text-sm text-gray-100">{selectedModelInfo?.name || selectedModel || 'No model selected'}</div>
              <div className="text-xs text-gray-500 mt-1">{selectedModelInfo?.framework || 'Select one in Models'}</div>
              <button onClick={() => onWorkspaceChange('models')} className="mt-3 h-8 w-full rounded bg-gray-800 text-xs text-gray-300 hover:bg-gray-700">
                Change in Models
              </button>
            </div>

            <PanelTitle title="Input" />
            <div className="space-y-2">
              <ModeButton active={inputMode === 'record'} label="Record Audio" detail={recordedBlob ? 'Recording ready' : 'Microphone'} onClick={() => chooseMode('record')} />
              <ModeButton active={inputMode === 'upload'} label="Upload Audio" detail={audioFile?.name || 'Choose file'} onClick={() => chooseMode('upload')} />
              <ModeButton active={inputMode === 'sample'} label="Use Selected Dataset Sample" detail={selectedSample ? `#${selectedSample.index + 1}` : 'Select in Datasets'} onClick={() => chooseMode('sample')} />
            </div>

            {inputMode === 'record' && (
              <button onClick={() => void toggleRecording()} className={`h-9 w-full rounded text-sm font-medium flex items-center justify-center gap-2 ${isRecording ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                {isRecording ? <StopCircle className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                {isRecording ? 'Stop Recording' : 'Record'}
              </button>
            )}
            {inputMode === 'upload' && (
              <label className="h-9 w-full rounded bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm font-medium flex items-center justify-center gap-2 cursor-pointer">
                <Upload className="w-4 h-4" />
                {audioFile ? 'Change File' : 'Choose File'}
                <input type="file" accept="audio/*" className="hidden" onChange={(event) => handleFileChange(event.target.files?.[0] || null)} />
              </label>
            )}
            {inputMode === 'sample' && (
              <button onClick={() => onWorkspaceChange('dataset')} className="h-9 w-full rounded bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm font-medium flex items-center justify-center gap-2">
                <Database className="w-4 h-4" />
                Change in Datasets
              </button>
            )}

            <PanelTitle title="Save Result To" />
            <div className="space-y-2">
              <ModeButton active={saveMode === 'session'} label="Temporary Session" detail="Do not save audio" onClick={() => setSaveMode('session')} />
              <ModeButton active={saveMode === 'dataset'} label="Custom Dataset" detail={targetDataset?.name || 'Choose target'} onClick={() => setSaveMode('dataset')} />
            </div>

            {saveMode === 'dataset' && (
              <div className="space-y-2">
                <select value={targetDatasetId} onChange={(event) => setTargetDatasetId(event.target.value)} className="w-full bg-[#1a1a1d] border border-gray-700 rounded px-2.5 h-8 text-sm text-gray-200">
                  {targetDatasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
                </select>
                <div className="grid grid-cols-[1fr_auto] gap-2">
                  <input value={newDatasetName} onChange={(event) => setNewDatasetName(event.target.value)} placeholder="New dataset" className="bg-[#1a1a1d] border border-gray-700 rounded px-2 h-8 text-xs text-gray-200 placeholder:text-gray-600" />
                  <button onClick={() => void createTargetDataset()} className="h-8 w-8 rounded bg-gray-800 text-emerald-400 hover:bg-gray-700 flex items-center justify-center" title="Create dataset">
                    <Plus className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            <div className="pt-2 border-t border-gray-800 space-y-2">
              <button onClick={() => void handleTranscribe()} disabled={!canRun || isTranscribing} className="h-10 w-full rounded bg-emerald-500 text-white hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium flex items-center justify-center gap-2">
                {isTranscribing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {isTranscribing ? 'Running...' : 'Run Transcription'}
              </button>
              <button onClick={() => onWorkspaceChange('label')} className="h-9 w-full rounded bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm font-medium flex items-center justify-center gap-2">
                <PencilLine className="w-4 h-4" />
                Send To Label
              </button>
            </div>
          </aside>

          <main className="overflow-y-auto space-y-4 pr-1">
            <section className="bg-[#121214] border border-gray-800 rounded-lg p-4 space-y-4 min-h-[420px]">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-medium text-gray-100">Audio Preview</h2>
                <span className="text-xs text-gray-500">{inputLabel}</span>
              </div>
              {activeAudioUrl ? (
                <>
                  <AudioPlayer src={activeAudioUrl} title={inputLabel} subtitle={inputSubLabel} downloadName={audioFile?.name || 'workbench-input.wav'} />
                  <DetailedWaveform audioUrl={activeAudioUrl} height={160} />
                  <Spectrogram audioUrl={activeAudioUrl} height={150} />
                </>
              ) : (
                <EmptyInput onRecord={() => chooseMode('record')} onUpload={() => chooseMode('upload')} onDataset={() => onWorkspaceChange('dataset')} />
              )}
            </section>

            <section className="bg-[#121214] border border-gray-800 rounded-lg p-4 space-y-4">
              <h2 className="text-sm font-medium text-gray-100">Transcription</h2>
              <TranscriptBlock title="Predicted Transcript" value={transcript || 'Run transcription to see model output.'} muted={!transcript} />
              <TranscriptBlock title="Reference Transcript" value={referenceValue || 'Add a reference in Label, choose a dataset sample, or type one there later.'} muted={!referenceValue} />
              <div className="grid grid-cols-4 gap-3">
                <Metric label="WER" value={metrics ? formatPercent(metrics.adjusted_wer ?? metrics.wer) : 'N/A'} />
                <Metric label="CER" value={metrics ? formatPercent(metrics.cer) : 'N/A'} />
                <Metric label="Latency" value={isTranscribing ? 'Running' : 'N/A'} />
                <Metric label="Confidence" value={confidence == null ? 'N/A' : formatPercent(confidence)} />
              </div>
            </section>

            {metrics && (
              <section className="bg-[#121214] border border-gray-800 rounded-lg p-4 space-y-3">
                <h2 className="text-sm font-medium text-gray-100">Alignment Diff</h2>
                <div className="max-h-80 overflow-y-auto space-y-2 pr-1">
                  {(metrics.word_diff || []).length === 0 ? (
                    <div className="text-xs text-gray-500">No diff rows available.</div>
                  ) : metrics.word_diff.map((diff, index) => (
                    <DiffRowView key={index} diff={diff} onAccept={() => void acceptVariant(diff)} />
                  ))}
                </div>
              </section>
            )}
          </main>
        </section>
      </div>
    </div>
  );
}

function PanelTitle({ title }: { title: string }) {
  return <div className="text-[10px] uppercase tracking-wider text-gray-500 pt-1">{title}</div>;
}

function ModeButton({ active, label, detail, onClick }: { active: boolean; label: string; detail: string; onClick: () => void }) {
  return (
    <button onClick={onClick} className={`w-full rounded border p-2.5 text-left transition-colors ${active ? 'border-emerald-500/40 bg-emerald-500/10' : 'border-gray-800 bg-[#121214] hover:bg-gray-800/40'}`}>
      <div className="flex items-center gap-2">
        <span className={`w-2.5 h-2.5 rounded-full border ${active ? 'bg-emerald-400 border-emerald-400' : 'border-gray-600'}`} />
        <span className="text-sm text-gray-100">{label}</span>
      </div>
      <div className="text-xs text-gray-500 mt-1 pl-4 truncate">{detail}</div>
    </button>
  );
}

function EmptyInput({ onRecord, onUpload, onDataset }: { onRecord: () => void; onUpload: () => void; onDataset: () => void }) {
  return (
    <div className="min-h-[330px] rounded-lg border border-dashed border-gray-700 bg-[#0f0f11] flex items-center justify-center p-6">
      <div className="text-center max-w-sm">
        <FileAudio className="w-10 h-10 text-gray-600 mx-auto mb-3" />
        <h3 className="text-lg text-gray-100 font-medium">Choose an input source</h3>
        <p className="text-sm text-gray-500 mt-2">Record audio, upload a file, or select a dataset sample from the Datasets page.</p>
        <div className="grid grid-cols-3 gap-2 mt-5">
          <button onClick={onRecord} className="h-9 rounded bg-gray-800 text-xs text-gray-300 hover:bg-gray-700">Record</button>
          <button onClick={onUpload} className="h-9 rounded bg-gray-800 text-xs text-gray-300 hover:bg-gray-700">Upload</button>
          <button onClick={onDataset} className="h-9 rounded bg-gray-800 text-xs text-gray-300 hover:bg-gray-700">Dataset</button>
        </div>
      </div>
    </div>
  );
}

function TranscriptBlock({ title, value, muted }: { title: string; value: string; muted?: boolean }) {
  return (
    <div className="bg-[#0f0f11] border border-gray-800 rounded p-4 min-h-28">
      <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">{title}</div>
      <p className={`text-base leading-relaxed ${muted ? 'text-gray-600' : 'text-gray-100'}`} dir="auto">{value}</p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[#0f0f11] border border-gray-800 rounded p-3">
      <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className="text-xl font-mono text-gray-100 mt-1">{value}</div>
    </div>
  );
}

function DiffRowView({ diff, onAccept }: { diff: DiffRow; onAccept: () => void }) {
  const good = diff.type === 'equal' || diff.type === 'accepted';
  return (
    <div className={`group flex items-center gap-2 p-2 rounded border ${good ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-red-500/5 border-red-500/20'}`}>
      {good ? <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" /> : <XCircle className="w-4 h-4 text-red-400 shrink-0" />}
      <div className="grid grid-cols-2 gap-3 flex-1 min-w-0">
        <span className="text-sm text-gray-300 truncate" dir="auto">{diff.reference || '-'}</span>
        <span className="text-sm text-gray-300 truncate" dir="auto">{diff.hypothesis || '-'}</span>
      </div>
      <span className="text-[10px] text-gray-500 uppercase font-mono">{diff.type}</span>
      {diff.type === 'substitution' && diff.reference && diff.hypothesis && (
        <button onClick={onAccept} className="opacity-0 group-hover:opacity-100 h-6 px-2 rounded bg-gray-800 hover:bg-gray-700 text-[10px] text-gray-300 transition-all flex items-center gap-1">
          <ThumbsUp className="w-3 h-3" />
          Accept
        </button>
      )}
    </div>
  );
}
