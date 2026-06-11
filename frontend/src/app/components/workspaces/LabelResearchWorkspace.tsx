import { useEffect, useState } from 'react';
import { Save, Plus, Trash2, Upload, Download } from 'lucide-react';
import { AudioPlayer } from '../AudioPlayer';
import { DetailedWaveform } from '../DetailedWaveform';
import { requestJson, useApi } from '../../api';

interface AcceptedVariant {
  reference_word: string;
  hypothesis_word: string;
}

export function LabelResearchWorkspace() {
  const { datasets, selectedSample, setStatus, loadSamples, refreshRegistries } = useApi();
  const [transcript, setTranscript] = useState('');
  const [notes, setNotes] = useState('');
  const [targetDatasetId, setTargetDatasetId] = useState('recorded-audio');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedUrl, setUploadedUrl] = useState('');
  const [refWord, setRefWord] = useState('');
  const [variantWord, setVariantWord] = useState('');
  const [variants, setVariants] = useState<AcceptedVariant[]>([]);

  useEffect(() => {
    void loadVariants();
  }, []);

  useEffect(() => {
    if (selectedSample?.reference && !transcript) setTranscript(selectedSample.reference);
  }, [selectedSample]);

  useEffect(() => {
    return () => {
      if (uploadedUrl) URL.revokeObjectURL(uploadedUrl);
    };
  }, [uploadedUrl]);

  function handleFileChange(file: File | null) {
    if (uploadedUrl) URL.revokeObjectURL(uploadedUrl);
    setUploadedFile(file);
    setUploadedUrl(file ? URL.createObjectURL(file) : '');
    if (file) setStatus(`Labeling audio selected: ${file.name}`);
  }

  async function loadVariants() {
    const data = await requestJson<{ variants: AcceptedVariant[] }>('/word-variants');
    setVariants(data.variants || []);
  }

  async function saveUploadedLabel() {
    if (!uploadedFile || !transcript.trim()) {
      setStatus('Choose audio and write a transcript first.');
      return;
    }
    const body = new FormData();
    body.append('audio', uploadedFile, uploadedFile.name);
    body.append('label', transcript);
    body.append('notes', notes);
    body.append('target_dataset_id', targetDatasetId);
    await requestJson('/label-audio', { method: 'POST', body });
    setStatus('Uploaded audio label saved.');
    await refreshRegistries();
    await loadSamples(targetDatasetId, 50);
  }

  async function saveSelectedSampleLabel() {
    if (!selectedSample || !transcript.trim()) {
      setStatus('Select a sample and write a transcript first.');
      return;
    }
    await requestJson('/label-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: selectedSample.dataset_id,
        sample_index: selectedSample.index,
        sample_limit: selectedSample.sample_limit,
        label: transcript,
        notes,
        target_dataset_id: targetDatasetId,
      }),
    });
    setStatus('Selected sample label saved.');
    await refreshRegistries();
    await loadSamples(targetDatasetId, 50);
  }

  async function addVariant() {
    if (!refWord || !variantWord) return;
    await requestJson('/word-variants', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reference_word: refWord, hypothesis_word: variantWord, source: 'label-ui' }),
    });
    setRefWord('');
    setVariantWord('');
    await loadVariants();
    setStatus('Accepted spelling variant saved.');
  }

  return (
    <div className="h-full grid grid-cols-[1fr_420px] divide-x divide-gray-800">
      <div className="flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div className="bg-[#121214] border border-gray-800 rounded-lg p-4 space-y-4">
            <h3 className="text-sm font-medium text-gray-200">Audio Source</h3>
            <div>
              <label className="text-xs text-gray-500 mb-2 block">Save Target Dataset</label>
              <select
                value={targetDatasetId}
                onChange={(event) => setTargetDatasetId(event.target.value)}
                className="w-full bg-[#0f0f11] border border-gray-700 rounded px-3 h-9 text-sm text-gray-200"
              >
                {datasets
                  .filter((dataset) => dataset.id === 'recorded-audio' || dataset.id.startsWith('custom-'))
                  .map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                  ))}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <label className="h-24 border-2 border-dashed border-gray-700 rounded-lg flex flex-col items-center justify-center gap-2 hover:border-gray-600 hover:bg-gray-800/30 transition-colors cursor-pointer">
                <Upload className="w-5 h-5 text-gray-500" />
                <span className="text-xs text-gray-400">{uploadedFile ? uploadedFile.name : 'Choose audio file'}</span>
                <input type="file" accept="audio/*" className="hidden" onChange={(e) => handleFileChange(e.target.files?.[0] || null)} />
              </label>

              <button
                onClick={() => setTranscript(selectedSample?.reference || '')}
                className="h-24 border border-gray-700 rounded-lg flex flex-col items-center justify-center gap-2 hover:bg-gray-800/30 transition-colors"
              >
                <span className="text-xs text-gray-400">{selectedSample ? `Selected clip #${selectedSample.index + 1}` : 'No dataset clip selected'}</span>
                <span className="text-xs font-mono text-emerald-400">{selectedSample ? `#${String(selectedSample.index + 1).padStart(4, '0')}` : 'None'}</span>
              </button>
            </div>
            <AudioPlayer
              src={uploadedUrl || selectedSample?.audio_url}
              title={uploadedFile?.name || (selectedSample ? `Clip #${selectedSample.index + 1}` : 'Select audio to preview')}
              subtitle={selectedSample?.dataset_id || undefined}
              downloadName={uploadedFile?.name || 'label-audio.wav'}
            />
            <DetailedWaveform audioUrl={uploadedUrl || selectedSample?.audio_url} height={80} />
          </div>

          <div className="bg-[#121214] border border-gray-800 rounded-lg p-4 space-y-3">
            <h3 className="text-sm font-medium text-gray-200">Human Transcription</h3>
            <div>
              <label className="text-xs text-gray-500 mb-2 block">Transcript</label>
              <textarea
                value={transcript}
                onChange={(e) => setTranscript(e.target.value)}
                placeholder="Write the correct transcript here..."
                dir="auto"
                rows={4}
                className="w-full bg-[#0f0f11] border border-gray-700 rounded px-3 py-2 text-gray-200 placeholder:text-gray-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
              />
              <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
                <span>Words: {transcript.trim().split(/\s+/).filter(Boolean).length}</span>
                <span>Characters: {transcript.length}</span>
              </div>
            </div>

            <div>
              <label className="text-xs text-gray-500 mb-2 block">Metadata & Notes</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Speaker, accent, source, quality, noise..."
                rows={5}
                className="w-full bg-[#0f0f11] border border-gray-700 rounded px-3 py-2 text-xs text-gray-200 placeholder:text-gray-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 font-mono"
              />
            </div>

            <div className="grid grid-cols-2 gap-2 pt-2">
              <button onClick={() => void saveUploadedLabel()} className="h-9 rounded flex items-center justify-center gap-2 text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-600 transition-colors">
                <Save className="w-4 h-4" />
                Save Upload Label
              </button>
              <button onClick={() => void saveSelectedSampleLabel()} className="h-9 rounded flex items-center justify-center gap-2 text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors">
                Save Selected Sample
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col overflow-hidden bg-[#0f0f11]">
        <div className="p-3 border-b border-gray-800">
          <h3 className="text-sm font-medium text-gray-200 mb-1">Accepted Word Spellings</h3>
          <p className="text-xs text-gray-500">Manage variant spellings for adjusted evaluation</p>
        </div>

        <div className="p-3 border-b border-gray-800 space-y-2">
          <div>
            <label className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 block">Reference Word</label>
            <input value={refWord} onChange={(e) => setRefWord(e.target.value)} dir="auto" className="w-full bg-[#1a1a1d] border border-gray-700 rounded px-2.5 h-8 text-sm text-gray-200 placeholder:text-gray-600" />
          </div>
          <div>
            <label className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 block">Accepted Variant</label>
            <input value={variantWord} onChange={(e) => setVariantWord(e.target.value)} dir="auto" className="w-full bg-[#1a1a1d] border border-gray-700 rounded px-2.5 h-8 text-sm text-gray-200 placeholder:text-gray-600" />
          </div>
          <button onClick={() => void addVariant()} className="w-full h-8 rounded flex items-center justify-center gap-2 text-xs font-medium bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-colors">
            <Plus className="w-3.5 h-3.5" />
            Add Variant
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-500">{variants.length} variants</span>
            <button className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1">
              <Download className="w-3 h-3" />
              Export
            </button>
          </div>
          {variants.length === 0 ? (
            <div className="text-xs text-gray-500">Accepted spelling variants will appear here.</div>
          ) : variants.map((variant, index) => (
            <div key={index} className="bg-[#121214] border border-gray-800 rounded-lg p-2.5 space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="text-[10px] text-gray-500 mb-0.5">Reference</div>
                  <div className="text-sm text-gray-200" dir="auto">{variant.reference_word}</div>
                </div>
                <div>
                  <div className="text-[10px] text-gray-500 mb-0.5">Variant</div>
                  <div className="text-sm text-gray-200" dir="auto">{variant.hypothesis_word}</div>
                </div>
              </div>
              <div className="flex items-center justify-between pt-1.5 border-t border-gray-800">
                <span className="text-[10px] text-gray-600">saved</span>
                <button className="text-red-400 hover:text-red-300 transition-colors">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
