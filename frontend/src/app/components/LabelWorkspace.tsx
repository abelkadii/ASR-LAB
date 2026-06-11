import { useState } from 'react';
import { Upload, Save, Plus, Trash2 } from 'lucide-react';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Textarea } from './ui/Textarea';

interface AcceptedVariant {
  reference: string;
  variant: string;
}

export function LabelWorkspace() {
  const [transcript, setTranscript] = useState('');
  const [notes, setNotes] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [refWord, setRefWord] = useState('');
  const [variantWord, setVariantWord] = useState('');
  const [variants, setVariants] = useState<AcceptedVariant[]>([
    { reference: 'كيفاش', variant: 'كيفاش' },
    { reference: 'واش', variant: 'واش' },
  ]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setUploadedFile(file);
  };

  const addVariant = () => {
    if (refWord && variantWord) {
      setVariants([...variants, { reference: refWord, variant: variantWord }]);
      setRefWord('');
      setVariantWord('');
    }
  };

  const removeVariant = (index: number) => {
    setVariants(variants.filter((_, i) => i !== index));
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto p-6 space-y-6">
        <div>
          <h1>Label Audio</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Create human-labeled transcripts and manage accepted spelling variants
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6">
            <div className="bg-card border border-border rounded-lg p-6 space-y-4">
              <h3>Upload Audio for Labeling</h3>

              <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="label-audio-upload"
                />
                <label htmlFor="label-audio-upload" className="cursor-pointer">
                  <Upload className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">Click to upload audio for labeling</p>
                  {uploadedFile && (
                    <p className="text-xs text-primary mt-2">{uploadedFile.name}</p>
                  )}
                </label>
              </div>

              <div>
                <label className="block text-sm mb-2">Human Transcript</label>
                <Textarea
                  value={transcript}
                  onChange={(e) => setTranscript(e.target.value)}
                  placeholder="اكتب النص الصحيح هنا..."
                  rows={4}
                  dir="rtl"
                />
              </div>

              <div>
                <label className="block text-sm mb-2">Notes (Speaker, Accent, Source, Quality)</label>
                <Textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="e.g., Male speaker, Casablanca accent, high quality recording"
                  rows={3}
                />
              </div>

              <div className="flex gap-2">
                <Button className="flex-1">
                  <Save className="w-4 h-4" />
                  Save Uploaded Audio Label
                </Button>
                <Button variant="outline" className="flex-1">
                  <Save className="w-4 h-4" />
                  Save Selected Sample Label
                </Button>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-card border border-border rounded-lg p-6 space-y-4">
              <h3>Accepted Word Spellings</h3>
              <p className="text-sm text-muted-foreground">
                Define spelling variants that should be accepted as correct during evaluation
              </p>

              <div className="space-y-3">
                <div>
                  <label className="block text-sm mb-1">Reference Word</label>
                  <Input
                    value={refWord}
                    onChange={(e) => setRefWord(e.target.value)}
                    placeholder="كيفاش"
                    dir="rtl"
                  />
                </div>

                <div>
                  <label className="block text-sm mb-1">Accepted Variant</label>
                  <Input
                    value={variantWord}
                    onChange={(e) => setVariantWord(e.target.value)}
                    placeholder="كيفاه"
                    dir="rtl"
                  />
                </div>

                <Button onClick={addVariant} variant="secondary" className="w-full">
                  <Plus className="w-4 h-4" />
                  Add Accepted Variant
                </Button>
              </div>

              <div className="border-t border-border pt-4 mt-4">
                <label className="block text-sm mb-3">Saved Variants</label>
                {variants.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    No variants saved yet
                  </p>
                ) : (
                  <div className="space-y-2">
                    {variants.map((variant, index) => (
                      <div
                        key={index}
                        className="flex items-center gap-3 p-3 bg-muted rounded-lg"
                      >
                        <div className="flex-1 grid grid-cols-2 gap-4">
                          <div>
                            <span className="text-xs text-muted-foreground block mb-1">Reference</span>
                            <span className="text-sm dir-rtl" dir="rtl">{variant.reference}</span>
                          </div>
                          <div>
                            <span className="text-xs text-muted-foreground block mb-1">Variant</span>
                            <span className="text-sm dir-rtl" dir="rtl">{variant.variant}</span>
                          </div>
                        </div>
                        <button
                          onClick={() => removeVariant(index)}
                          className="p-2 hover:bg-destructive/10 text-destructive rounded transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
