import { Download, Pause, Play, RotateCcw, Volume2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

interface AudioPlayerProps {
  src?: string;
  title?: string;
  subtitle?: string;
  downloadName?: string;
}

export function AudioPlayer({ src, title = 'Select a clip to preview', subtitle, downloadName = 'audio.wav' }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(0.9);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.volume = volume;
  }, [volume]);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, [src]);

  async function togglePlay() {
    const audio = audioRef.current;
    if (!audio || !src) return;
    if (audio.paused) {
      await audio.play();
    } else {
      audio.pause();
    }
  }

  function seek(value: number) {
    const audio = audioRef.current;
    if (!audio || !Number.isFinite(duration)) return;
    audio.currentTime = value;
    setCurrentTime(value);
  }

  function replay() {
    const audio = audioRef.current;
    if (!audio || !src) return;
    audio.currentTime = 0;
    void audio.play();
  }

  const progress = duration ? (currentTime / duration) * 100 : 0;

  return (
    <div className="relative overflow-hidden rounded-lg border border-emerald-500/20 bg-[#0f1113] p-3 shadow-[0_0_0_1px_rgba(16,185,129,0.04),0_18px_48px_rgba(0,0,0,0.24)]">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-emerald-400/70 to-transparent" />
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || 0)}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime || 0)}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
      />

      <div className="grid grid-cols-[auto_1fr_auto] gap-3 items-center">
        <button
          onClick={() => void togglePlay()}
          disabled={!src}
          aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
          className="size-11 rounded-full bg-emerald-500 text-[#06120e] hover:bg-emerald-400 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed flex items-center justify-center transition-colors shadow-[0_0_28px_rgba(16,185,129,0.2)]"
        >
          {isPlaying ? <Pause className="size-5" /> : <Play className="size-5 ml-0.5" />}
        </button>

        <div className="min-w-0">
          <div className="flex items-center justify-between gap-3 mb-2">
            <div className="min-w-0">
              <div className="text-sm font-medium text-gray-100 truncate">{title}</div>
              <div className="text-[11px] text-gray-500 truncate">{subtitle || 'Audio preview'}</div>
            </div>
            <div className="text-xs font-mono text-gray-400 whitespace-nowrap">
              {formatTime(currentTime)} / {formatTime(duration)}
            </div>
          </div>

          <div className="relative h-2 rounded-full bg-gray-800 overflow-hidden">
            <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-500 to-cyan-400" style={{ width: `${progress}%` }} />
            <input
              aria-label="Audio position"
              type="range"
              min={0}
              max={duration || 0}
              step={0.01}
              value={Math.min(currentTime, duration || 0)}
              onChange={(event) => seek(Number(event.target.value))}
              disabled={!src || !duration}
              className="absolute inset-0 h-full w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
            />
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={replay}
            disabled={!src}
            aria-label="Replay audio"
            className="size-8 rounded bg-gray-800/80 text-gray-300 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
          >
            <RotateCcw className="size-4" />
          </button>
          {src && (
            <a
              href={src}
              download={downloadName}
              aria-label="Download audio"
              className="size-8 rounded bg-gray-800/80 text-gray-300 hover:bg-gray-700 flex items-center justify-center transition-colors"
            >
              <Download className="size-4" />
            </a>
          )}
        </div>
      </div>

      <div className="mt-3 flex items-center gap-2">
        <Volume2 className="size-3.5 text-gray-500" />
        <input
          aria-label="Audio volume"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={volume}
          onChange={(event) => setVolume(Number(event.target.value))}
          className="modern-audio-range w-32"
        />
      </div>
    </div>
  );
}

function formatTime(value: number) {
  if (!Number.isFinite(value) || value <= 0) return '0:00';
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}
