import { useEffect, useRef } from 'react';

interface SpectrogramProps {
  audioUrl?: string;
  height?: number;
}

type VisualPalette = {
  background: string;
  panel: string;
  grid: string;
  text: string;
  muted: string;
  wave: string;
  waveFill: string;
  bar: string;
  barSoft: string;
  center: string;
};

const DARK: VisualPalette = {
  background: '#090b0c',
  panel: '#0f1413',
  grid: 'rgba(148, 163, 184, 0.14)',
  text: 'rgba(226, 232, 240, 0.84)',
  muted: 'rgba(148, 163, 184, 0.6)',
  wave: '#34d399',
  waveFill: 'rgba(16, 185, 129, 0.16)',
  bar: '#10b981',
  barSoft: 'rgba(16, 185, 129, 0.22)',
  center: 'rgba(226, 232, 240, 0.26)',
};

const LIGHT: VisualPalette = {
  background: '#ffffff',
  panel: '#f8fafc',
  grid: 'rgba(15, 23, 42, 0.10)',
  text: 'rgba(15, 23, 42, 0.78)',
  muted: 'rgba(71, 85, 105, 0.62)',
  wave: '#047857',
  waveFill: 'rgba(16, 185, 129, 0.14)',
  bar: '#059669',
  barSoft: 'rgba(5, 150, 105, 0.16)',
  center: 'rgba(15, 23, 42, 0.16)',
};

function palette() {
  const root = document.documentElement;
  const saved = window.localStorage.getItem('speechrecon-theme');
  if (root.classList.contains('theme-light') || saved === 'light') return LIGHT;
  return DARK;
}

async function decodeAudio(url: string) {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const AudioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextClass) throw new Error('AudioContext is not available');
  const context = new AudioContextClass();
  try {
    return await context.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    await context.close();
  }
}

function drawBase(ctx: CanvasRenderingContext2D, width: number, height: number, p: VisualPalette) {
  ctx.fillStyle = p.background;
  ctx.fillRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, p.panel);
  gradient.addColorStop(1, p.background);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i += 1) {
    const y = Math.round((height / 5) * i) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (let i = 1; i < 12; i += 1) {
    const x = Math.round((width / 12) * i) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  ctx.strokeStyle = p.center;
  ctx.beginPath();
  ctx.moveTo(0, Math.round(height * 0.48) + 0.5);
  ctx.lineTo(width, Math.round(height * 0.48) + 0.5);
  ctx.stroke();
}

function drawEmpty(ctx: CanvasRenderingContext2D, width: number, height: number, message: string) {
  const p = palette();
  drawBase(ctx, width, height, p);
  ctx.fillStyle = p.muted;
  ctx.font = '12px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message, width / 2, height / 2);
}

function rms(samples: Float32Array, start: number, end: number) {
  let sum = 0;
  let count = 0;
  for (let i = start; i < end; i += 1) {
    const value = samples[i] || 0;
    sum += value * value;
    count += 1;
  }
  return count ? Math.sqrt(sum / count) : 0;
}

function drawAudioFingerprint(ctx: CanvasRenderingContext2D, width: number, height: number, buffer: AudioBuffer) {
  const p = palette();
  const samples = buffer.getChannelData(0);
  const columns = Math.min(260, Math.max(120, Math.floor(width / 5)));
  const samplesPerColumn = Math.max(1, Math.floor(samples.length / columns));
  const topBand = Math.round(height * 0.20);
  const waveTop = Math.round(height * 0.30);
  const waveHeight = Math.round(height * 0.48);
  const centerY = waveTop + waveHeight / 2;
  const waveScale = waveHeight * 0.48;

  drawBase(ctx, width, height, p);

  const levels: number[] = [];
  let peak = 0;
  for (let x = 0; x < columns; x += 1) {
    const start = x * samplesPerColumn;
    const end = Math.min(samples.length, start + samplesPerColumn);
    const level = rms(samples, start, end);
    levels.push(level);
    peak = Math.max(peak, level);
  }

  const columnWidth = width / columns;
  ctx.fillStyle = p.barSoft;
  for (let x = 0; x < columns; x += 1) {
    const normalized = peak ? Math.min(1, levels[x] / peak) : 0;
    const barHeight = Math.max(2, normalized * topBand);
    ctx.fillRect(
      Math.floor(x * columnWidth),
      Math.floor(topBand - barHeight + 8),
      Math.max(1, Math.ceil(columnWidth - 1)),
      Math.ceil(barHeight),
    );
  }

  const area = new Path2D();
  area.moveTo(0, centerY);
  for (let x = 0; x < columns; x += 1) {
    const normalized = peak ? Math.min(1, levels[x] / peak) : 0;
    area.lineTo(x * columnWidth, centerY - normalized * waveScale);
  }
  for (let x = columns - 1; x >= 0; x -= 1) {
    const normalized = peak ? Math.min(1, levels[x] / peak) : 0;
    area.lineTo(x * columnWidth, centerY + normalized * waveScale);
  }
  area.closePath();
  ctx.fillStyle = p.waveFill;
  ctx.fill(area);

  const topLine = new Path2D();
  const bottomLine = new Path2D();
  for (let x = 0; x < columns; x += 1) {
    const normalized = peak ? Math.min(1, levels[x] / peak) : 0;
    const px = x * columnWidth;
    const topY = centerY - normalized * waveScale;
    const bottomY = centerY + normalized * waveScale;
    if (x === 0) {
      topLine.moveTo(px, topY);
      bottomLine.moveTo(px, bottomY);
    } else {
      topLine.lineTo(px, topY);
      bottomLine.lineTo(px, bottomY);
    }
  }

  ctx.strokeStyle = p.wave;
  ctx.lineWidth = Math.max(1.5, Math.min(3, width / 420));
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke(topLine);
  ctx.stroke(bottomLine);

  const footerY = height - Math.round(height * 0.15);
  ctx.fillStyle = p.bar;
  for (let i = 0; i < 32; i += 1) {
    const start = Math.floor((samples.length / 32) * i);
    const end = Math.floor((samples.length / 32) * (i + 1));
    const level = rms(samples, start, end);
    const normalized = peak ? Math.min(1, level / peak) : 0;
    const x = Math.round((width / 32) * i + 4);
    const barWidth = Math.max(2, Math.floor(width / 32) - 8);
    const barHeight = Math.max(3, normalized * height * 0.13);
    ctx.globalAlpha = 0.35 + normalized * 0.55;
    ctx.fillRect(x, footerY - barHeight / 2, barWidth, barHeight);
  }
  ctx.globalAlpha = 1;

  ctx.fillStyle = p.text;
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText(`${(buffer.duration || 0).toFixed(2)}s`, width - 8, 8);
  ctx.textAlign = 'left';
  ctx.fillText(`${Math.round(buffer.sampleRate / 1000)} kHz`, 8, 8);
}

export function Spectrogram({ audioUrl, height = 120 }: SpectrogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufferRef = useRef<AudioBuffer | null>(null);

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    function resizeAndDraw(message?: string) {
      const dpr = Math.max(1, Math.min(2.5, window.devicePixelRatio || 1));
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.floor(rect.width * dpr));
      const canvasHeight = Math.max(76, Math.floor(height * dpr));
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== canvasHeight) canvas.height = canvasHeight;

      if (bufferRef.current) {
        drawAudioFingerprint(ctx, width, canvasHeight, bufferRef.current);
      } else {
        drawEmpty(ctx, width, canvasHeight, message || (audioUrl ? 'Rendering audio view...' : 'Select a clip to view audio'));
      }
    }

    const observer = new ResizeObserver(() => resizeAndDraw());
    observer.observe(canvas);
    const themeObserver = new MutationObserver(() => resizeAndDraw());
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

    bufferRef.current = null;
    resizeAndDraw();

    if (audioUrl) {
      decodeAudio(audioUrl)
        .then((buffer) => {
          if (cancelled) return;
          bufferRef.current = buffer;
          resizeAndDraw();
        })
        .catch(() => {
          if (!cancelled) resizeAndDraw('Audio view unavailable');
        });
    }

    return () => {
      cancelled = true;
      observer.disconnect();
      themeObserver.disconnect();
    };
  }, [audioUrl, height]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-md border border-gray-800 bg-white shadow-[inset_0_1px_0_rgba(255,255,255,0.7)]"
      style={{ height: `${height}px` }}
    />
  );
}
