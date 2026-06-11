import { useEffect, useRef } from 'react';

interface DetailedWaveformProps {
  audioUrl?: string;
  height?: number;
}

type WavePalette = {
  background: string;
  grid: string;
  center: string;
  text: string;
  wave: string;
  fill: string;
};

const DARK: WavePalette = {
  background: '#0b0d0e',
  grid: 'rgba(148, 163, 184, 0.14)',
  center: 'rgba(226, 232, 240, 0.28)',
  text: 'rgba(148, 163, 184, 0.72)',
  wave: '#34d399',
  fill: 'rgba(16, 185, 129, 0.16)',
};

const LIGHT: WavePalette = {
  background: '#ffffff',
  grid: 'rgba(15, 23, 42, 0.10)',
  center: 'rgba(15, 23, 42, 0.18)',
  text: 'rgba(71, 85, 105, 0.68)',
  wave: '#047857',
  fill: 'rgba(16, 185, 129, 0.13)',
};

function palette() {
  const root = document.documentElement;
  const saved = window.localStorage.getItem('speechrecon-theme');
  if (root.classList.contains('theme-light') || saved === 'light') return LIGHT;
  return DARK;
}

function drawFrame(ctx: CanvasRenderingContext2D, width: number, height: number, message?: string) {
  const p = palette();
  ctx.fillStyle = p.background;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = Math.round((height / 4) * i) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (let i = 1; i < 10; i += 1) {
    const x = Math.round((width / 10) * i) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  ctx.strokeStyle = p.center;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(0, Math.round(height / 2) + 0.5);
  ctx.lineTo(width, Math.round(height / 2) + 0.5);
  ctx.stroke();
  ctx.setLineDash([]);

  if (message) {
    ctx.fillStyle = p.text;
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, width / 2, height / 2);
  }
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

function drawWaveform(ctx: CanvasRenderingContext2D, width: number, height: number, buffer: AudioBuffer) {
  const p = palette();
  const data = buffer.getChannelData(0);
  const columns = Math.min(width, 1200);
  const step = Math.max(1, Math.floor(data.length / columns));
  const center = height / 2;
  const scale = height * 0.42;
  const topPath = new Path2D();
  const bottomPath = new Path2D();
  const fillPath = new Path2D();

  drawFrame(ctx, width, height);
  fillPath.moveTo(0, center);

  for (let x = 0; x < columns; x += 1) {
    const start = x * step;
    let min = 1;
    let max = -1;
    for (let i = start; i < Math.min(start + step, data.length); i += 1) {
      const value = data[i] || 0;
      if (value < min) min = value;
      if (value > max) max = value;
    }
    const drawX = (x / columns) * width;
    const yTop = center + min * scale;
    const yBottom = center + max * scale;
    if (x === 0) {
      topPath.moveTo(drawX, yTop);
      bottomPath.moveTo(drawX, yBottom);
    } else {
      topPath.lineTo(drawX, yTop);
      bottomPath.lineTo(drawX, yBottom);
    }
    fillPath.lineTo(drawX, yTop);
  }

  for (let x = columns - 1; x >= 0; x -= 1) {
    const start = x * step;
    let max = -1;
    for (let i = start; i < Math.min(start + step, data.length); i += 1) {
      const value = data[i] || 0;
      if (value > max) max = value;
    }
    fillPath.lineTo((x / columns) * width, center + max * scale);
  }
  fillPath.closePath();

  ctx.fillStyle = p.fill;
  ctx.fill(fillPath);
  ctx.strokeStyle = p.wave;
  ctx.lineWidth = Math.max(1.25, Math.min(2.5, width / 480));
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke(topPath);
  ctx.stroke(bottomPath);

  ctx.fillStyle = p.text;
  ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  for (let i = 0; i <= 5; i += 1) {
    const x = (width / 5) * i;
    const time = ((buffer.duration / 5) * i).toFixed(1);
    ctx.fillText(`${time}s`, x, height - 5);
  }
}

export function DetailedWaveform({ audioUrl, height = 100 }: DetailedWaveformProps) {
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
      const canvasHeight = Math.max(72, Math.floor(height * dpr));
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== canvasHeight) canvas.height = canvasHeight;

      if (bufferRef.current) {
        drawWaveform(ctx, width, canvasHeight, bufferRef.current);
      } else {
        drawFrame(ctx, width, canvasHeight, message || (audioUrl ? 'Rendering waveform...' : 'Select a clip to view its waveform'));
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
          if (!cancelled) resizeAndDraw('Waveform unavailable');
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
      className="w-full rounded-md border border-gray-800 bg-white"
      style={{ height: `${height}px` }}
    />
  );
}
