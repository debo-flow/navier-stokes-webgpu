export interface GhiaDataset {
  re: number;
  y: number[];
  u: number[];
}

// Reference vertical centerline u(x=0.5, y) values from:
// Ghia, U., Ghia, K. N., & Shin, C. T. (1982). Journal of Computational Physics, 48(3), 387-411.
const Y_COORDS = [
  1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
  0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000,
];

export const GHIA_DATASETS: GhiaDataset[] = [
  {
    re: 100,
    y: Y_COORDS,
    u: [
      1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
      -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000,
    ],
  },
  {
    re: 400,
    y: Y_COORDS,
    u: [
      1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135,
      -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000,
    ],
  },
  {
    re: 1000,
    y: Y_COORDS,
    u: [
      1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702,
      -0.06080, -0.10648, -0.27805, -0.38288, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000,
    ],
  },
];

export function nearestGhiaDataset(re: number): GhiaDataset {
  return GHIA_DATASETS.reduce((best, curr) =>
    Math.abs(curr.re - re) < Math.abs(best.re - re) ? curr : best
  );
}

export function renderGhiaChart(
  canvas: HTMLCanvasElement,
  cpuU: Float32Array,
  nx: number,
  ny: number,
  re: number
): number {
  const ctx = canvas.getContext('2d')!;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 28, padR = 10, padT = 10, padB = 18;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const toX = (u: number) => padL + ((u + 0.5) / 1.5) * plotW;
  const toY = (y: number) => padT + (1.0 - y) * plotH;

  // Grid lines & zero-velocity axis
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toX(0), padT);
  ctx.lineTo(toX(0), padT + plotH);
  ctx.moveTo(padL, toY(0.5));
  ctx.lineTo(padL + plotW, toY(0.5));
  ctx.stroke();

  // Sample live centerline u(0.5, y)
  const midI = Math.floor(nx / 2);
  const sampleSimU = (yNorm: number) => {
    const gy = Math.min(ny - 1, Math.max(0, yNorm * (ny - 1)));
    const j0 = Math.floor(gy);
    const j1 = Math.min(ny - 1, j0 + 1);
    const f = gy - j0;
    return cpuU[j0 * nx + midI] * (1 - f) + cpuU[j1 * nx + midI] * f;
  };

  // Draw live simulation curve
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.75;
  ctx.beginPath();
  for (let j = 0; j < ny; j++) {
    const yNorm = j / (ny - 1);
    const uVal = cpuU[j * nx + midI];
    const px = toX(uVal);
    const py = toY(yNorm);
    if (j === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Draw Ghia et al. (1982) reference markers & compute RMS error
  const ref = nearestGhiaDataset(re);
  let sqErr = 0;
  ctx.fillStyle = '#f97316';
  for (let k = 0; k < ref.y.length; k++) {
    const simU = sampleSimU(ref.y[k]);
    sqErr += (simU - ref.u[k]) ** 2;
    ctx.beginPath();
    ctx.arc(toX(ref.u[k]), toY(ref.y[k]), 2.6, 0, Math.PI * 2);
    ctx.fill();
  }

  // Axis labels
  ctx.fillStyle = '#64748b';
  ctx.font = '9px monospace';
  ctx.fillText('-0.5', padL - 4, h - 4);
  ctx.fillText('0', toX(0) - 3, h - 4);
  ctx.fillText('1.0', padL + plotW - 14, h - 4);
  ctx.fillText('y=1', 4, padT + 7);
  ctx.fillText('y=0', 4, padT + plotH);

  return Math.sqrt(sqErr / ref.y.length);
}
