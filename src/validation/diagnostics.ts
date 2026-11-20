export interface AeroSample {
  t: number;
  cd: number;
  cl: number;
}

export interface TaylorGreenSample {
  t: number;
  l2Error: number;
  simKE: number;
  exactKE: number;
}

/**
 * Integrates pressure and viscous wall stresses over the immersed solid boundary
 * to compute dimensionless drag (C_D) and lift (C_L) coefficients:
 *   F = integral( -p * n + nu * (du_t / dn) ) dS
 *   C_D = F_x / (0.5 * U^2 * L),   C_L = F_y / (0.5 * U^2 * L)
 */
export function computeAeroCoefficients(
  cpuU: Float32Array,
  cpuV: Float32Array,
  cpuP: Float32Array,
  cpuSolid: Float32Array,
  nx: number,
  ny: number,
  dx: number,
  nu: number,
  charLength: number
): { cd: number; cl: number } {
  let fx = 0.0;
  let fy = 0.0;

  const cellVel = (i: number, j: number): [number, number] => {
    const id = j * nx + i;
    const uR = i + 1 < nx ? cpuU[id + 1] : cpuU[id];
    const vT = j + 1 < ny ? cpuV[id + nx] : cpuV[id];
    return [0.5 * (cpuU[id] + uR), 0.5 * (cpuV[id] + vT)];
  };

  for (let j = 1; j < ny - 1; j++) {
    for (let i = 1; i < nx - 1; i++) {
      const id = j * nx + i;
      if (cpuSolid[id] < 0.5) continue;

      // Left face: outward normal n = (-1, 0) into fluid at (i - 1, j)
      if (cpuSolid[id - 1] < 0.5) {
        const [, vf] = cellVel(i - 1, j);
        fx += cpuP[id - 1] * dx;
        fy += nu * (vf / dx) * dx;
      }
      // Right face: outward normal n = (+1, 0) into fluid at (i + 1, j)
      if (cpuSolid[id + 1] < 0.5) {
        const [, vf] = cellVel(i + 1, j);
        fx -= cpuP[id + 1] * dx;
        fy += nu * (vf / dx) * dx;
      }
      // Bottom face: outward normal n = (0, -1) into fluid at (i, j - 1)
      if (cpuSolid[id - nx] < 0.5) {
        const [uf] = cellVel(i, j - 1);
        fy += cpuP[id - nx] * dx;
        fx += nu * (uf / dx) * dx;
      }
      // Top face: outward normal n = (0, +1) into fluid at (i, j + 1)
      if (cpuSolid[id + nx] < 0.5) {
        const [uf] = cellVel(i, j + 1);
        fy -= cpuP[id + nx] * dx;
        fx += nu * (uf / dx) * dx;
      }
    }
  }

  const dynHead = 0.5 * 1.0 * charLength; // rho = 1, U_inf = 1
  return {
    cd: fx / dynHead,
    cl: fy / dynHead,
  };
}

/**
 * Evaluates exact 2D Taylor-Green Vortex decay on [0, 1]^2:
 *   u(x, y, t) =  sin(2*pi*x) * cos(2*pi*y) * exp(-8*pi^2 * nu * t)
 *   v(x, y, t) = -cos(2*pi*x) * sin(2*pi*y) * exp(-8*pi^2 * nu * t)
 *   E_exact(t) = 0.25 * exp(-16*pi^2 * nu * t)
 */
export function computeTaylorGreenMetrics(
  cpuU: Float32Array,
  cpuV: Float32Array,
  nx: number,
  ny: number,
  nu: number,
  time: number
): TaylorGreenSample {
  const twoPi = 2.0 * Math.PI;
  const decay = Math.exp(-8.0 * Math.PI * Math.PI * nu * time);

  let errSq = 0.0;
  let refSq = 0.0;
  let simKESum = 0.0;

  // Subsample stride 2 for fast diagnostic pass while preserving statistical accuracy
  const stride = nx >= 512 ? 2 : 1;
  let samples = 0;

  for (let j = 0; j < ny; j += stride) {
    const yu = (j + 0.5) / ny;
    const yv = j / ny;
    const cosYu = Math.cos(twoPi * yu);
    const sinYv = Math.sin(twoPi * yv);

    for (let i = 0; i < nx; i += stride) {
      const id = j * nx + i;
      const xu = i / nx;
      const xv = (i + 0.5) / nx;

      const uExact = Math.sin(twoPi * xu) * cosYu * decay;
      const vExact = -Math.cos(twoPi * xv) * sinYv * decay;

      const uSim = cpuU[id];
      const vSim = cpuV[id];

      const du = uSim - uExact;
      const dv = vSim - vExact;
      errSq += du * du + dv * dv;
      refSq += uExact * uExact + vExact * vExact;
      simKESum += 0.5 * (uSim * uSim + vSim * vSim);
      samples++;
    }
  }

  return {
    t: time,
    l2Error: Math.sqrt(errSq / Math.max(refSq, 1e-12)),
    simKE: simKESum / samples,
    exactKE: 0.25 * decay * decay,
  };
}

export function renderAeroChart(canvas: HTMLCanvasElement, history: AeroSample[]) {
  const ctx = canvas.getContext('2d')!;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;

  const padL = 32, padR = 8, padT = 8, padB = 14;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let minV = -1.5;
  let maxV = 2.5;
  for (const s of history) {
    minV = Math.min(minV, s.cd, s.cl);
    maxV = Math.max(maxV, s.cd, s.cl);
  }
  const margin = Math.max(0.2, (maxV - minV) * 0.12);
  minV -= margin;
  maxV += margin;

  const toX = (k: number) => padL + (k / (history.length - 1)) * plotW;
  const toY = (v: number) => padT + (1 - (v - minV) / (maxV - minV)) * plotH;

  // Zero reference line
  const y0 = toY(0);
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, y0);
  ctx.lineTo(padL + plotW, y0);
  ctx.stroke();

  // Plot C_D (rose)
  ctx.strokeStyle = '#f43f5e';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach((s, k) => {
    const x = toX(k), y = toY(s.cd);
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Plot C_L (cyan)
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach((s, k) => {
    const x = toX(k), y = toY(s.cl);
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Y-axis ticks
  ctx.fillStyle = '#64748b';
  ctx.font = '9px monospace';
  ctx.fillText(maxV.toFixed(1), 2, padT + 7);
  ctx.fillText('0.0', 6, Math.min(h - 14, Math.max(18, y0 + 3)));
  ctx.fillText(minV.toFixed(1), 2, padT + plotH);
}

export function renderTaylorGreenChart(canvas: HTMLCanvasElement, history: TaylorGreenSample[]) {
  const ctx = canvas.getContext('2d')!;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;

  const padL = 32, padR = 8, padT = 8, padB = 14;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const toX = (k: number) => padL + (k / (history.length - 1)) * plotW;
  const toY = (ke: number) => padT + (1 - Math.min(0.26, Math.max(0, ke)) / 0.26) * plotH;

  // Exact analytical E(t) dashed curve
  ctx.strokeStyle = '#f97316';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  history.forEach((s, k) => {
    const x = toX(k), y = toY(s.exactKE);
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  // Simulated E(t) solid curve
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.75;
  ctx.beginPath();
  history.forEach((s, k) => {
    const x = toX(k), y = toY(s.simKE);
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = '#64748b';
  ctx.font = '9px monospace';
  ctx.fillText('0.25', 2, padT + 7);
  ctx.fillText('0.12', 2, padT + plotH * 0.5);
  ctx.fillText('0.00', 2, padT + plotH);
}

export function updateColorbar(visMode: number) {
  const titleEl = document.getElementById('cbTitle')!;
  const barEl = document.getElementById('cbBar')!;
  const minEl = document.getElementById('cbMin')!;
  const midEl = document.getElementById('cbMid')!;
  const maxEl = document.getElementById('cbMax')!;

  const viridisGrad = 'linear-gradient(to right, #440154, #482475, #20908c, #fde725)';
  const rdbuGrad = 'linear-gradient(to right, #0547ad, #141a24, #d91f2e)';
  const infernoGrad = 'linear-gradient(to right, #080812, #6b176e, #ed6121, #fcfca3)';

  if (visMode === 0) {
    titleEl.textContent = 'Vorticity ω [1/s]';
    barEl.style.background = rdbuGrad;
    minEl.textContent = '-16.7';
    midEl.textContent = '0.0';
    maxEl.textContent = '+16.7';
  } else if (visMode === 1) {
    titleEl.textContent = 'Velocity Magnitude |u| [m/s]';
    barEl.style.background = viridisGrad;
    minEl.textContent = '0.00';
    midEl.textContent = '0.67';
    maxEl.textContent = '1.33';
  } else if (visMode === 2) {
    titleEl.textContent = 'Pressure p [Pa]';
    barEl.style.background = rdbuGrad;
    minEl.textContent = '-0.40';
    midEl.textContent = '0.00';
    maxEl.textContent = '+0.40';
  } else if (visMode === 3) {
    titleEl.textContent = 'Passive Scalar Dye φ';
    barEl.style.background = infernoGrad;
    minEl.textContent = '0.00';
    midEl.textContent = '0.50';
    maxEl.textContent = '1.00';
  } else if (visMode === 4) {
    titleEl.textContent = 'LIC Streamlines + |u| [m/s]';
    barEl.style.background = viridisGrad;
    minEl.textContent = '0.00';
    midEl.textContent = '0.67';
    maxEl.textContent = '1.33';
  } else {
    titleEl.textContent = 'Isocontours + |u| [m/s]';
    barEl.style.background = viridisGrad;
    minEl.textContent = '0.00';
    midEl.textContent = '0.67';
    maxEl.textContent = '1.33';
  }
}
