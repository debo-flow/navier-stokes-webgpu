import commonWGSL from './shaders/common.wgsl?raw';
import advectWGSL from './shaders/advect_maccormack.wgsl?raw';
import forcesWGSL from './shaders/forces_diffuse.wgsl?raw';
import multigridWGSL from './shaders/multigrid.wgsl?raw';
import projectionWGSL from './shaders/projection.wgsl?raw';
import renderWGSL from './shaders/render.wgsl?raw';
import { nearestGhiaDataset, renderGhiaChart } from './validation/ghia1982';
import {
  AeroSample,
  TaylorGreenSample,
  computeAeroCoefficients,
  computeTaylorGreenMetrics,
  renderAeroChart,
  renderTaylorGreenChart,
  updateColorbar,
} from './validation/diagnostics';

interface SimState {
  nx: number;
  ny: number;
  re: number;
  vCycles: number;
  cfl: number;
  preset: number;
  visMode: number;
  paused: boolean;
  time: number;
  dt: number;
  mouseX: number;
  mouseY: number;
  mouseVx: number;
  mouseVy: number;
  mouseDown: number;
  hovering: boolean;
  cursorPxX: number;
  cursorPxY: number;
}

interface MGLevel {
  nx: number;
  ny: number;
  dx: number;
  pBuf: GPUBuffer;
  rhsBuf: GPUBuffer;
  resBuf: GPUBuffer;
  solidBuf: GPUBuffer;
  uniformRed: GPUBuffer;
  uniformBlack: GPUBuffer;
  clearBG: GPUBindGroup;
  smoothRedBG: GPUBindGroup;
  smoothBlackBG: GPUBindGroup;
  residualBG: GPUBindGroup;
  restrictBG?: GPUBindGroup;
  prolongateBG?: GPUBindGroup;
}

async function init() {
  if (!navigator.gpu) {
    alert('WebGPU is not supported in this browser. Try Chrome, Edge, or Firefox Nightly.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');
  const device = await adapter.requestDevice();

  const canvas = document.getElementById('cfd-canvas') as HTMLCanvasElement;
  const overlayCanvas = document.getElementById('overlay-canvas') as HTMLCanvasElement;
  const overlayCtx = overlayCanvas.getContext('2d')!;
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const state: SimState = {
    nx: 512,
    ny: 512,
    re: 400,
    vCycles: 6,
    cfl: 0.5,
    preset: 0,
    visMode: 0,
    paused: false,
    time: 0,
    dt: 0.001,
    mouseX: 0,
    mouseY: 0,
    mouseVx: 0,
    mouseVy: 0,
    mouseDown: 0,
    hovering: false,
    cursorPxX: 0,
    cursorPxY: 0,
  };

  // Compile shader modules
  const advectMod = device.createShaderModule({ code: commonWGSL + '\n' + advectWGSL });
  const forcesMod = device.createShaderModule({ code: commonWGSL + '\n' + forcesWGSL });
  const mgMod = device.createShaderModule({ code: commonWGSL + '\n' + multigridWGSL });
  const projMod = device.createShaderModule({ code: commonWGSL + '\n' + projectionWGSL });
  const renderMod = device.createShaderModule({ code: commonWGSL + '\n' + renderWGSL });

  // Compute & render pipelines
  const advectPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: advectMod, entryPoint: 'main' },
  });
  const forcesPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: forcesMod, entryPoint: 'main' },
  });
  const divPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: projMod, entryPoint: 'computeDivergence' },
  });
  const projectPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: projMod, entryPoint: 'projectVelocity' },
  });

  const mgClearPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: mgMod, entryPoint: 'clearField' },
  });
  const mgSmoothPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: mgMod, entryPoint: 'smoothRBGS' },
  });
  const mgResPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: mgMod, entryPoint: 'computeResidual' },
  });
  const mgRestrictPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: mgMod, entryPoint: 'restrictResidual' },
  });
  const mgProlongPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: mgMod, entryPoint: 'prolongateAdd' },
  });

  const renderPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderMod, entryPoint: 'vsMain' },
    fragment: { module: renderMod, entryPoint: 'fsMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  const uniformBuf = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let uBuf!: GPUBuffer, uTmpBuf!: GPUBuffer;
  let vBuf!: GPUBuffer, vTmpBuf!: GPUBuffer;
  let divBuf!: GPUBuffer;
  let dyeBuf!: GPUBuffer, dyeTmpBuf!: GPUBuffer;
  let readbackBuf!: GPUBuffer;
  let mgLevels: MGLevel[] = [];

  let advectBG!: GPUBindGroup, forcesBG!: GPUBindGroup;
  let divBG!: GPUBindGroup, projectBG!: GPUBindGroup;
  let renderBG!: GPUBindGroup;

  let cpuU = new Float32Array(0);
  let cpuV = new Float32Array(0);
  let cpuP = new Float32Array(0);
  let cpuDiv = new Float32Array(0);
  let cpuSolid = new Float32Array(0);
  let isReadingBack = false;
  let gridVersion = 0;

  let aeroHistory: AeroSample[] = [];
  let tgHistory: TaylorGreenSample[] = [];

  function characteristicLength(): number {
    if (state.preset === 0) return 0.11; // Cylinder diameter
    if (state.preset === 3) return 0.24; // NACA 0012 chord
    return 1.0;
  }

  function buildGridBuffers() {
    gridVersion++;
    const count = state.nx * state.ny;
    const bytes = count * 4;

    [uBuf, uTmpBuf, vBuf, vTmpBuf, divBuf, dyeBuf, dyeTmpBuf, readbackBuf].forEach((b) => b?.destroy());
    mgLevels.forEach((lvl) => {
      lvl.pBuf.destroy();
      lvl.rhsBuf.destroy();
      lvl.resBuf.destroy();
      lvl.solidBuf.destroy();
      lvl.uniformRed.destroy();
      lvl.uniformBlack.destroy();
    });
    mgLevels = [];

    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const uniformUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

    uBuf = device.createBuffer({ size: bytes, usage: storageUsage });
    uTmpBuf = device.createBuffer({ size: bytes, usage: storageUsage });
    vBuf = device.createBuffer({ size: bytes, usage: storageUsage });
    vTmpBuf = device.createBuffer({ size: bytes, usage: storageUsage });
    divBuf = device.createBuffer({ size: bytes, usage: storageUsage });
    dyeBuf = device.createBuffer({ size: bytes, usage: storageUsage });
    dyeTmpBuf = device.createBuffer({ size: bytes, usage: storageUsage });
    readbackBuf = device.createBuffer({
      size: bytes * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Build Geometric Multigrid hierarchy down to 16x16
    let lnx = state.nx;
    let lny = state.ny;
    let ldx = 1.0 / state.nx;
    while (lnx >= 16 && lny >= 16) {
      const lBytes = lnx * lny * 4;
      const pBuf = device.createBuffer({ size: lBytes, usage: storageUsage });
      const rhsBuf = device.createBuffer({ size: lBytes, usage: storageUsage });
      const resBuf = device.createBuffer({ size: lBytes, usage: storageUsage });
      const solidBuf = device.createBuffer({ size: lBytes, usage: storageUsage });
      const uniformRed = device.createBuffer({ size: 64, usage: uniformUsage });
      const uniformBlack = device.createBuffer({ size: 64, usage: uniformUsage });

      const clearBG = device.createBindGroup({
        layout: mgClearPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformRed } },
          { binding: 1, resource: { buffer: pBuf } },
        ],
      });
      const smoothRedBG = device.createBindGroup({
        layout: mgSmoothPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformRed } },
          { binding: 1, resource: { buffer: pBuf } },
          { binding: 2, resource: { buffer: rhsBuf } },
          { binding: 3, resource: { buffer: solidBuf } },
        ],
      });
      const smoothBlackBG = device.createBindGroup({
        layout: mgSmoothPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBlack } },
          { binding: 1, resource: { buffer: pBuf } },
          { binding: 2, resource: { buffer: rhsBuf } },
          { binding: 3, resource: { buffer: solidBuf } },
        ],
      });
      const residualBG = device.createBindGroup({
        layout: mgResPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformRed } },
          { binding: 1, resource: { buffer: pBuf } },
          { binding: 2, resource: { buffer: rhsBuf } },
          { binding: 3, resource: { buffer: solidBuf } },
          { binding: 4, resource: { buffer: resBuf } },
        ],
      });

      mgLevels.push({
        nx: lnx,
        ny: lny,
        dx: ldx,
        pBuf,
        rhsBuf,
        resBuf,
        solidBuf,
        uniformRed,
        uniformBlack,
        clearBG,
        smoothRedBG,
        smoothBlackBG,
        residualBG,
      });

      lnx >>= 1;
      lny >>= 1;
      ldx *= 2.0;
    }

    for (let l = 0; l < mgLevels.length - 1; l++) {
      const fine = mgLevels[l];
      const coarse = mgLevels[l + 1];

      fine.restrictBG = device.createBindGroup({
        layout: mgRestrictPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fine.uniformRed } },
          { binding: 1, resource: { buffer: fine.resBuf } },
          { binding: 2, resource: { buffer: fine.solidBuf } },
          { binding: 3, resource: { buffer: coarse.rhsBuf } },
          { binding: 4, resource: { buffer: coarse.solidBuf } },
        ],
      });
      fine.prolongateBG = device.createBindGroup({
        layout: mgProlongPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fine.uniformRed } },
          { binding: 1, resource: { buffer: coarse.pBuf } },
          { binding: 2, resource: { buffer: coarse.solidBuf } },
          { binding: 3, resource: { buffer: fine.pBuf } },
          { binding: 4, resource: { buffer: fine.solidBuf } },
        ],
      });
    }

    const fineLvl = mgLevels[0];

    cpuU = new Float32Array(count);
    cpuV = new Float32Array(count);
    cpuP = new Float32Array(count);
    cpuDiv = new Float32Array(count);
    cpuSolid = new Float32Array(count);

    advectBG = device.createBindGroup({
      layout: advectPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: uBuf } },
        { binding: 2, resource: { buffer: vBuf } },
        { binding: 3, resource: { buffer: dyeBuf } },
        { binding: 4, resource: { buffer: fineLvl.solidBuf } },
        { binding: 5, resource: { buffer: uTmpBuf } },
        { binding: 6, resource: { buffer: vTmpBuf } },
        { binding: 7, resource: { buffer: dyeTmpBuf } },
      ],
    });
    forcesBG = device.createBindGroup({
      layout: forcesPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: uTmpBuf } },
        { binding: 2, resource: { buffer: vTmpBuf } },
        { binding: 3, resource: { buffer: fineLvl.solidBuf } },
        { binding: 4, resource: { buffer: uBuf } },
        { binding: 5, resource: { buffer: vBuf } },
        { binding: 6, resource: { buffer: dyeTmpBuf } },
      ],
    });
    divBG = device.createBindGroup({
      layout: divPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: uBuf } },
        { binding: 2, resource: { buffer: vBuf } },
        { binding: 3, resource: { buffer: fineLvl.pBuf } },
        { binding: 4, resource: { buffer: divBuf } },
        { binding: 5, resource: { buffer: fineLvl.rhsBuf } },
        { binding: 6, resource: { buffer: fineLvl.solidBuf } },
      ],
    });
    // Only bind entries statically referenced by projectVelocity()
    projectBG = device.createBindGroup({
      layout: projectPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: uBuf } },
        { binding: 2, resource: { buffer: vBuf } },
        { binding: 3, resource: { buffer: fineLvl.pBuf } },
        { binding: 6, resource: { buffer: fineLvl.solidBuf } },
      ],
    });
    renderBG = device.createBindGroup({
      layout: renderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: uBuf } },
        { binding: 2, resource: { buffer: vBuf } },
        { binding: 3, resource: { buffer: fineLvl.pBuf } },
        { binding: 4, resource: { buffer: dyeTmpBuf } },
        { binding: 5, resource: { buffer: fineLvl.solidBuf } },
      ],
    });

    resetScenario();
  }

  function resetScenario() {
    const { nx, ny, preset } = state;
    const count = nx * ny;
    const uInit = new Float32Array(count);
    const vInit = new Float32Array(count);
    const pInit = new Float32Array(count);
    const dyeInit = new Float32Array(count);
    const solidInit = new Float32Array(count);

    const aoa = (12.0 * Math.PI) / 180.0;
    const cosA = Math.cos(aoa);
    const sinA = Math.sin(aoa);
    const chord = 0.24;
    const leX = 0.22;
    const leY = 0.50;
    const twoPi = 2.0 * Math.PI;

    for (let j = 0; j < ny; j++) {
      const y = (j + 0.5) / ny;
      const yv = j / ny;
      for (let i = 0; i < nx; i++) {
        const x = (i + 0.5) / nx;
        const xu = i / nx;
        const id = j * nx + i;

        if (preset === 0) {
          if (Math.hypot(x - 0.25, y - 0.5) < 0.055) {
            solidInit[id] = 1.0;
          } else {
            uInit[id] = 1.0;
          }
        } else if (preset === 1) {
          if (Math.abs(y - 0.5) < 0.015) dyeInit[id] = 0.85;
        } else if (preset === 2) {
          const width = 0.03;
          if (y > 0.25 && y < 0.75) {
            uInit[id] = 1.0;
            dyeInit[id] = 1.0;
          } else {
            uInit[id] = -1.0;
            dyeInit[id] = 0.0;
          }
          vInit[id] =
            0.08 *
            Math.sin(4.0 * Math.PI * x) *
            (Math.exp(-((y - 0.25) ** 2) / (width * width)) +
              Math.exp(-((y - 0.75) ** 2) / (width * width)));
        } else if (preset === 3) {
          const dx = x - leX;
          const dy = y - leY;
          const xc = (dx * cosA - dy * sinA) / chord;
          const yc = (dx * sinA + dy * cosA) / chord;
          if (xc >= 0.0 && xc <= 1.0) {
            const yt =
              5.0 *
              0.12 *
              (0.2969 * Math.sqrt(xc) -
                0.1260 * xc -
                0.3516 * xc * xc +
                0.2843 * xc ** 3 -
                0.1015 * xc ** 4);
            if (Math.abs(yc) <= yt) {
              solidInit[id] = 1.0;
            } else {
              uInit[id] = 1.0;
            }
          } else {
            uInit[id] = 1.0;
          }
        } else if (preset === 4) {
          // Exact Taylor-Green Vortex initialized on MAC staggered faces
          uInit[id] = Math.sin(twoPi * xu) * Math.cos(twoPi * y);
          vInit[id] = -Math.cos(twoPi * x) * Math.sin(twoPi * yv);
          pInit[id] = 0.25 * (Math.cos(4.0 * Math.PI * x) + Math.cos(4.0 * Math.PI * y));
          dyeInit[id] = 0.5 + 0.5 * Math.sin(twoPi * x) * Math.sin(twoPi * y);
        }
      }
    }

    cpuSolid.set(solidInit);
    device.queue.writeBuffer(uBuf, 0, uInit);
    device.queue.writeBuffer(vBuf, 0, vInit);
    device.queue.writeBuffer(mgLevels[0].pBuf, 0, pInit);
    device.queue.writeBuffer(dyeBuf, 0, dyeInit);
    device.queue.writeBuffer(dyeTmpBuf, 0, dyeInit);

    // Downsample solid masks down the Multigrid hierarchy
    let prevSolid = solidInit;
    device.queue.writeBuffer(mgLevels[0].solidBuf, 0, prevSolid);
    for (let l = 1; l < mgLevels.length; l++) {
      const lvl = mgLevels[l];
      const fine = mgLevels[l - 1];
      const coarseSolid = new Float32Array(lvl.nx * lvl.ny);
      for (let cj = 0; cj < lvl.ny; cj++) {
        for (let ci = 0; ci < lvl.nx; ci++) {
          const f00 = prevSolid[cj * 2 * fine.nx + ci * 2];
          const f10 = prevSolid[cj * 2 * fine.nx + ci * 2 + 1];
          const f01 = prevSolid[(cj * 2 + 1) * fine.nx + ci * 2];
          const f11 = prevSolid[(cj * 2 + 1) * fine.nx + ci * 2 + 1];
          coarseSolid[cj * lvl.nx + ci] = f00 + f10 + f01 + f11 >= 2.0 ? 1.0 : 0.0;
        }
      }
      device.queue.writeBuffer(lvl.solidBuf, 0, coarseSolid);
      prevSolid = coarseSolid;
    }

    // Toggle scenario-specific diagnostic cards
    document.getElementById('aeroCard')!.classList.toggle('hidden', preset !== 0 && preset !== 3);
    document.getElementById('ghiaCard')!.classList.toggle('hidden', preset !== 1);
    document.getElementById('tgCard')!.classList.toggle('hidden', preset !== 4);

    aeroHistory = [];
    tgHistory = [];
    state.time = 0;
    writeAllUniforms();
  }

  function writeAllUniforms() {
    const dx = 1.0 / state.nx;
    const nu = characteristicLength() / state.re;

    const writeBuf = (target: GPUBuffer, nx: number, ny: number, ldx: number, rb: number, cNx = 0, cNy = 0) => {
      const buf = new ArrayBuffer(64);
      const u32 = new Uint32Array(buf);
      const f32 = new Float32Array(buf);
      u32[0] = nx;
      u32[1] = ny;
      f32[2] = state.dt;
      f32[3] = ldx;
      f32[4] = nu;
      u32[5] = state.preset;
      u32[6] = state.visMode;
      u32[7] = rb;
      f32[8] = state.mouseX;
      f32[9] = state.mouseY;
      f32[10] = state.mouseVx;
      f32[11] = state.mouseVy;
      u32[12] = state.mouseDown;
      f32[13] = state.time;
      u32[14] = cNx;
      u32[15] = cNy;
      device.queue.writeBuffer(target, 0, buf);
    };

    writeBuf(uniformBuf, state.nx, state.ny, dx, 0);

    for (let l = 0; l < mgLevels.length; l++) {
      const lvl = mgLevels[l];
      const cNx = l + 1 < mgLevels.length ? mgLevels[l + 1].nx : 0;
      const cNy = l + 1 < mgLevels.length ? mgLevels[l + 1].ny : 0;
      writeBuf(lvl.uniformRed, lvl.nx, lvl.ny, lvl.dx, 0, cNx, cNy);
      writeBuf(lvl.uniformBlack, lvl.nx, lvl.ny, lvl.dx, 1, cNx, cNy);
    }
  }

  function smoothLevel(encoder: GPUCommandEncoder, lvl: MGLevel, sweeps: number) {
    const wgX = Math.ceil(lvl.nx / 16);
    const wgY = Math.ceil(lvl.ny / 16);
    for (let s = 0; s < sweeps; s++) {
      const pRed = encoder.beginComputePass();
      pRed.setPipeline(mgSmoothPipe);
      pRed.setBindGroup(0, lvl.smoothRedBG);
      pRed.dispatchWorkgroups(wgX, wgY);
      pRed.end();

      const pBlack = encoder.beginComputePass();
      pBlack.setPipeline(mgSmoothPipe);
      pBlack.setBindGroup(0, lvl.smoothBlackBG);
      pBlack.dispatchWorkgroups(wgX, wgY);
      pBlack.end();
    }
  }

  function stepSimulation() {
    writeAllUniforms();

    const wgX = Math.ceil(state.nx / 16);
    const wgY = Math.ceil(state.ny / 16);
    const encoder = device.createCommandEncoder();

    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(advectPipe);
      pass.setBindGroup(0, advectBG);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(forcesPipe);
      pass.setBindGroup(0, forcesBG);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(divPipe);
      pass.setBindGroup(0, divBG);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }

    const numLevels = mgLevels.length;
    for (let vc = 0; vc < state.vCycles; vc++) {
      for (let l = 0; l < numLevels - 1; l++) {
        const fine = mgLevels[l];
        const coarse = mgLevels[l + 1];
        const fWgX = Math.ceil(fine.nx / 16);
        const fWgY = Math.ceil(fine.ny / 16);
        const cWgX = Math.ceil(coarse.nx / 16);
        const cWgY = Math.ceil(coarse.ny / 16);

        smoothLevel(encoder, fine, 3);

        const resPass = encoder.beginComputePass();
        resPass.setPipeline(mgResPipe);
        resPass.setBindGroup(0, fine.residualBG);
        resPass.dispatchWorkgroups(fWgX, fWgY);
        resPass.end();

        const restPass = encoder.beginComputePass();
        restPass.setPipeline(mgRestrictPipe);
        restPass.setBindGroup(0, fine.restrictBG!);
        restPass.dispatchWorkgroups(cWgX, cWgY);
        restPass.end();

        const clrPass = encoder.beginComputePass();
        clrPass.setPipeline(mgClearPipe);
        clrPass.setBindGroup(0, coarse.clearBG);
        clrPass.dispatchWorkgroups(cWgX, cWgY);
        clrPass.end();
      }

      smoothLevel(encoder, mgLevels[numLevels - 1], 14);

      for (let l = numLevels - 2; l >= 0; l--) {
        const fine = mgLevels[l];
        const fWgX = Math.ceil(fine.nx / 16);
        const fWgY = Math.ceil(fine.ny / 16);

        const prolPass = encoder.beginComputePass();
        prolPass.setPipeline(mgProlongPipe);
        prolPass.setBindGroup(0, fine.prolongateBG!);
        prolPass.dispatchWorkgroups(fWgX, fWgY);
        prolPass.end();

        smoothLevel(encoder, fine, 3);
      }
    }

    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(projectPipe);
      pass.setBindGroup(0, projectBG);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }
    encoder.copyBufferToBuffer(dyeTmpBuf, 0, dyeBuf, 0, state.nx * state.ny * 4);

    device.queue.submit([encoder.finish()]);
    state.time += state.dt;
  }

  async function sampleDiagnostics() {
    if (isReadingBack) return;
    isReadingBack = true;

    const version = gridVersion;
    const rb = readbackBuf;
    const nx = state.nx;
    const ny = state.ny;
    const count = nx * ny;
    const bytes = count * 4;

    try {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(uBuf, 0, rb, 0, bytes);
      enc.copyBufferToBuffer(vBuf, 0, rb, bytes, bytes);
      enc.copyBufferToBuffer(mgLevels[0].pBuf, 0, rb, bytes * 2, bytes);
      enc.copyBufferToBuffer(divBuf, 0, rb, bytes * 3, bytes);
      device.queue.submit([enc.finish()]);

      await rb.mapAsync(GPUMapMode.READ);
      if (version !== gridVersion) {
        rb.unmap();
        return;
      }

      const mapped = new Float32Array(rb.getMappedRange());
      cpuU.set(mapped.subarray(0, count));
      cpuV.set(mapped.subarray(count, count * 2));
      cpuP.set(mapped.subarray(count * 2, count * 3));
      cpuDiv.set(mapped.subarray(count * 3, count * 4));
      rb.unmap();
    } catch {
      return;
    } finally {
      isReadingBack = false;
    }

    let maxU = 0.01;
    let maxDiv = 0.0;
    for (let k = 0; k < count; k += 4) {
      const mag = Math.hypot(cpuU[k], cpuV[k]);
      if (mag > maxU) maxU = mag;
      const d = Math.abs(cpuDiv[k]);
      if (d > maxDiv) maxDiv = d;
    }

    const dx = 1.0 / nx;
    const L = characteristicLength();
    const nu = L / state.re;
    const dtCFL = (state.cfl * dx) / maxU;
    const dtVisc = (0.2 * dx * dx) / Math.max(nu, 1e-7);
    state.dt = Math.min(dtCFL, dtVisc, 0.005);

    document.getElementById('telTime')!.textContent = `${state.time.toFixed(3)} s`;
    document.getElementById('telDt')!.textContent = `${state.dt.toExponential(2)} s`;
    document.getElementById('telU')!.textContent = maxU.toFixed(3);
    document.getElementById('telDiv')!.textContent = maxDiv.toExponential(2);

    // 1. Lift & Drag Coefficients for Cylinder (0) and NACA 0012 Airfoil (3)
    if (state.preset === 0 || state.preset === 3) {
      const { cd, cl } = computeAeroCoefficients(cpuU, cpuV, cpuP, cpuSolid, nx, ny, dx, nu, L);
      aeroHistory.push({ t: state.time, cd, cl });
      if (aeroHistory.length > 180) aeroHistory.shift();
      document.getElementById('aeroCd')!.textContent = cd.toFixed(3);
      document.getElementById('aeroCl')!.textContent = cl.toFixed(3);
      renderAeroChart(document.getElementById('aeroCanvas') as HTMLCanvasElement, aeroHistory);
    }

    // 2. Ghia et al. (1982) Benchmark for Lid-Driven Cavity (1)
    if (state.preset === 1) {
      const ref = nearestGhiaDataset(state.re);
      document.getElementById('ghiaTitle')!.textContent = `Ghia et al. (1982) — Re=${ref.re}`;
      const rms = renderGhiaChart(
        document.getElementById('ghiaCanvas') as HTMLCanvasElement,
        cpuU,
        nx,
        ny,
        state.re
      );
      document.getElementById('ghiaRms')!.textContent = `RMS: ${rms.toFixed(4)}`;
    }

    // 3. Taylor-Green Vortex Analytical Error & Kinetic Energy Decay (4)
    if (state.preset === 4) {
      const tg = computeTaylorGreenMetrics(cpuU, cpuV, nx, ny, nu, state.time);
      tgHistory.push(tg);
      if (tgHistory.length > 180) tgHistory.shift();
      document.getElementById('tgL2')!.textContent = `L2: ${tg.l2Error.toExponential(2)}`;
      renderTaylorGreenChart(document.getElementById('tgCanvas') as HTMLCanvasElement, tgHistory);
    }
  }

  function renderProbeOverlay() {
    const dpr = devicePixelRatio;
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;
    overlayCtx.clearRect(0, 0, w, h);

    if (!state.hovering || cpuU.length === 0) return;

    const px = state.cursorPxX * dpr;
    const py = state.cursorPxY * dpr;
    const gx = Math.min(state.nx - 2, Math.max(1, Math.floor(state.mouseX * state.nx)));
    const gy = Math.min(state.ny - 2, Math.max(1, Math.floor(state.mouseY * state.ny)));
    const id = gy * state.nx + gx;

    const uVal = cpuU[id];
    const vVal = cpuV[id];
    const pVal = cpuP[id];
    const dx = 1.0 / state.nx;
    const omega = (cpuV[id + 1] - cpuV[id - 1] - (cpuU[id + state.nx] - cpuU[id - state.nx])) / (2.0 * dx);

    overlayCtx.save();
    overlayCtx.strokeStyle = 'rgba(56, 189, 248, 0.9)';
    overlayCtx.lineWidth = 1.25 * dpr;
    const r = 9 * dpr;
    overlayCtx.beginPath();
    overlayCtx.arc(px, py, r, 0, Math.PI * 2);
    overlayCtx.moveTo(px - r * 1.6, py);
    overlayCtx.lineTo(px - r * 0.5, py);
    overlayCtx.moveTo(px + r * 0.5, py);
    overlayCtx.lineTo(px + r * 1.6, py);
    overlayCtx.moveTo(px, py - r * 1.6);
    overlayCtx.lineTo(px, py - r * 0.5);
    overlayCtx.moveTo(px, py + r * 0.5);
    overlayCtx.lineTo(px, py + r * 1.6);
    overlayCtx.stroke();

    const tag = `|u|=${Math.hypot(uVal, vVal).toFixed(2)}  ω=${omega.toFixed(1)}  p=${pVal.toFixed(3)}`;
    overlayCtx.font = `${10.5 * dpr}px ui-monospace, monospace`;
    const tw = overlayCtx.measureText(tag).width + 12 * dpr;
    const th = 20 * dpr;
    const bx = px + 14 * dpr + tw > w ? px - tw - 14 * dpr : px + 14 * dpr;
    const by = py - th - 10 * dpr < 0 ? py + 14 * dpr : py - th - 10 * dpr;

    overlayCtx.fillStyle = 'rgba(11, 14, 20, 0.85)';
    overlayCtx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
    overlayCtx.lineWidth = 1 * dpr;
    overlayCtx.beginPath();
    overlayCtx.roundRect(bx, by, tw, th, 4 * dpr);
    overlayCtx.fill();
    overlayCtx.stroke();

    overlayCtx.fillStyle = '#e2e8f0';
    overlayCtx.fillText(tag, bx + 6 * dpr, by + 13.5 * dpr);
    overlayCtx.restore();
  }

  function renderFrame() {
    const encoder = device.createCommandEncoder();
    const view = context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(renderPipe);
    pass.setBindGroup(0, renderBG);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);

    renderProbeOverlay();
  }

  function bindUI() {
    const bindSelect = (id: string, cb: (v: number) => void) => {
      document.getElementById(id)!.addEventListener('change', (e) => {
        cb(Number((e.target as HTMLSelectElement).value));
      });
    };
    const bindSlider = (id: string, labelId: string, fmt: (v: number) => string, cb: (v: number) => void) => {
      document.getElementById(id)!.addEventListener('input', (e) => {
        const val = Number((e.target as HTMLInputElement).value);
        document.getElementById(labelId)!.textContent = fmt(val);
        cb(val);
      });
    };

    bindSelect('preset', (v) => { state.preset = v; resetScenario(); });
    bindSelect('resolution', (v) => { state.nx = v; state.ny = v; buildGridBuffers(); });
    bindSelect('visMode', (v) => {
      state.visMode = v;
      updateColorbar(v);
      writeAllUniforms();
    });

    bindSlider('reSlider', 'reVal', (v) => `${v}`, (v) => { state.re = v; });
    bindSlider('iterSlider', 'iterVal', (v) => `${v}`, (v) => { state.vCycles = v; });
    bindSlider('cflSlider', 'cflVal', (v) => v.toFixed(2), (v) => { state.cfl = v; });

    document.getElementById('pauseBtn')!.addEventListener('click', (e) => {
      state.paused = !state.paused;
      (e.target as HTMLButtonElement).textContent = state.paused ? 'Resume' : 'Pause';
    });
    document.getElementById('resetBtn')!.addEventListener('click', () => resetScenario());

    let prevX = 0, prevY = 0;
    canvas.addEventListener('pointerdown', (e) => {
      state.mouseDown = 1;
      const r = canvas.getBoundingClientRect();
      prevX = (e.clientX - r.left) / r.width;
      prevY = 1.0 - (e.clientY - r.top) / r.height;
    });
    window.addEventListener('pointerup', () => { state.mouseDown = 0; });
    canvas.addEventListener('pointerleave', () => { state.hovering = false; });
    canvas.addEventListener('pointermove', (e) => {
      const r = canvas.getBoundingClientRect();
      state.hovering = true;
      state.cursorPxX = e.clientX - r.left;
      state.cursorPxY = e.clientY - r.top;

      const nx = state.cursorPxX / r.width;
      const ny = 1.0 - state.cursorPxY / r.height;
      state.mouseVx = (nx - prevX) * 60.0;
      state.mouseVy = (ny - prevY) * 60.0;
      state.mouseX = nx;
      state.mouseY = ny;
      prevX = nx;
      prevY = ny;

      const gx = Math.min(state.nx - 2, Math.max(1, Math.floor(nx * state.nx)));
      const gy = Math.min(state.ny - 2, Math.max(1, Math.floor(ny * state.ny)));
      const id = gy * state.nx + gx;
      if (cpuU.length > id) {
        const uVal = cpuU[id];
        const vVal = cpuV[id];
        const pVal = cpuP[id];
        const dx = 1.0 / state.nx;
        const omega =
          (cpuV[id + 1] - cpuV[id - 1] - (cpuU[id + state.nx] - cpuU[id - state.nx])) / (2.0 * dx);

        document.getElementById('probePos')!.textContent = `(${nx.toFixed(2)}, ${ny.toFixed(2)})`;
        document.getElementById('probeVel')!.textContent = `(${uVal.toFixed(2)}, ${vVal.toFixed(2)})`;
        document.getElementById('probeP')!.textContent = pVal.toFixed(4);
        document.getElementById('probeW')!.textContent = omega.toFixed(2);
      }
    });

    const resize = () => {
      const w = canvas.clientWidth * devicePixelRatio;
      const h = canvas.clientHeight * devicePixelRatio;
      canvas.width = w;
      canvas.height = h;
      overlayCanvas.width = w;
      overlayCanvas.height = h;
    };
    window.addEventListener('resize', resize);
    resize();
    updateColorbar(state.visMode);
  }

  bindUI();
  buildGridBuffers();

  let frame = 0;
  function loop() {
    if (!state.paused) {
      stepSimulation();
      stepSimulation();
      if (frame % 15 === 0) sampleDiagnostics();
    }
    renderFrame();
    frame++;
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

init();
