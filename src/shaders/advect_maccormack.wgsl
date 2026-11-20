@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> uIn: array<f32>;
@group(0) @binding(2) var<storage, read> vIn: array<f32>;
@group(0) @binding(3) var<storage, read> dyeIn: array<f32>;
@group(0) @binding(4) var<storage, read> solid: array<f32>;
@group(0) @binding(5) var<storage, read_write> uOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> vOut: array<f32>;
@group(0) @binding(7) var<storage, read_write> dyeOut: array<f32>;

fn traceRK2(gx: f32, gy: f32, dtGrid: f32) -> vec2<f32> {
  let u0 = sampleU(&uIn, gx, gy, params.nx, params.ny);
  let v0 = sampleV(&vIn, gx, gy, params.nx, params.ny);
  let midX = gx - 0.5 * dtGrid * u0;
  let midY = gy - 0.5 * dtGrid * v0;
  let u1 = sampleU(&uIn, midX, midY, params.nx, params.ny);
  let v1 = sampleV(&vIn, midX, midY, params.nx, params.ny);
  return vec2<f32>(gx - dtGrid * u1, gy - dtGrid * v1);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let id = gid.y * params.nx + gid.x;
  let perX = isPeriodicX(params.preset);
  let perY = isPeriodicY(params.preset);

  if (solid[id] > 0.5) {
    uOut[id] = 0.0;
    vOut[id] = 0.0;
    dyeOut[id] = 0.0;
    return;
  }

  let dtGrid = params.dt / params.dx;

  // 1. Advect U at left face (i, j + 0.5)
  let posU = vec2<f32>(f32(i), f32(j) + 0.5);
  let backU = traceRK2(posU.x, posU.y, dtGrid);
  let fwdU = traceRK2(backU.x, backU.y, -dtGrid);
  let uHat = sampleU(&uIn, backU.x, backU.y, params.nx, params.ny);
  let uFwd = sampleU(&uIn, fwdU.x, fwdU.y, params.nx, params.ny);
  let uNew = uHat + 0.5 * (uIn[id] - uFwd);

  let ux0 = i32(floor(backU.x));
  let uy0 = i32(floor(backU.y - 0.5));
  let u00 = uIn[idxP(ux0, uy0, params.nx, params.ny, perX, perY)];
  let u10 = uIn[idxP(ux0 + 1, uy0, params.nx, params.ny, perX, perY)];
  let u01 = uIn[idxP(ux0, uy0 + 1, params.nx, params.ny, perX, perY)];
  let u11 = uIn[idxP(ux0 + 1, uy0 + 1, params.nx, params.ny, perX, perY)];
  uOut[id] = clamp(uNew, min(min(u00, u10), min(u01, u11)), max(max(u00, u10), max(u01, u11)));

  // 2. Advect V at bottom face (i + 0.5, j)
  let posV = vec2<f32>(f32(i) + 0.5, f32(j));
  let backV = traceRK2(posV.x, posV.y, dtGrid);
  let fwdV = traceRK2(backV.x, backV.y, -dtGrid);
  let vHat = sampleV(&vIn, backV.x, backV.y, params.nx, params.ny);
  let vFwd = sampleV(&vIn, fwdV.x, fwdV.y, params.nx, params.ny);
  let vNew = vHat + 0.5 * (vIn[id] - vFwd);

  let vx0 = i32(floor(backV.x - 0.5));
  let vy0 = i32(floor(backV.y));
  let v00 = vIn[idxP(vx0, vy0, params.nx, params.ny, perX, perY)];
  let v10 = vIn[idxP(vx0 + 1, vy0, params.nx, params.ny, perX, perY)];
  let v01 = vIn[idxP(vx0, vy0 + 1, params.nx, params.ny, perX, perY)];
  let v11 = vIn[idxP(vx0 + 1, vy0 + 1, params.nx, params.ny, perX, perY)];
  vOut[id] = clamp(vNew, min(min(v00, v10), min(v01, v11)), max(max(v00, v10), max(v01, v11)));

  // 3. Advect passive dye at cell center (i + 0.5, j + 0.5)
  let posC = vec2<f32>(f32(i) + 0.5, f32(j) + 0.5);
  let backC = traceRK2(posC.x, posC.y, dtGrid);
  let fwdC = traceRK2(backC.x, backC.y, -dtGrid);
  let dHat = sampleCenter(&dyeIn, backC.x, backC.y, params.nx, params.ny);
  let dFwd = sampleCenter(&dyeIn, fwdC.x, fwdC.y, params.nx, params.ny);
  let dNew = dHat + 0.5 * (dyeIn[id] - dFwd);

  let dx0 = i32(floor(backC.x - 0.5));
  let dy0 = i32(floor(backC.y - 0.5));
  let d00 = dyeIn[idxP(dx0, dy0, params.nx, params.ny, perX, perY)];
  let d10 = dyeIn[idxP(dx0 + 1, dy0, params.nx, params.ny, perX, perY)];
  let d01 = dyeIn[idxP(dx0, dy0 + 1, params.nx, params.ny, perX, perY)];
  let d11 = dyeIn[idxP(dx0 + 1, dy0 + 1, params.nx, params.ny, perX, perY)];
  dyeOut[id] = clamp(dNew, min(min(d00, d10), min(d01, d11)), max(max(d00, d10), max(d01, d11)));
}
