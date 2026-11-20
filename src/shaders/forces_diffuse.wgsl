@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> uIn: array<f32>;
@group(0) @binding(2) var<storage, read> vIn: array<f32>;
@group(0) @binding(3) var<storage, read> solid: array<f32>;
@group(0) @binding(4) var<storage, read_write> uOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> vOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> dye: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let id = gid.y * params.nx + gid.x;
  let perX = isPeriodicX(params.preset);

  if (solid[id] > 0.5) {
    uOut[id] = 0.0;
    vOut[id] = 0.0;
    return;
  }

  // Explicit 5-point Laplacian diffusion: u += dt * nu * laplacian(u)
  let invDx2 = 1.0 / (params.dx * params.dx);
  let uC = uIn[id];
  let uL = uIn[idxP(i - 1, j, params.nx, params.ny, perX)];
  let uR = uIn[idxP(i + 1, j, params.nx, params.ny, perX)];
  let uB = uIn[idxP(i, j - 1, params.nx, params.ny, perX)];
  let uT = uIn[idxP(i, j + 1, params.nx, params.ny, perX)];
  var uVal = uC + params.dt * params.nu * (uL + uR + uB + uT - 4.0 * uC) * invDx2;

  let vC = vIn[id];
  let vL = vIn[idxP(i - 1, j, params.nx, params.ny, perX)];
  let vR = vIn[idxP(i + 1, j, params.nx, params.ny, perX)];
  let vB = vIn[idxP(i, j - 1, params.nx, params.ny, perX)];
  let vT = vIn[idxP(i, j + 1, params.nx, params.ny, perX)];
  var vVal = vC + params.dt * params.nu * (vL + vR + vB + vT - 4.0 * vC) * invDx2;

  // Interactive mouse force + dye injection
  if (params.mouseDown == 1u) {
    let px = (f32(gid.x) + 0.5) / f32(params.nx);
    let py = (f32(gid.y) + 0.5) / f32(params.ny);
    let d = distance(vec2<f32>(px, py), vec2<f32>(params.mouseX, params.mouseY));
    if (d < 0.025) {
      let w = exp(-d * d / 0.00018);
      uVal += params.mouseVx * w * 18.0 * params.dt;
      vVal += params.mouseVy * w * 18.0 * params.dt;
      dye[id] = clamp(dye[id] + w * 0.85, 0.0, 1.0);
    }
  }

  // Scenario-specific Boundary Conditions
  if (params.preset == 0u || params.preset == 3u) {
    // External wind tunnel (Cylinder / Airfoil)
    if (gid.x <= 1u) {
      uVal = 1.0;
      vVal = 0.0;
      let yNorm = (f32(gid.y) + 0.5) / f32(params.ny);
      if (abs(yNorm - 0.5) < 0.08 && fract(yNorm * 40.0) < 0.5) {
        dye[id] = 1.0;
      }
    } else if (gid.x >= params.nx - 1u) {
      uVal = uIn[idx(i - 1, j, params.nx, params.ny)];
      vVal = vIn[idx(i - 1, j, params.nx, params.ny)];
    }
    if (gid.y == 0u || gid.y >= params.ny - 1u) {
      vVal = 0.0;
    }
  } else if (params.preset == 1u) {
    // Lid-Driven Cavity: top wall U = 1.0, side/bottom walls no-slip
    if (gid.y == params.ny - 1u) {
      uVal = 1.0;
      vVal = 0.0;
    } else if (gid.y == 0u || gid.x == 0u || gid.x == params.nx - 1u) {
      uVal = 0.0;
      vVal = 0.0;
    }
  } else if (params.preset == 2u) {
    // Kelvin-Helmholtz: periodic X, free-slip Y
    if (gid.y == 0u || gid.y >= params.ny - 1u) {
      vVal = 0.0;
    }
  }

  // Zero normal flux into solid obstacles
  if (!perX && i > 0 && solid[idx(i - 1, j, params.nx, params.ny)] > 0.5) { uVal = 0.0; }
  if (j > 0 && solid[idx(i, j - 1, params.nx, params.ny)] > 0.5) { vVal = 0.0; }

  uOut[id] = uVal;
  vOut[id] = vVal;
}
