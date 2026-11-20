@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> p: array<f32>;
@group(0) @binding(4) var<storage, read_write> div: array<f32>;
@group(0) @binding(5) var<storage, read_write> rhs0: array<f32>;
@group(0) @binding(6) var<storage, read> solid: array<f32>;

@compute @workgroup_size(16, 16)
fn computeDivergence(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let id = gid.y * params.nx + gid.x;
  let perX = isPeriodicX(params.preset);

  if (solid[id] > 0.5) {
    div[id] = 0.0;
    rhs0[id] = 0.0;
    p[id] = 0.0;
    return;
  }

  let uR = u[idxP(i + 1, j, params.nx, params.ny, perX)];
  let uL = u[id];
  let vT = v[idxP(i, j + 1, params.nx, params.ny, perX)];
  let vB = v[id];

  let d = (uR - uL + vT - vB) / params.dx;
  div[id] = d;
  rhs0[id] = d / params.dt;
}

@compute @workgroup_size(16, 16)
fn projectVelocity(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let id = gid.y * params.nx + gid.x;
  let perX = isPeriodicX(params.preset);

  if (solid[id] > 0.5) {
    u[id] = 0.0;
    v[id] = 0.0;
    return;
  }

  let scale = params.dt / params.dx;

  if (perX) {
    let idL = idxP(i - 1, j, params.nx, params.ny, true);
    if (solid[idL] < 0.5) {
      u[id] -= scale * (p[id] - p[idL]);
    } else {
      u[id] = 0.0;
    }
  } else {
    if (gid.x > 0u && solid[idx(i - 1, j, params.nx, params.ny)] < 0.5) {
      u[id] -= scale * (p[id] - p[idx(i - 1, j, params.nx, params.ny)]);
    } else {
      u[id] = 0.0;
    }
  }

  if (gid.y > 0u && solid[idx(i, j - 1, params.nx, params.ny)] < 0.5) {
    v[id] -= scale * (p[id] - p[idx(i, j - 1, params.nx, params.ny)]);
  } else {
    v[id] = 0.0;
  }

  // Re-enforce exact Dirichlet boundary velocities
  if ((params.preset == 0u || params.preset == 3u) && gid.x <= 1u) {
    u[id] = 1.0;
    v[id] = 0.0;
  } else if (params.preset == 1u) {
    if (gid.y == params.ny - 1u) {
      u[id] = 1.0;
      v[id] = 0.0;
    } else if (gid.y == 0u || gid.x == 0u || gid.x == params.nx - 1u) {
      u[id] = 0.0;
      v[id] = 0.0;
    }
  }
}
