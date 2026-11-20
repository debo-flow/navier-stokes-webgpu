@group(0) @binding(0) var<uniform> params: SimParams;

fn gatherNeighbors(
  i: i32,
  j: i32,
  nx: u32,
  ny: u32,
  preset: u32,
  p: ptr<storage, array<f32>, read>,
  solid: ptr<storage, array<f32>, read>
) -> vec2<f32> {
  let perX = isPeriodicX(preset);
  let perY = isPeriodicY(preset);
  var sumP = 0.0;
  var count = 0.0;

  if (perX) {
    let idL = idxP(i - 1, j, nx, ny, true, perY);
    if ((*solid)[idL] < 0.5) { sumP += (*p)[idL]; count += 1.0; }
    let idR = idxP(i + 1, j, nx, ny, true, perY);
    if ((*solid)[idR] < 0.5) { sumP += (*p)[idR]; count += 1.0; }
  } else {
    if (i > 0) {
      let idL = idx(i - 1, j, nx, ny);
      if ((*solid)[idL] < 0.5) { sumP += (*p)[idL]; count += 1.0; }
    }
    if (u32(i) + 1u < nx) {
      let idR = idx(i + 1, j, nx, ny);
      if ((*solid)[idR] < 0.5) {
        let isTunnel = (preset == 0u || preset == 3u);
        if (!isTunnel || u32(i) + 1u < nx - 1u) {
          sumP += (*p)[idR];
        }
        count += 1.0;
      }
    }
  }

  if (perY) {
    let idB = idxP(i, j - 1, nx, ny, perX, true);
    if ((*solid)[idB] < 0.5) { sumP += (*p)[idB]; count += 1.0; }
    let idT = idxP(i, j + 1, nx, ny, perX, true);
    if ((*solid)[idT] < 0.5) { sumP += (*p)[idT]; count += 1.0; }
  } else {
    if (j > 0) {
      let idB = idx(i, j - 1, nx, ny);
      if ((*solid)[idB] < 0.5) { sumP += (*p)[idB]; count += 1.0; }
    }
    if (u32(j) + 1u < ny) {
      let idT = idx(i, j + 1, nx, ny);
      if ((*solid)[idT] < 0.5) { sumP += (*p)[idT]; count += 1.0; }
    }
  }

  return vec2<f32>(sumP, count);
}

// 1. Zero-initialize coarse error buffer
@group(0) @binding(1) var<storage, read_write> clearBuf: array<f32>;

@compute @workgroup_size(16, 16)
fn clearField(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny) { return; }
  clearBuf[gid.y * params.nx + gid.x] = 0.0;
}

// 2. Red-Black Gauss-Seidel Smoother
@group(0) @binding(1) var<storage, read_write> pSmooth: array<f32>;
@group(0) @binding(2) var<storage, read> rhsSmooth: array<f32>;
@group(0) @binding(3) var<storage, read> solidSmooth: array<f32>;

@compute @workgroup_size(16, 16)
fn smoothRBGS(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny) { return; }
  if (((gid.x + gid.y) & 1u) != params.rbColor) { return; }

  let i = i32(gid.x);
  let j = i32(gid.y);
  let id = gid.y * params.nx + gid.x;

  if (solidSmooth[id] > 0.5) {
    pSmooth[id] = 0.0;
    return;
  }

  let perX = isPeriodicX(params.preset);
  let perY = isPeriodicY(params.preset);
  var sumP = 0.0;
  var count = 0.0;

  if (perX) {
    let idL = idxP(i - 1, j, params.nx, params.ny, true, perY);
    if (solidSmooth[idL] < 0.5) { sumP += pSmooth[idL]; count += 1.0; }
    let idR = idxP(i + 1, j, params.nx, params.ny, true, perY);
    if (solidSmooth[idR] < 0.5) { sumP += pSmooth[idR]; count += 1.0; }
  } else {
    if (i > 0) {
      let idL = idx(i - 1, j, params.nx, params.ny);
      if (solidSmooth[idL] < 0.5) { sumP += pSmooth[idL]; count += 1.0; }
    }
    if (gid.x + 1u < params.nx) {
      let idR = idx(i + 1, j, params.nx, params.ny);
      if (solidSmooth[idR] < 0.5) {
        let isTunnel = (params.preset == 0u || params.preset == 3u);
        if (!isTunnel || gid.x + 1u < params.nx - 1u) {
          sumP += pSmooth[idR];
        }
        count += 1.0;
      }
    }
  }

  if (perY) {
    let idB = idxP(i, j - 1, params.nx, params.ny, perX, true);
    if (solidSmooth[idB] < 0.5) { sumP += pSmooth[idB]; count += 1.0; }
    let idT = idxP(i, j + 1, params.nx, params.ny, perX, true);
    if (solidSmooth[idT] < 0.5) { sumP += pSmooth[idT]; count += 1.0; }
  } else {
    if (j > 0) {
      let idB = idx(i, j - 1, params.nx, params.ny);
      if (solidSmooth[idB] < 0.5) { sumP += pSmooth[idB]; count += 1.0; }
    }
    if (gid.y + 1u < params.ny) {
      let idT = idx(i, j + 1, params.nx, params.ny);
      if (solidSmooth[idT] < 0.5) { sumP += pSmooth[idT]; count += 1.0; }
    }
  }

  if (count > 0.0) {
    let pGS = (sumP - params.dx * params.dx * rhsSmooth[id]) / count;
    pSmooth[id] = mix(pSmooth[id], pGS, 1.35);
  }
}

// 3. Residual computation: r = rhs - laplacian(p)
@group(0) @binding(1) var<storage, read> pRes: array<f32>;
@group(0) @binding(2) var<storage, read> rhsRes: array<f32>;
@group(0) @binding(3) var<storage, read> solidRes: array<f32>;
@group(0) @binding(4) var<storage, read_write> resOut: array<f32>;

@compute @workgroup_size(16, 16)
fn computeResidual(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let id = gid.y * params.nx + gid.x;

  if (solidRes[id] > 0.5) {
    resOut[id] = 0.0;
    return;
  }

  let nb = gatherNeighbors(i, j, params.nx, params.ny, params.preset, &pRes, &solidRes);
  if (nb.y > 0.0) {
    let lapP = (nb.x - nb.y * pRes[id]) / (params.dx * params.dx);
    resOut[id] = rhsRes[id] - lapP;
  } else {
    resOut[id] = 0.0;
  }
}

// 4. Restriction: downsample residual fine -> coarse
@group(0) @binding(1) var<storage, read> resFine: array<f32>;
@group(0) @binding(2) var<storage, read> solidFine: array<f32>;
@group(0) @binding(3) var<storage, read_write> rhsCoarse: array<f32>;
@group(0) @binding(4) var<storage, read> solidCoarse: array<f32>;

@compute @workgroup_size(16, 16)
fn restrictResidual(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.coarseNx || gid.y >= params.coarseNy) { return; }
  let cid = gid.y * params.coarseNx + gid.x;

  if (solidCoarse[cid] > 0.5) {
    rhsCoarse[cid] = 0.0;
    return;
  }

  let fx = gid.x * 2u;
  let fy = gid.y * 2u;
  var sum = 0.0;
  var w = 0.0;

  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      let fid = (fy + dy) * params.nx + (fx + dx);
      if (solidFine[fid] < 0.5) {
        sum += resFine[fid];
        w += 1.0;
      }
    }
  }

  rhsCoarse[cid] = select(0.0, sum / w, w > 0.0);
}

// 5. Prolongation: bilinearly interpolate coarse error -> add to fine p
@group(0) @binding(1) var<storage, read> errCoarse: array<f32>;
@group(0) @binding(2) var<storage, read> solidCoarseP: array<f32>;
@group(0) @binding(3) var<storage, read_write> pFine: array<f32>;
@group(0) @binding(4) var<storage, read> solidFineP: array<f32>;

@compute @workgroup_size(16, 16)
fn prolongateAdd(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny) { return; }
  let fid = gid.y * params.nx + gid.x;
  if (solidFineP[fid] > 0.5) { return; }

  let perX = isPeriodicX(params.preset);
  let perY = isPeriodicY(params.preset);
  var cx = 0.5 * f32(gid.x) - 0.25;
  if (!perX) { cx = clamp(cx, 0.0, f32(params.coarseNx) - 1.001); }
  var cy = 0.5 * f32(gid.y) - 0.25;
  if (!perY) { cy = clamp(cy, 0.0, f32(params.coarseNy) - 1.001); }

  let x0 = i32(floor(cx));
  let y0 = i32(floor(cy));
  let tx = cx - f32(x0);
  let ty = cy - f32(y0);

  let id00 = idxP(x0, y0, params.coarseNx, params.coarseNy, perX, perY);
  let id10 = idxP(x0 + 1, y0, params.coarseNx, params.coarseNy, perX, perY);
  let id01 = idxP(x0, y0 + 1, params.coarseNx, params.coarseNy, perX, perY);
  let id11 = idxP(x0 + 1, y0 + 1, params.coarseNx, params.coarseNy, perX, perY);

  let w00 = (1.0 - tx) * (1.0 - ty) * select(0.0, 1.0, solidCoarseP[id00] < 0.5);
  let w10 = tx * (1.0 - ty) * select(0.0, 1.0, solidCoarseP[id10] < 0.5);
  let w01 = (1.0 - tx) * ty * select(0.0, 1.0, solidCoarseP[id01] < 0.5);
  let w11 = tx * ty * select(0.0, 1.0, solidCoarseP[id11] < 0.5);
  let wSum = w00 + w10 + w01 + w11;

  if (wSum > 1e-5) {
    let corr = (w00 * errCoarse[id00] + w10 * errCoarse[id10] + w01 * errCoarse[id01] + w11 * errCoarse[id11]) / wSum;
    pFine[fid] += corr;
  }
}
