struct SimParams {
  nx: u32,
  ny: u32,
  dt: f32,
  dx: f32,
  nu: f32,
  preset: u32,      // 0: cylinder, 1: cavity, 2: kelvin, 3: airfoil, 4: taylor-green
  visMode: u32,     // 0: vorticity, 1: speed, 2: pressure, 3: dye, 4: LIC, 5: contours
  rbColor: u32,     // 0: red, 1: black
  mouseX: f32,
  mouseY: f32,
  mouseVx: f32,
  mouseVy: f32,
  mouseDown: u32,
  time: f32,
  coarseNx: u32,
  coarseNy: u32,
};

fn isPeriodicX(preset: u32) -> bool {
  return preset == 2u || preset == 4u;
}

fn isPeriodicY(preset: u32) -> bool {
  return preset == 4u;
}

fn idxP(x: i32, y: i32, nx: u32, ny: u32, periodicX: bool, periodicY: bool) -> u32 {
  var cx = x;
  if (periodicX) {
    cx = ((x % i32(nx)) + i32(nx)) % i32(nx);
  } else {
    cx = clamp(x, 0, i32(nx) - 1);
  }
  var cy = y;
  if (periodicY) {
    cy = ((y % i32(ny)) + i32(ny)) % i32(ny);
  } else {
    cy = clamp(y, 0, i32(ny) - 1);
  }
  return u32(cy) * nx + u32(cx);
}

fn idx(x: i32, y: i32, nx: u32, ny: u32) -> u32 {
  return idxP(x, y, nx, ny, false, false);
}

fn sampleCenter(field: ptr<storage, array<f32>, read>, gx: f32, gy: f32, nx: u32, ny: u32) -> f32 {
  let perX = isPeriodicX(params.preset);
  let perY = isPeriodicY(params.preset);
  var x = gx - 0.5;
  if (!perX) { x = clamp(x, 0.0, f32(nx) - 1.001); }
  var y = gy - 0.5;
  if (!perY) { y = clamp(y, 0.0, f32(ny) - 1.001); }

  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let fx = x - f32(x0);
  let fy = y - f32(y0);

  let v00 = (*field)[idxP(x0, y0, nx, ny, perX, perY)];
  let v10 = (*field)[idxP(x0 + 1, y0, nx, ny, perX, perY)];
  let v01 = (*field)[idxP(x0, y0 + 1, nx, ny, perX, perY)];
  let v11 = (*field)[idxP(x0 + 1, y0 + 1, nx, ny, perX, perY)];

  return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

fn sampleU(u: ptr<storage, array<f32>, read>, gx: f32, gy: f32, nx: u32, ny: u32) -> f32 {
  let perX = isPeriodicX(params.preset);
  let perY = isPeriodicY(params.preset);
  var x = gx;
  if (!perX) { x = clamp(x, 0.0, f32(nx) - 1.001); }
  var y = gy - 0.5;
  if (!perY) { y = clamp(y, 0.0, f32(ny) - 1.001); }

  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let fx = x - f32(x0);
  let fy = y - f32(y0);

  let v00 = (*u)[idxP(x0, y0, nx, ny, perX, perY)];
  let v10 = (*u)[idxP(x0 + 1, y0, nx, ny, perX, perY)];
  let v01 = (*u)[idxP(x0, y0 + 1, nx, ny, perX, perY)];
  let v11 = (*u)[idxP(x0 + 1, y0 + 1, nx, ny, perX, perY)];

  return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

fn sampleV(v: ptr<storage, array<f32>, read>, gx: f32, gy: f32, nx: u32, ny: u32) -> f32 {
  let perX = isPeriodicX(params.preset);
  let perY = isPeriodicY(params.preset);
  var x = gx - 0.5;
  if (!perX) { x = clamp(x, 0.0, f32(nx) - 1.001); }
  var y = gy;
  if (!perY) { y = clamp(y, 0.0, f32(ny) - 1.001); }

  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let fx = x - f32(x0);
  let fy = y - f32(y0);

  let v00 = (*v)[idxP(x0, y0, nx, ny, perX, perY)];
  let v10 = (*v)[idxP(x0 + 1, y0, nx, ny, perX, perY)];
  let v01 = (*v)[idxP(x0, y0 + 1, nx, ny, perX, perY)];
  let v11 = (*v)[idxP(x0 + 1, y0 + 1, nx, ny, perX, perY)];

  return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}
