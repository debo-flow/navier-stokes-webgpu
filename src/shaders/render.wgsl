@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<f32>;
@group(0) @binding(4) var<storage, read> dye: array<f32>;
@group(0) @binding(5) var<storage, read> solid: array<f32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var out: VSOut;
  out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
  out.uv = positions[vid] * 0.5 + 0.5;
  return out;
}

fn viridis(tIn: f32) -> vec3<f32> {
  let t = clamp(tIn, 0.0, 1.0);
  let c0 = vec3<f32>(0.267, 0.004, 0.329);
  let c1 = vec3<f32>(0.282, 0.140, 0.457);
  let c2 = vec3<f32>(0.127, 0.566, 0.550);
  let c3 = vec3<f32>(0.993, 0.906, 0.144);
  if (t < 0.33) { return mix(c0, c1, t / 0.33); }
  if (t < 0.66) { return mix(c1, c2, (t - 0.33) / 0.33); }
  return mix(c2, c3, (t - 0.66) / 0.34);
}

fn rdbu(vIn: f32) -> vec3<f32> {
  let t = clamp(vIn, -1.0, 1.0);
  let neg = vec3<f32>(0.02, 0.28, 0.68);
  let mid = vec3<f32>(0.08, 0.10, 0.14);
  let pos = vec3<f32>(0.85, 0.12, 0.18);
  if (t < 0.0) { return mix(mid, neg, pow(-t, 0.75)); }
  return mix(mid, pos, pow(t, 0.75));
}

fn inferno(tIn: f32) -> vec3<f32> {
  let t = clamp(tIn, 0.0, 1.0);
  let c0 = vec3<f32>(0.03, 0.03, 0.07);
  let c1 = vec3<f32>(0.42, 0.09, 0.43);
  let c2 = vec3<f32>(0.93, 0.38, 0.13);
  let c3 = vec3<f32>(0.99, 0.99, 0.64);
  if (t < 0.33) { return mix(c0, c1, t / 0.33); }
  if (t < 0.66) { return mix(c1, c2, (t - 0.33) / 0.33); }
  return mix(c2, c3, (t - 0.66) / 0.34);
}

fn hash21(p: vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn licNoise(pos: vec2<f32>) -> f32 {
  let i = floor(pos);
  let f = fract(pos);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2<f32>(1.0, 0.0));
  let c = hash21(i + vec2<f32>(0.0, 1.0));
  let d = hash21(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn computeLIC(gx: f32, gy: f32) -> f32 {
  let freq = 0.75;
  var acc = licNoise(vec2<f32>(gx, gy) * freq);
  var wSum = 1.0;

  var pFwd = vec2<f32>(gx, gy);
  for (var s = 1; s <= 12; s++) {
    let vel = vec2<f32>(
      sampleU(&u, pFwd.x, pFwd.y, params.nx, params.ny),
      sampleV(&v, pFwd.x, pFwd.y, params.nx, params.ny)
    );
    let len = length(vel);
    if (len < 1e-4) { break; }
    pFwd += (vel / len) * 0.75;
    let w = cos(f32(s) * 0.12);
    acc += licNoise(pFwd * freq) * w;
    wSum += w;
  }

  var pBack = vec2<f32>(gx, gy);
  for (var s = 1; s <= 12; s++) {
    let vel = vec2<f32>(
      sampleU(&u, pBack.x, pBack.y, params.nx, params.ny),
      sampleV(&v, pBack.x, pBack.y, params.nx, params.ny)
    );
    let len = length(vel);
    if (len < 1e-4) { break; }
    pBack -= (vel / len) * 0.75;
    let w = cos(f32(s) * 0.12);
    acc += licNoise(pBack * freq) * w;
    wSum += w;
  }

  return smoothstep(0.28, 0.72, acc / wSum);
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4<f32> {
  let gx = in.uv.x * f32(params.nx);
  let gy = in.uv.y * f32(params.ny);
  let i = i32(clamp(gx, 0.0, f32(params.nx) - 1.0));
  let j = i32(clamp(gy, 0.0, f32(params.ny) - 1.0));
  let id = u32(j) * params.nx + u32(i);

  if (solid[id] > 0.5) {
    return vec4<f32>(0.22, 0.27, 0.34, 1.0);
  }

  let uc = sampleU(&u, gx, gy, params.nx, params.ny);
  let vc = sampleV(&v, gx, gy, params.nx, params.ny);
  let speed = length(vec2<f32>(uc, vc));

  let dvdx = (sampleV(&v, gx + 1.0, gy, params.nx, params.ny) - sampleV(&v, gx - 1.0, gy, params.nx, params.ny)) / (2.0 * params.dx);
  let dudy = (sampleU(&u, gx, gy + 1.0, params.nx, params.ny) - sampleU(&u, gx, gy - 1.0, params.nx, params.ny)) / (2.0 * params.dx);
  let omega = dvdx - dudy;

  var color = vec3<f32>(0.0);

  if (params.visMode == 0u) {
    color = rdbu(omega * 0.06);
  } else if (params.visMode == 1u) {
    color = viridis(speed * 0.75);
  } else if (params.visMode == 2u) {
    let pc = sampleCenter(&p, gx, gy, params.nx, params.ny);
    color = rdbu(pc * 2.5);
  } else if (params.visMode == 3u) {
    let dc = sampleCenter(&dye, gx, gy, params.nx, params.ny);
    color = inferno(dc);
  } else if (params.visMode == 4u) {
    let lic = computeLIC(gx, gy);
    let base = viridis(speed * 0.75);
    color = base * (0.45 + 0.75 * lic);
  } else {
    color = viridis(speed * 0.75);
    let contour = abs(fract(speed * 12.0) - 0.5);
    let line = smoothstep(0.04, 0.0, contour);
    color = mix(color, vec3<f32>(1.0), line * 0.45);
  }

  return vec4<f32>(color, 1.0);
}
