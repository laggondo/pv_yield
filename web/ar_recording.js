// AR video recording on the phone (#46, #47): an `immersive-ar` WebXR session (Chrome on Android, ARCore) with camera
// access. ARCore tracks the phone's camera in metres, with y up, from the position where the session started (the
// reference position, origin of the recording). Keyframes are kept by movement (> 5 cm or > 5° since the last one, at
// most 4 per second, up to 300): the camera image (read back from WebGL, stored as JPEG), the camera pose, the
// projection matrix and the latest compass reading. GPS is recorded at the start. The compass readings turn the AR
// coordinates to north (heading offset, estimated in Python, core/ar_recording.py), so the frames can be shown on the
// sky map like photos. The recording also notes how things went on the phone (diagnostics: image size, read-back time,
// compass events during the session, ...), for the feasibility checks of #47.
// After Stop, and when a recording is loaded (project, browser storage), the section shows an inspection: key figures,
// the camera path from above, two of the frames and the frames on the sky map. The point cloud follows in #48.

const RECORDING_FORMAT_VERSION = 1;
const MAX_FRAMES = 300;
const MIN_MOVE_M = 0.05;             // a new keyframe when the camera moved more than this ...
const MIN_TURN_DEG = 5;              // ... or turned more than this since the last one,
const MIN_INTERVAL_S = 0.25;         // but at most 4 per second
const JPEG_QUALITY = 0.85;
const STATUS_INTERVAL_MS = 200;
const COMPASS_AFTER_WAIT_MS = 2000;  // after the session, wait this long for a compass reading
const SKY_MAP_FRAME_PX = 480;        // longer side of the frames as merged onto the sky map
const SKY_MAP_MAX_FRAMES = 120;      // frames on the sky map, chosen to look in different directions ...
const SKY_MAP_MIN_TURN_DEG = 3;      // ... at least this far apart
const PATH_PLOT_PX = 600;

const frameFileName = number => `recording/frame_${number}.jpg`;
const degrees = radians => radians * 180 / Math.PI;
const format = (value, digits) => Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

// Whether this browser can record: WebXR with immersive AR (Chrome on Android with ARCore); not on iPhones or computers.
export async function arSupported() {
  if (!navigator.xr || !window.isSecureContext) return false;
  // Throws e.g. where a permissions policy blocks WebXR: then it can't record either.
  return navigator.xr.isSessionSupported("immersive-ar").catch(() => false);
}

// Draws the camera image (an opaque WebXR texture) into a texture of our own and reads its pixels back (RGBA, the
// bottom row first, as WebGL stores it).
function createCameraReader(gl) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(`Could not compile the shader for the camera image: ${gl.getShaderInfoLog(shader)}`);
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, "attribute vec2 position; varying vec2 uv; void main() { uv = position * 0.5 + 0.5; gl_Position = vec4(position, 0.0, 1.0); }"));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, "precision mediump float; varying vec2 uv; uniform sampler2D camera; void main() { gl_FragColor = vec4(texture2D(camera, uv).rgb, 1.0); }"));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`Could not link the shader for the camera image: ${gl.getProgramInfoLog(program)}`);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const positionLocation = gl.getAttribLocation(program, "position");
  const cameraLocation = gl.getUniformLocation(program, "camera");
  const target = gl.createTexture();
  const framebuffer = gl.createFramebuffer();
  let targetSize = "";
  return (cameraTexture, width, height, restoreFramebuffer) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    if (targetSize !== `${width}x${height}`) {
      gl.bindTexture(gl.TEXTURE_2D, target);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      for (const [name, value] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, name, value);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
      targetSize = `${width}x${height}`;
    }
    gl.viewport(0, 0, width, height);
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cameraTexture);
    gl.uniform1i(cameraLocation, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, restoreFramebuffer);
    return pixels;
  };
}

// JPEG of pixels read back from WebGL (bottom row first, so the rows are flipped).
async function encodeJpeg(pixels, width, height) {
  const image = new ImageData(width, height);
  const rowBytes = 4 * width;
  for (let row = 0; row < height; row++) image.data.set(pixels.subarray((height - 1 - row) * rowBytes, (height - row) * rowBytes), row * rowBytes);
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(width, height);
    canvas.getContext("2d").putImageData(image, 0, 0);
    return canvas.convertToBlob({ type: "image/jpeg", quality: JPEG_QUALITY });
  }
  const canvas = Object.assign(document.createElement("canvas"), { width, height });
  canvas.getContext("2d").putImageData(image, 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Could not encode a frame as JPEG")), "image/jpeg", JPEG_QUALITY));
}

// The phone's position from GPS, as stored with the recording; an error message instead if there is none.
function currentGps() {
  return new Promise(resolve => {
    if (!navigator.geolocation) return resolve({ error: "this browser has no location service" });
    navigator.geolocation.getCurrentPosition(({ coords, timestamp }) => resolve({
      latitude: coords.latitude, longitude: coords.longitude, altitude_m: coords.altitude, accuracy_m: coords.accuracy, altitude_accuracy_m: coords.altitudeAccuracy, time: new Date(timestamp).toISOString(),
    }), error => resolve({ error: error.message }), { enableHighAccuracy: true, timeout: 30_000, maximumAge: 0 });
  });
}

const mean = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const timingStats = values => values.length ? { mean: Math.round(mean(values)), max: Math.round(Math.max(...values)), n: values.length } : null;

// Angle (degrees) between two orientations given as quaternions {x, y, z, w}.
function turnDeg(a, b) {
  const dot = Math.abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
  return degrees(2 * Math.acos(Math.min(1, dot)));
}

// Create the recorder on the page's elements: `call` runs a worker action, `whileBusy` runs an action with the page
// marked busy, `editor` is the sky editor (frames on the sky map), `site()` gives the form's site, `onRecorded(data)` is
// called after a new recording was made (e.g. to set the site from its GPS position), `onChanged()` after the
// recording changed by the user (new recording, heading offset or reference note changed, removed), to store it.
export function createArRecorder({ element, call, report, fail, whileBusy, editor, site, onRecorded, onChanged }) {
  let recording = null;                // {data: content of recording.json, frames: [Blob]} or null
  let summary = null;                  // the latest inspection (Python's recordingSummary)
  let skyMapFrames = new Map();        // frame index -> {width, height, data, scale}: frames decoded for the sky map
  let thumbnailUrls = [];
  let active = null;                   // state of the running AR session

  // ---- Recording ----

  function screenAngle() {
    return screen.orientation?.angle ?? window.orientation ?? 0;
  }

  const relativeTime = time => (time - active.startTime) / 1000;

  function onCompass(event) {
    if (event.alpha === null || event.beta === null || event.gamma === null) return;
    const reading = { alpha_deg: event.alpha, beta_deg: event.beta, gamma_deg: event.gamma, time: event.timeStamp, screen_angle_deg: screenAngle() };
    if (!active.started) active.compassBefore = reading;
    else if (!active.ended) active.compassEvents++;
    else if (!active.compassAfter) {
      active.compassAfter = reading;
      active.compassAfterArrived?.();
    }
    active.compass = reading;
  }

  function onOrientation() {
    if (active.started && !active.ended) active.orientationEvents++;
  }

  function compassEntry(reading) {
    if (!reading) return null;
    const { time, ...angles } = reading;
    return { ...angles, time_s: relativeTime(time) };
  }

  function isKeyframe(transform, time) {
    const last = active.lastKept;
    if (!last) return true;
    if ((time - last.time) / 1000 < MIN_INTERVAL_S) return false;
    const { position: p, orientation } = transform;
    return Math.hypot(p.x - last.position.x, p.y - last.position.y, p.z - last.position.z) > MIN_MOVE_M || turnDeg(orientation, last.orientation) > MIN_TURN_DEG;
  }

  // Keep a frame: read the camera image back, note the pose; the JPEG is encoded in the background.
  function keepFrame(view, time, emulatedPosition) {
    const { width, height } = view.camera;
    const start = performance.now();
    const texture = active.binding.getCameraImage(view.camera);
    const pixels = active.reader(texture, width, height, active.layer.framebuffer);
    active.readbackMs.push(performance.now() - start);
    const { position: p, orientation: q } = view.transform;
    active.frames.push({ file: frameFileName(active.frames.length + 1), time_s: relativeTime(time), position_m: [p.x, p.y, p.z], orientation_xyzw: [q.x, q.y, q.z, q.w], emulated_position: emulatedPosition,
      width, height, projection_matrix: Array.from(view.projectionMatrix), screen_angle_deg: screenAngle(), compass: compassEntry(active.compass) });
    const encodeStart = performance.now();
    active.blobs.push(encodeJpeg(pixels, width, height).then(blob => {
      active.encodeMs.push(performance.now() - encodeStart);
      return blob;
    }));
    active.lastKept = { position: { x: p.x, y: p.y, z: p.z }, orientation: { x: q.x, y: q.y, z: q.z, w: q.w }, time };
  }

  // Extent of the camera path so far (diagonal of its bounding box), from all poses, not only the kept ones.
  function extendPath(position) {
    const box = active.box;
    for (const axis of ["x", "y", "z"]) {
      box.min[axis] = Math.min(box.min[axis], position[axis]);
      box.max[axis] = Math.max(box.max[axis], position[axis]);
    }
    return Math.hypot(box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z);
  }

  function showRecordingStatus(time, baseline, trackingLost) {
    if (time - active.lastStatus < STATUS_INTERVAL_MS) return;
    active.lastStatus = time;
    element("ar-status").textContent = `${active.frames.length} frames · moved ${format(baseline, 2)} m · tracking ${trackingLost === null ? "starting" : trackingLost ? "LOST" : "ok"}`
      + (active.frames.length >= MAX_FRAMES ? ` · limit of ${MAX_FRAMES} frames reached: stop now` : "");
  }

  function onXrFrame(time, frame) {
    if (active.ended) return;
    active.session.requestAnimationFrame(onXrFrame);
    active.xrFrames++;
    const gl = active.gl;
    // The page draws nothing over the camera image (the browser shows it): a transparent layer.
    gl.bindFramebuffer(gl.FRAMEBUFFER, active.layer.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const pose = frame.getViewerPose(active.space);
    if (!pose) return showRecordingStatus(time, 0, null);
    const view = pose.views[0];
    const baseline = extendPath(view.transform.position);
    if (pose.emulatedPosition) active.trackingLostFrames++;
    if (!view.camera) active.framesWithoutCamera++;
    else if (active.frames.length < MAX_FRAMES && isKeyframe(view.transform, time)) keepFrame(view, time, pose.emulatedPosition);
    showRecordingStatus(time, baseline, pose.emulatedPosition);
  }

  // Start recording; must run right in the tap's handler (the browser asks for camera access).
  async function start() {
    const overlay = element("ar-overlay");
    active = { started: false, ended: false, compass: null, compassBefore: null, compassAfter: null, compassEvents: 0, orientationEvents: 0, frames: [], blobs: [], lastKept: null, xrFrames: 0,
      framesWithoutCamera: 0, trackingLostFrames: 0, readbackMs: [], encodeMs: [], lastStatus: 0, box: { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } }, contextLost: false };
    window.addEventListener("deviceorientationabsolute", onCompass);
    window.addEventListener("deviceorientation", onOrientation);
    overlay.hidden = false;
    element("ar-status").textContent = "Starting ...";
    let session;
    try {
      session = await navigator.xr.requestSession("immersive-ar", { requiredFeatures: ["camera-access", "local"], optionalFeatures: ["dom-overlay"], domOverlay: { root: overlay } });
    } catch (error) {
      overlay.hidden = true;
      removeListeners();
      throw new Error(`Could not start the AR session: ${error.message ?? error} (it needs Chrome on Android with Google Play Services for AR, and access to the camera)`);
    }
    Object.assign(active, { session, started: true, startTime: performance.now(), created: new Date().toISOString(), gps: currentGps() });
    session.addEventListener("end", () => whileBusy(finish));
    try {
      const canvas = document.createElement("canvas");
      canvas.addEventListener("webglcontextlost", () => {
        active.contextLost = true;
        report("The WebGL context was lost during the AR session");
      });
      active.gl = canvas.getContext("webgl2", { xrCompatible: true }) ?? canvas.getContext("webgl", { xrCompatible: true });
      if (!active.gl) throw new Error("This browser offers no WebGL, which the AR session needs");
      active.layer = new XRWebGLLayer(session, active.gl);
      session.updateRenderState({ baseLayer: active.layer });
      active.binding = new XRWebGLBinding(session, active.gl);
      active.space = await session.requestReferenceSpace("local");
      active.reader = createCameraReader(active.gl);
    } catch (error) {
      active.setupFailed = true;
      session.end();
      throw error;
    }
    active.diagnostics = { dom_overlay: session.domOverlayState?.type ?? "not granted", environment_blend_mode: session.environmentBlendMode ?? null,
      webgl: typeof WebGL2RenderingContext !== "undefined" && active.gl instanceof WebGL2RenderingContext ? "webgl2" : "webgl", compass_event: "ondeviceorientationabsolute" in window };
    session.requestAnimationFrame(onXrFrame);
    report(`AR session started (DOM overlay: ${active.diagnostics.dom_overlay}, ${active.diagnostics.webgl})`);
  }

  function stop() {
    active?.session?.end();
  }

  function removeListeners() {
    window.removeEventListener("deviceorientationabsolute", onCompass);
    window.removeEventListener("deviceorientation", onOrientation);
  }

  // After the session ended (Stop, or the phone's back button): wait for the JPEGs, a compass reading after the session
  // and GPS, then keep the recording and show the inspection.
  async function finish() {
    active.ended = true;
    active.endTime = performance.now();
    element("ar-overlay").hidden = true;
    if (active.setupFailed) {
      removeListeners();
      active = null;
      return;
    }
    const blobs = await Promise.all(active.blobs);
    if (!active.compassAfter) await new Promise(resolve => {
      active.compassAfterArrived = resolve;
      setTimeout(resolve, COMPASS_AFTER_WAIT_MS);
    });
    removeListeners();
    const gps = await active.gps;
    const frames = active.frames;
    report(`AR session ended: ${frames.length} frames kept of ${active.xrFrames} in ${format((active.endTime - active.startTime) / 1000, 1)} s${gps.error ? `; no GPS position: ${gps.error}` : ""}`);
    if (!frames.length) throw new Error("The AR session ended without frames: hold the phone still for a moment at the start until tracking begins, and check that camera access was allowed");
    const durationS = (active.endTime - active.startTime) / 1000;
    const data = {
      format_version: RECORDING_FORMAT_VERSION, kind: "ar_recording", created: active.created, user_agent: navigator.userAgent, reference_note: element("recording-reference").value.trim(),
      gps: gps.error ? null : gps, heading_offset_deg: null, heading_source: null,
      compass_before: compassEntry(active.compassBefore), compass_after: compassEntry(active.compassAfter),
      camera: { width: frames[0].width, height: frames[0].height, projection_matrix: frames[0].projection_matrix },
      diagnostics: { ...active.diagnostics, camera_image: `${frames[0].width} x ${frames[0].height}`, readback_ms: timingStats(active.readbackMs), encode_ms: timingStats(active.encodeMs),
        jpeg_kb: Math.round(mean(blobs.map(blob => blob.size)) / 1024), xr_frames: active.xrFrames, xr_frames_per_s: Math.round(active.xrFrames / durationS),
        xr_frames_without_camera: active.framesWithoutCamera, xr_frames_tracking_lost: active.trackingLostFrames, compass_events_during_session: active.compassEvents,
        orientation_events_during_session: active.orientationEvents, compass_rate_per_s: Math.round(active.compassEvents / durationS), webgl_context_lost: active.contextLost,
        max_frames_reached: frames.length >= MAX_FRAMES, gps_error: gps.error ?? null, duration_s: Math.round(10 * durationS) / 10,
        // As "moved" in the status line: extent of the camera path over all poses, not only the kept frames.
        moved_all_poses_m: Math.hypot(active.box.max.x - active.box.min.x, active.box.max.y - active.box.min.y, active.box.max.z - active.box.min.z) },
      frames,
    };
    active = null;
    await setRecording({ data, frames: blobs });
    await onRecorded(data);
    await onChanged();
    element("recording-inspection").scrollIntoView({ block: "start" });
  }

  // ---- Inspection ----

  // The recording to show (null: none), e.g. a new one, from a project or from the browser's storage.
  async function setRecording(newRecording) {
    recording = newRecording;
    summary = null;
    skyMapFrames = new Map();
    thumbnailUrls.forEach(url => URL.revokeObjectURL(url));
    thumbnailUrls = [];
    element("recording-inspection").hidden = !recording;
    element("recording-remove").disabled = !recording;
    if (recording) element("recording-reference").value = recording.data.reference_note ?? "";
    if (!recording) {
      editor.setFrameSources([]);
      return;
    }
    await inspect();
    showFrames();
  }

  // Summarize the recording (Python) with the heading offset (by hand, else from the compass), and show it.
  async function inspect() {
    const { latitude, longitude } = site();
    const { data } = recording;
    summary = await call("recordingSummary", data, data.heading_source === "manual" ? data.heading_offset_deg : null, latitude, longitude);
    element("recording-heading").value = Math.round(summary.heading_offset_deg * 10) / 10;
    showSummary();
    drawPath();
    await showOnSkyMap();
  }

  function showSummary() {
    const { data } = recording, s = summary, estimate = s.compass_estimate;
    const heading = s.heading_source === "manual" ? "set by hand"
      : s.heading_source === "none" ? "no compass readings: set it by hand"
        : `from ${estimate.n_readings} compass readings${s.heading_source === "compass_before_after" ? " just before and after the session" : ""}, spread ${format(estimate.spread_deg, 1)}°`;
    const gps = data.gps ? `GPS at the start: ${format(data.gps.latitude, 5)}, ${format(data.gps.longitude, 5)} (±${format(data.gps.accuracy_m, 0)} m)${s.gps_distance_km === null ? "" : `, ${format(s.gps_distance_km, s.gps_distance_km < 1 ? 2 : 1)} km from the site${s.gps_distance_km > 1 ? ": check the site" : ""}`}` : "no GPS position";
    element("recording-summary").innerHTML = [
      `${s.n_frames} frames over ${format(s.duration_s, 1)} s (recorded ${data.created ? `${data.created.slice(0, 16).replace("T", " ")} UTC` : "?"})${s.n_tracking_lost ? `; tracking lost in ${s.n_tracking_lost} frames` : ""}`,
      `Camera path: extent (baseline) ${format(s.baseline_m, 2)} m, length ${format(s.path_length_m, 2)} m, start to end ${format(s.start_to_end_m, 2)} m`,
      `Heading at the start ${format(s.heading_offset_deg, 1)}° (${heading}); the frames cover ${format(100 * s.covered_fraction, 0)} % of the sky`,
      gps, `Reference position: ${data.reference_note || "not described"}`,
    ].map(line => `<div>${line.replace(/</g, "&lt;")}</div>`).join("");
    const rounded = (key, value) => typeof value === "number" ? Math.round(value * 1000) / 1000 : value;
    element("recording-diagnostics").textContent = JSON.stringify({ ...data.diagnostics, camera_path_m: { baseline: s.baseline_m, length: s.path_length_m, start_to_end: s.start_to_end_m }, projection: s.projection, compass_estimate: s.compass_estimate, user_agent: data.user_agent }, rounded, 2);
  }

  // The camera path from above (east right, north up), in metres, with the reference position (start) and the end.
  function drawPath() {
    const canvas = element("recording-path"), context = canvas.getContext("2d");
    canvas.width = canvas.height = PATH_PLOT_PX;
    const scale = canvas.width / Math.max(1, canvas.getBoundingClientRect().width || canvas.width);
    const path = summary.path;
    const east = path.map(point => point[0]), north = path.map(point => point[1]);
    const span = Math.max(1, Math.max(...east) - Math.min(...east), Math.max(...north) - Math.min(...north)) * 1.2;
    const centre = [(Math.max(...east) + Math.min(...east)) / 2, (Math.max(...north) + Math.min(...north)) / 2];
    const pixelsPerMetre = PATH_PLOT_PX / span;
    const toPixel = ([x, y]) => [PATH_PLOT_PX / 2 + (x - centre[0]) * pixelsPerMetre, PATH_PLOT_PX / 2 - (y - centre[1]) * pixelsPerMetre];
    context.fillStyle = "#fff";
    context.fillRect(0, 0, PATH_PLOT_PX, PATH_PLOT_PX);
    // Grid lines every metre (every 10 m for long paths).
    const step = span > 20 ? 10 : 1;
    context.strokeStyle = "#e4e4e4";
    context.lineWidth = scale;
    for (let value = Math.floor((centre[0] - span / 2) / step) * step; value <= centre[0] + span / 2; value += step) {
      const [x] = toPixel([value, 0]);
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, PATH_PLOT_PX); context.stroke();
    }
    for (let value = Math.floor((centre[1] - span / 2) / step) * step; value <= centre[1] + span / 2; value += step) {
      const [, y] = toPixel([0, value]);
      context.beginPath(); context.moveTo(0, y); context.lineTo(PATH_PLOT_PX, y); context.stroke();
    }
    context.strokeStyle = "#d95f02";
    context.lineWidth = 3 * scale;
    context.beginPath();
    path.forEach((point, index) => index ? context.lineTo(...toPixel(point)) : context.moveTo(...toPixel(point)));
    context.stroke();
    for (const [point, colour] of [[[0, 0], "#2e7d32"], [path.at(-1), "#c62828"]]) {
      context.fillStyle = colour;
      context.beginPath();
      context.arc(...toPixel(point), 7 * scale, 0, 2 * Math.PI);
      context.fill();
    }
    element("recording-path-legend").textContent = `Camera path from above (north up), grid lines every ${step} m; green: reference position (start), red: end.`;
  }

  // The first and the middle frame as images.
  function showFrames() {
    const container = element("recording-frames");
    container.innerHTML = "";
    const indices = [...new Set([0, Math.floor(recording.frames.length / 2)])];
    for (const index of indices) {
      const url = URL.createObjectURL(recording.frames[index]);
      thumbnailUrls.push(url);
      const figure = document.createElement("figure");
      const view = summary.views[index];
      figure.append(Object.assign(document.createElement("img"), { src: url, alt: `Frame ${index + 1}` }),
        Object.assign(document.createElement("figcaption"), { textContent: `Frame ${index + 1}: azimuth ${format(view.azimuth_deg, 0)}°, elevation ${format(view.elevation_deg, 0)}°, field of view ${format(view.fov_deg, 0)}°` }));
      container.appendChild(figure);
    }
  }

  // Frames looking in different directions (greedy, in recording order), at most SKY_MAP_MAX_FRAMES.
  function framesForSkyMap() {
    const chosen = [], minDot = Math.cos(SKY_MAP_MIN_TURN_DEG * Math.PI / 180);
    summary.cameras.forEach((camera, index) => {
      if (recording.data.frames[index].emulated_position || chosen.length >= SKY_MAP_MAX_FRAMES) return;
      const forward = camera.forward;
      if (chosen.every(other => forward[0] * summary.cameras[other].forward[0] + forward[1] * summary.cameras[other].forward[1] + forward[2] * summary.cameras[other].forward[2] < minDot)) chosen.push(index);
    });
    return chosen;
  }

  // Decode a frame at most SKY_MAP_FRAME_PX large for the sky map (kept for later changes of the heading offset).
  async function decodedFrame(index) {
    if (!skyMapFrames.has(index)) {
      const bitmap = await createImageBitmap(recording.frames[index]);
      const scale = Math.min(1, SKY_MAP_FRAME_PX / Math.max(bitmap.width, bitmap.height));
      const canvas = Object.assign(document.createElement("canvas"), { width: Math.round(bitmap.width * scale), height: Math.round(bitmap.height * scale) });
      const context = canvas.getContext("2d");
      context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();
      skyMapFrames.set(index, { width: canvas.width, height: canvas.height, data: context.getImageData(0, 0, canvas.width, canvas.height).data, scale });
    }
    return skyMapFrames.get(index);
  }

  async function showOnSkyMap() {
    const sources = [];
    for (const index of framesForSkyMap()) {
      const { width, height, data, scale } = await decodedFrame(index);
      const camera = summary.cameras[index];
      sources.push({ camera: { ...camera, focal_length_px: camera.focal_length_px * scale }, width, height, data });
    }
    editor.setFrameSources(sources);
  }

  // A heading offset set by hand turns all frames on the sky map; `fromCompass` (or an empty field) goes back to the
  // compass's estimate.
  async function headingChanged(fromCompass = false) {
    if (!recording) return;
    const text = element("recording-heading").value.trim();
    const value = Number(text);
    Object.assign(recording.data, fromCompass || text === "" || Number.isNaN(value) ? { heading_offset_deg: null, heading_source: null } : { heading_offset_deg: value, heading_source: "manual" });
    await inspect();
    report(`AR recording: heading at the start ${format(summary.heading_offset_deg, 1)}° (${summary.heading_source === "manual" ? "set by hand" : "from the compass"})`);
    await onChanged();
  }

  async function referenceChanged() {
    if (!recording) return;
    recording.data.reference_note = element("recording-reference").value.trim();
    showSummary();
    await onChanged();
  }

  async function remove() {
    await setRecording(null);
    report("AR recording removed");
    await onChanged();
  }

  element("ar-stop").addEventListener("click", stop);
  element("recording-heading").addEventListener("change", () => whileBusy(() => headingChanged()));
  element("recording-heading-compass").addEventListener("click", () => whileBusy(() => headingChanged(true)));
  element("recording-reference").addEventListener("change", () => referenceChanged().catch(fail));
  element("recording-remove").addEventListener("click", () => whileBusy(remove));
  // Copy the diagnostics as text, e.g. to paste them into an issue.
  element("recording-diagnostics-copy").addEventListener("click", async () => {
    const button = element("recording-diagnostics-copy");
    await navigator.clipboard.writeText(element("recording-diagnostics").textContent).then(() => {
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = "Copy"; }, 1500);
    }, fail);
  });

  return {
    start, stop, setRecording,
    // Refresh the inspection, e.g. after the site changed (distance of the GPS position).
    refresh: async () => { if (recording) await inspect(); },
    get recording() { return recording; },
  };
}
