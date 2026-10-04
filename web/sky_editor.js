// Marking obstructed sky patches by hand (photo method, #17): on photos taken with the phone's camera (camera
// orientation from the phone's sensors) or loaded from files, and on a map of the sky hemisphere. The sky patches are
// drawn over the view; tapping a patch toggles it, dragging marks or frees all patches it passes over. Loaded
// obstructed sky descriptions (e.g. from LiDAR) are the start state, and the edits are baked into the flags (#12).
// The camera geometry (sensor angles to camera orientation, projection of the sky nodes to pixels) is computed in
// Python (core/photo.py); this module only draws and handles input. Photos are assumed to be taken from the panel
// position: no offset is applied (#11).

const PHOTO_SET_FORMAT_VERSION = 1;
const MAX_PHOTO_PX = 1600;           // longer side of stored photos, to keep memory and saved files small
const SKY_MAP_PX = 800;
const SKY_MAP_RADIUS = SKY_MAP_PX / 2 - 40;   // radius of the horizon in the sky map; the compass letters sit outside it
const OBSTRUCTED_FILL = "rgba(220, 30, 30, 0.45)";
const COMPASS = [["N", 0], ["E", 90], ["S", 180], ["W", 270]];

// Create the editor on the page's elements; `call` runs a worker action, `report` logs, `onApplied(summary)` is called
// after the marked flags were sent to Python as the new obstructed sky description.
export function createSkyEditor({ element, call, report, fail, defaultFov, nSkyNodes, onApplied }) {
  const canvas = element("editor-canvas");
  const context = canvas.getContext("2d");
  let sky = null;                      // {nodes, triangles}
  let flags = null;                    // Uint8Array, one entry per patch
  let skyMapPixels = null;             // node positions in the sky map
  const views = [{ kind: "sky_map", name: "Sky map" }];
  let current = 0;
  const methodsUsed = new Set();
  let stroke = null;                   // {value, changed} while dragging
  const camera = { stream: null, orientation: null, listener: null, eventName: null, projection: null, running: false, requestInFlight: false };

  // ---- Geometry ----

  // Polar map of the hemisphere as in the sky plot: north up, east right, radius proportional to the zenith angle.
  function skyMapPosition([x, y, z]) {
    const zenith = Math.acos(Math.max(-1, Math.min(1, z))) * 180 / Math.PI;
    const azimuth = Math.atan2(x, y);
    const radius = SKY_MAP_RADIUS * zenith / 90;
    return [SKY_MAP_PX / 2 + radius * Math.sin(azimuth), SKY_MAP_PX / 2 - radius * Math.cos(azimuth)];
  }

  // Corner pixels of each patch drawable in a view (null if a corner lies behind the camera).
  function patchCorners(view) {
    if (view.kind === "sky_map") return sky.triangles.map(triangle => triangle.map(node => skyMapPixels[node]));
    const projection = view.projection;
    if (!projection) return [];
    return sky.triangles.map(triangle => triangle.every(node => projection.in_front[node]) ? triangle.map(node => projection.pixels[node]) : null);
  }

  function insideTriangle([px, py], [[ax, ay], [bx, by], [cx, cy]]) {
    const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
    const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
    const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
    return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
  }

  function patchAt(view, point) {
    const corners = patchCorners(view);
    for (let index = 0; index < corners.length; index++) if (corners[index] && insideTriangle(point, corners[index])) return index;
    return -1;
  }

  // ---- Drawing ----

  // Canvas pixels per CSS pixel as displayed, so lines and letters keep their size on screen whatever the image size.
  function displayScale(target) {
    const displayedWidth = target.getBoundingClientRect().width;
    return displayedWidth > 0 ? target.width / displayedWidth : 1;
  }

  // Draw the patches (obstructed filled red, all outlined) and the compass markers onto a 2D context; `scale` is the
  // canvas's display scale.
  function drawPatches(target, corners, scale, markers) {
    target.lineWidth = scale;
    target.strokeStyle = "rgba(255, 255, 255, 0.75)";
    target.fillStyle = OBSTRUCTED_FILL;
    corners.forEach((triangle, index) => {
      if (!triangle) return;
      target.beginPath();
      target.moveTo(...triangle[0]);
      target.lineTo(...triangle[1]);
      target.lineTo(...triangle[2]);
      target.closePath();
      if (flags[index]) target.fill();
      target.stroke();
    });
    target.font = `bold ${Math.round(16 * scale)}px system-ui, sans-serif`;
    target.textAlign = "center";
    target.textBaseline = "middle";
    for (const { label, x, y } of markers) {
      target.lineWidth = 3 * scale;
      target.strokeStyle = "black";
      target.strokeText(label, x, y);
      target.fillStyle = "yellow";
      target.fillText(label, x, y);
    }
  }

  function draw() {
    if (!sky) return;
    const view = views[current];
    if (view.kind === "sky_map") {
      canvas.width = canvas.height = SKY_MAP_PX;
      context.fillStyle = "#5b7fa6";
      context.fillRect(0, 0, SKY_MAP_PX, SKY_MAP_PX);
      const markers = COMPASS.map(([label, azimuth]) => {
        const radius = SKY_MAP_RADIUS + 20;
        return { label, x: SKY_MAP_PX / 2 + radius * Math.sin(azimuth * Math.PI / 180), y: SKY_MAP_PX / 2 - radius * Math.cos(azimuth * Math.PI / 180) };
      });
      drawPatches(context, patchCorners(view), displayScale(canvas), markers);
    } else {
      canvas.width = view.image.width;
      canvas.height = view.image.height;
      context.drawImage(view.image, 0, 0);
      drawPatches(context, patchCorners(view), displayScale(canvas), view.projection?.markers ?? []);
    }
    const count = flags.reduce((sum, flag) => sum + flag, 0);
    element("editor-summary").textContent = `${count} of ${flags.length} sky patches marked as obstructed`;
  }

  // ---- View list and photo settings ----

  function showViews() {
    element("editor-views").innerHTML = "";
    views.forEach((view, index) => {
      const button = Object.assign(document.createElement("button"), { textContent: view.name, className: index === current ? "small selected" : "small" });
      button.addEventListener("click", () => select(index));
      element("editor-views").appendChild(button);
    });
    const view = views[current];
    element("photo-settings").hidden = view.kind !== "photo";
    if (view.kind === "photo") {
      for (const key of ["azimuth_deg", "elevation_deg", "roll_deg", "fov_deg"]) element(`photo-${key}`).value = Math.round(view.view[key] * 10) / 10;
    }
  }

  function select(index) {
    current = index;
    showViews();
    draw();
  }

  // Project the sky nodes into a photo (Python); repeated requests while one runs are merged into one.
  async function project(view) {
    view.projectionWanted = true;
    if (view.projecting) return;
    view.projecting = true;
    try {
      while (view.projectionWanted) {
        view.projectionWanted = false;
        view.projection = await call("projectSky", view.view);
      }
    } finally {
      view.projecting = false;
    }
    if (views[current] === view) draw();
  }

  function photoSettingChanged() {
    const view = views[current];
    if (view.kind !== "photo") return;
    for (const key of ["azimuth_deg", "elevation_deg", "roll_deg", "fov_deg"]) {
      const value = Number(element(`photo-${key}`).value);
      if (element(`photo-${key}`).value.trim() !== "" && !Number.isNaN(value)) view.view[key] = value;
    }
    project(view).catch(fail);
  }

  // A photo (canvas, at most MAX_PHOTO_PX) with its camera view; selected for marking.
  async function addPhoto(image, view, name, taken) {
    const scale = Math.min(1, MAX_PHOTO_PX / Math.max(image.width, image.height));
    const photoCanvas = document.createElement("canvas");
    photoCanvas.width = Math.round(image.width * scale);
    photoCanvas.height = Math.round(image.height * scale);
    photoCanvas.getContext("2d").drawImage(image, 0, 0, photoCanvas.width, photoCanvas.height);
    const photo = { kind: "photo", name, taken, image: photoCanvas, view: { ...view, width: photoCanvas.width, height: photoCanvas.height } };
    views.push(photo);
    select(views.length - 1);
    await project(photo);
    report(`Photo ${name}: azimuth ${photo.view.azimuth_deg.toFixed(1)}°, elevation ${photo.view.elevation_deg.toFixed(1)}°, roll ${photo.view.roll_deg.toFixed(1)}°, field of view ${photo.view.fov_deg}°`);
  }

  function removePhoto() {
    if (views[current].kind !== "photo") return;
    views.splice(current, 1);
    select(Math.min(current, views.length - 1));
  }

  // ---- Marking ----

  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
  }

  function markAt(event) {
    const index = patchAt(views[current], canvasPoint(event));
    if (index < 0) return;
    if (stroke.value === null) stroke.value = flags[index] ? 0 : 1;
    if (flags[index] !== stroke.value) {
      flags[index] = stroke.value;
      stroke.changed = true;
      draw();
    }
  }

  // Send the flags to Python as the new obstructed sky description, with the contributing methods and the photos' views.
  async function apply() {
    const photos = views.filter(view => view.kind === "photo").map(view => ({ name: view.name, taken: view.taken, ...view.view }));
    const summary = await call("applyEdits", Array.from(flags), [...methodsUsed], { photos });
    await onApplied(summary);
  }

  canvas.addEventListener("pointerdown", event => {
    if (!sky) return;
    canvas.setPointerCapture(event.pointerId);
    stroke = { value: null, changed: false };
    markAt(event);
  });
  canvas.addEventListener("pointermove", event => { if (stroke) markAt(event); });
  const endStroke = () => {
    if (!stroke) return;
    const changed = stroke.changed;
    stroke = null;
    if (!changed) return;
    methodsUsed.add(views[current].kind);
    apply().catch(fail);
  };
  canvas.addEventListener("pointerup", endStroke);
  canvas.addEventListener("pointercancel", endStroke);

  // ---- Camera ----

  // Latest absolute device orientation; iOS reports the compass heading separately (webkitCompassHeading).
  function onOrientation(event) {
    if (event.alpha === null || event.beta === null || event.gamma === null) return;
    const absolute = event.absolute || camera.eventName === "deviceorientationabsolute";
    const alpha = event.webkitCompassHeading !== undefined ? 360 - event.webkitCompassHeading : event.alpha;
    camera.orientation = { alpha_deg: alpha, beta_deg: event.beta, gamma_deg: event.gamma, absolute: absolute || event.webkitCompassHeading !== undefined };
  }

  function screenAngle() {
    return screen.orientation?.angle ?? window.orientation ?? 0;
  }

  function liveView() {
    const video = element("camera-video");
    const { alpha_deg, beta_deg, gamma_deg } = camera.orientation;
    return { alpha_deg, beta_deg, gamma_deg, screen_angle_deg: screenAngle(), fov_deg: defaultFov(), width: video.videoWidth, height: video.videoHeight };
  }

  // Overlay the sky patches on the live camera image, one projection at a time.
  async function liveOverlay() {
    if (!camera.running) return;
    const video = element("camera-video");
    const overlay = element("camera-overlay");
    if (camera.orientation && video.videoWidth && !camera.requestInFlight) {
      camera.requestInFlight = true;
      try {
        camera.projection = await call("projectSky", liveView());
      } catch (error) {
        camera.running = false;
        camera.requestInFlight = false;
        return fail(error);
      }
      camera.requestInFlight = false;
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      const overlayContext = overlay.getContext("2d");
      overlayContext.clearRect(0, 0, overlay.width, overlay.height);
      drawPatches(overlayContext, patchCorners({ kind: "photo", projection: camera.projection }), displayScale(overlay), camera.projection.markers);
      const { azimuth_deg, elevation_deg, roll_deg } = camera.projection.view;
      element("camera-status").textContent = `Camera: azimuth ${azimuth_deg.toFixed(0)}°, elevation ${elevation_deg.toFixed(0)}°, roll ${roll_deg.toFixed(0)}°${camera.orientation.absolute ? "" : " (no compass: correct the azimuth after taking the photo)"}`;
    }
    requestAnimationFrame(liveOverlay);
  }

  async function startCamera() {
    // iOS asks for permission to use the motion sensors, which must happen right after the tap.
    if (typeof DeviceOrientationEvent !== "undefined" && typeof DeviceOrientationEvent.requestPermission === "function") {
      const permission = await DeviceOrientationEvent.requestPermission();
      if (permission !== "granted") report("No access to the motion sensors: set the camera orientation by hand after taking the photo");
    }
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser offers no camera access (it needs https or localhost); load a photo file instead");
    camera.eventName = "ondeviceorientationabsolute" in window ? "deviceorientationabsolute" : "deviceorientation";
    camera.listener = onOrientation;
    window.addEventListener(camera.eventName, camera.listener);
    camera.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1440 } }, audio: false });
    const video = element("camera-video");
    video.srcObject = camera.stream;
    await video.play();
    element("camera").hidden = false;
    element("editor-view").hidden = true;
    camera.running = true;
    element("camera-status").textContent = "Waiting for the motion sensors ...";
    report(`Camera started (${video.videoWidth} x ${video.videoHeight} pixels, orientation from ${camera.eventName})`);
    requestAnimationFrame(liveOverlay);
  }

  function stopCamera() {
    camera.running = false;
    camera.stream?.getTracks().forEach(track => track.stop());
    camera.stream = null;
    if (camera.listener) window.removeEventListener(camera.eventName, camera.listener);
    camera.listener = null;
    element("camera").hidden = true;
    element("editor-view").hidden = false;
  }

  async function takePhoto() {
    const video = element("camera-video");
    const frame = document.createElement("canvas");
    frame.width = video.videoWidth;
    frame.height = video.videoHeight;
    frame.getContext("2d").drawImage(video, 0, 0);
    // The orientation at the moment of the shot; without sensors, a default view to correct by hand.
    const view = camera.orientation ? (await call("projectSky", liveView())).view : { azimuth_deg: 180, elevation_deg: 30, roll_deg: 0, fov_deg: defaultFov() };
    const taken = new Date().toISOString();
    stopCamera();
    await addPhoto(frame, view, `Photo ${views.length}`, taken);
  }

  // ---- Files: photos and photo sets ----

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`Could not read the image ${url.slice(0, 80)}`));
      image.src = url;
    });
  }

  // An image file (camera orientation entered by hand) or a photo set saved from this page.
  async function loadPhotoFile(file) {
    if (file.name.toLowerCase().endsWith(".json")) {
      const data = JSON.parse(await file.text());
      if (data.kind !== "photo_set") throw new Error(`${file.name} is not a photo set saved from this page (kind ${JSON.stringify(data.kind)})`);
      if (data.format_version > PHOTO_SET_FORMAT_VERSION) throw new Error(`${file.name} has photo set format version ${data.format_version}; this page reads up to ${PHOTO_SET_FORMAT_VERSION}`);
      for (const photo of data.photos) await addPhoto(await loadImage(photo.image), photo.view, photo.name, photo.taken);
      report(`Loaded ${data.photos.length} photos from ${file.name}`);
      return;
    }
    const url = URL.createObjectURL(file);
    try {
      await addPhoto(await loadImage(url), { azimuth_deg: 180, elevation_deg: 30, roll_deg: 0, fov_deg: defaultFov() }, file.name, new Date(file.lastModified).toISOString());
    } finally {
      URL.revokeObjectURL(url);
    }
    report(`Loaded ${file.name}: set its camera orientation and field of view so that the compass letters and the horizon match the photo`);
  }

  function photoSetText() {
    const photos = views.filter(view => view.kind === "photo").map(view => ({ name: view.name, taken: view.taken, view: view.view, image: view.image.toDataURL("image/jpeg", 0.85) }));
    return JSON.stringify({ format_version: PHOTO_SET_FORMAT_VERSION, kind: "photo_set", photos });
  }

  // ---- Opening and closing ----

  // (Re)start from the current obstructed sky description, e.g. after loading another one.
  async function reload() {
    const start = await call("startEditing", nSkyNodes());
    sky = { nodes: start.nodes, triangles: start.triangles };
    flags = Uint8Array.from(start.obstructed);
    skyMapPixels = sky.nodes.map(skyMapPosition);
    methodsUsed.clear();
    report(`Marking obstructions on ${start.triangles.length} sky patches; start state: ${start.method === "none" ? "free sky" : start.method}`);
    for (const view of views) if (view.kind === "photo") await project(view);
    showViews();
    draw();
  }

  async function open() {
    element("editor").hidden = false;
    await reload();
    element("editor").scrollIntoView({ behavior: "smooth" });
  }

  function close() {
    stopCamera();
    element("editor").hidden = true;
  }

  async function freeAll() {
    flags.fill(0);
    methodsUsed.add(views[current].kind);
    draw();
    await apply();
  }

  for (const key of ["azimuth_deg", "elevation_deg", "roll_deg", "fov_deg"]) element(`photo-${key}`).addEventListener("input", photoSettingChanged);
  element("photo-remove").addEventListener("click", removePhoto);

  return {
    open, close, reload, startCamera, stopCamera, takePhoto, loadPhotoFile, photoSetText, freeAll,
    get isOpen() { return !element("editor").hidden; },
    get hasPhotos() { return views.some(view => view.kind === "photo"); },
  };
}
