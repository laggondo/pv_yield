// Marking obstructed sky patches by hand (photo method, #17), on the obstruction tab: the sky map (a map of the sky
// hemisphere, always shown at the top of the tab) and photos taken with the phone's camera (camera orientation from the
// phone's sensors) or loaded from files, one of them shown at a time below the list of photos. The sky patches are
// drawn over both; tapping or dragging marks the patches as obstructed or frees them, as set by the mode switch (one
// setting, shown above both). On photos, the sky grid can be moved to match the photo, and the sky map can be zoomed
// (right mouse button or two fingers; see CONTROLS_TEXT). The sky map shows the photos merged onto the hemisphere, and
// the live camera view shows which patches the photos cover. All views show the sun paths of the solstices and
// equinoxes, so photos can focus on the relevant part of the sky. The current obstructed sky description (loaded,
// computed from LiDAR, or free) is the start state, and the edits are baked into the flags (#12).
// The camera geometry (sensor angles to camera orientation, projection of the sky nodes to pixels) is computed in
// Python (core/photo.py); this module only draws and handles input. Photos are assumed to be taken from the panel
// position: no offset is applied (#11).

const PHOTO_SET_FORMAT_VERSION = 1;
const MAX_PHOTO_PX = 1600;           // longer side of stored photos, to keep memory and saved files small
const SKY_MAP_PX = 800;
const SKY_MAP_RADIUS = SKY_MAP_PX / 2 - 40;   // radius of the horizon in the sky map; the compass letters sit outside it
const OBSTRUCTED_FILL = "rgba(220, 30, 30, 0.45)";
const COVERED_FILL = "rgba(60, 150, 255, 0.4)";
const SKY_MAP_BACKGROUND = [91, 127, 166];
const COMPASS = [["N", 0], ["E", 90], ["S", 180], ["W", 270]];
const DEFAULT_PHOTO_VIEW = { azimuth_deg: 180, elevation_deg: 15, roll_deg: 0 };   // loaded photos and photos without sensors
const SHUTTER_KEYS = ["AudioVolumeUp", "AudioVolumeDown", "VolumeUp", "VolumeDown", "Camera", "Enter", " "];
const CONTROLS_TEXT = {
  sky_map: "Drag to mark several patches. Zoom with the mouse wheel or by pinching; move the zoomed map by dragging with the right mouse button or two fingers.",
  photo: "Tap or click sky patches to mark them as obstructed (red) or free them, as set by the switch; drag for several. "
    + "Align the sky grid with the photo: drag with the right mouse button (Shift: rotate) and use the mouse wheel for the field of view; "
    + "on touch screens, drag with two fingers, twist to rotate and pinch for the field of view.",
};
const MAX_SKY_MAP_ZOOM = 10;
const SKY_MAP_REDRAW_DELAY_MS = 250;  // the merged photos are recomputed this long after the last change of a photo's view

// Create the editor on the page's elements; `call` runs a worker action, `report` logs, `onApplied(summary)` is called
// after the marked flags were sent to Python as the new obstructed sky description, `onPhotosChanged()` a moment after
// photos were added, removed or aligned (to store them for the next visit). Marking starts with `reload()`.
export function createSkyEditor({ element, call, report, fail, defaultFov, nSkyNodes, site, onApplied, onPhotosChanged }) {
  let sky = null;                      // {nodes, triangles}; null until the first reload
  let flags = null;                    // Uint8Array, one entry per patch
  let skyMapPixels = null;             // node positions in the sky map
  let sunPaths = [];                   // [{label, color, directions}] for the site; empty if the site is unknown
  const skyMapView = { kind: "sky_map", name: "Sky map" };
  const photos = [];                   // [{kind: "photo", name, taken, image, view, projection}]
  let selected = null;                 // the photo shown for marking, or null
  const methodsUsed = new Set();
  let skyMapPhotos = null;             // canvas with the photos merged onto the sky map; null: to be recomputed
  const skyMapZoom = { factor: 1, x: 0, y: 0 };   // sky map shown at factor × size, shifted by (x, y) canvas pixels
  let markValue = 1;                   // what tapping sets: 1 marks as obstructed, 0 frees
  let photosChangedTimer = null;
  let skyMapRedrawTimer = null;
  const camera = { stream: null, orientation: null, listener: null, eventName: null, projection: null, running: false, requestInFlight: false, photosTaken: 0, shooting: false };

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

  // ---- Coverage and the photos merged onto the sky map ----

  // Flag per patch: 1 if a photo shows it completely (all corners in front of the camera and inside the image).
  function coveredPatches() {
    const covered = new Uint8Array(sky.triangles.length);
    for (const photo of photos) {
      if (!photo.projection) continue;
      const { pixels, in_front } = photo.projection, { width, height } = photo.view;
      const inside = node => in_front[node] && pixels[node][0] >= 0 && pixels[node][0] <= width && pixels[node][1] >= 0 && pixels[node][1] <= height;
      sky.triangles.forEach((triangle, index) => { if (triangle.every(inside)) covered[index] = 1; });
    }
    return covered;
  }

  // Share of the hemisphere's solid angle covered by photos, as text (patches weighted by their solid angle).
  function percentCovered(covered) {
    let coveredAngle = 0, totalAngle = 0;
    sky.triangles.forEach((triangle, index) => {
      const [a, b, c] = triangle.map(node => sky.nodes[node]);
      const numerator = Math.abs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]));
      const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
      const solidAngle = 2 * Math.atan2(numerator, 1 + dot(a, b) + dot(b, c) + dot(a, c));
      totalAngle += solidAngle;
      if (covered[index]) coveredAngle += solidAngle;
    });
    return `${Math.round(100 * coveredAngle / totalAngle)} %`;
  }

  // The photos mapped onto the sky map: each pixel shows the direction it stands for, taken from the photo that sees
  // this direction closest to its image centre (least distorted; seams lie halfway between photo centres). Directions
  // outside all photos keep the background colour. The camera axes come from Python (projectSky).
  function mergedSkyMapCanvas() {
    const image = new ImageData(SKY_MAP_PX, SKY_MAP_PX);
    const sources = photos.filter(photo => photo.projection).map(photo => {
      photo.pixelData ??= photo.image.getContext("2d").getImageData(0, 0, photo.image.width, photo.image.height).data;
      return { ...photo.projection.camera, width: photo.image.width, height: photo.image.height, data: photo.pixelData };
    });
    const centre = SKY_MAP_PX / 2;
    for (let y = 0; y < SKY_MAP_PX; y++) {
      for (let x = 0; x < SKY_MAP_PX; x++) {
        const offset = 4 * (y * SKY_MAP_PX + x);
        image.data.set([...SKY_MAP_BACKGROUND, 255], offset);
        const radius = Math.hypot(x - centre, y - centre);
        if (radius > SKY_MAP_RADIUS) continue;
        const zenith = radius / SKY_MAP_RADIUS * Math.PI / 2, azimuth = Math.atan2(x - centre, centre - y);
        const direction = [Math.sin(zenith) * Math.sin(azimuth), Math.sin(zenith) * Math.cos(azimuth), Math.cos(zenith)];
        let best = null, bestScore = 0;
        for (const photo of sources) {
          const depth = direction[0] * photo.forward[0] + direction[1] * photo.forward[1] + direction[2] * photo.forward[2];
          if (depth <= 0) continue;
          const u = photo.width / 2 + photo.focal_length_px * (direction[0] * photo.right[0] + direction[1] * photo.right[1] + direction[2] * photo.right[2]) / depth;
          const v = photo.height / 2 - photo.focal_length_px * (direction[0] * photo.up[0] + direction[1] * photo.up[1] + direction[2] * photo.up[2]) / depth;
          // Distance to the nearest image edge relative to the image size: > 0 inside, largest at the centre.
          const score = Math.min(u, photo.width - u, v, photo.height - v) / Math.max(photo.width, photo.height);
          if (score > bestScore) [best, bestScore] = [{ photo, u, v }, score];
        }
        if (best) {
          const source = 4 * (Math.floor(best.v) * best.photo.width + Math.floor(best.u));
          image.data.set(best.photo.data.subarray(source, source + 4), offset);
        }
      }
    }
    const merged = document.createElement("canvas");
    merged.width = merged.height = SKY_MAP_PX;
    merged.getContext("2d").putImageData(image, 0, 0);
    return merged;
  }

  // ---- Zooming the sky map ----

  // Keep the zoomed map covering the canvas.
  function clampSkyMapZoom() {
    skyMapZoom.factor = Math.max(1, Math.min(MAX_SKY_MAP_ZOOM, skyMapZoom.factor));
    const smallest = SKY_MAP_PX * (1 - skyMapZoom.factor);
    skyMapZoom.x = Math.max(smallest, Math.min(0, skyMapZoom.x));
    skyMapZoom.y = Math.max(smallest, Math.min(0, skyMapZoom.y));
  }

  // Zoom by a factor about a point (canvas pixels), which stays in place.
  function zoomSkyMap(factor, [x, y]) {
    const mapX = (x - skyMapZoom.x) / skyMapZoom.factor, mapY = (y - skyMapZoom.y) / skyMapZoom.factor;
    skyMapZoom.factor *= factor;
    clampSkyMapZoom();
    skyMapZoom.x = x - mapX * skyMapZoom.factor;
    skyMapZoom.y = y - mapY * skyMapZoom.factor;
    clampSkyMapZoom();
  }

  function panSkyMap(dx, dy) {
    skyMapZoom.x += dx;
    skyMapZoom.y += dy;
    clampSkyMapZoom();
  }

  // ---- Drawing ----

  // Canvas pixels per CSS pixel as displayed, so lines and letters keep their size on screen whatever the image size.
  function displayScale(target) {
    const displayedWidth = target.getBoundingClientRect().width;
    return displayedWidth > 0 ? target.width / displayedWidth : 1;
  }

  // Draw the patches (all outlined; those with a flag in `filled` filled, by default the obstructed ones in red), the
  // horizon and the compass markers onto a 2D context; `scale` is the canvas's display scale.
  function drawPatches(target, corners, scale, { markers = [], horizon = [], sunPaths = [], filled = flags, fillStyle = OBSTRUCTED_FILL } = {}) {
    target.lineWidth = scale;
    target.strokeStyle = "rgba(255, 255, 255, 0.75)";
    target.fillStyle = fillStyle;
    corners.forEach((triangle, index) => {
      if (!triangle) return;
      target.beginPath();
      target.moveTo(...triangle[0]);
      target.lineTo(...triangle[1]);
      target.lineTo(...triangle[2]);
      target.closePath();
      if (filled[index]) target.fill();
      target.stroke();
    });
    const drawLine = piece => {
      target.beginPath();
      piece.forEach(([x, y], index) => index ? target.lineTo(x, y) : target.moveTo(x, y));
      target.stroke();
    };
    target.lineWidth = 3 * scale;
    target.strokeStyle = "yellow";
    horizon.forEach(drawLine);
    // Sun paths in their colours on a dark outline, visible on sky and photos alike.
    for (const { color, pieces } of sunPaths) {
      target.lineWidth = 5 * scale;
      target.strokeStyle = "rgba(0, 0, 0, 0.6)";
      pieces.forEach(drawLine);
      target.lineWidth = 3 * scale;
      target.strokeStyle = color;
      pieces.forEach(drawLine);
    }
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

  function drawSkyMap() {
    if (!sky) return;
    const canvas = element("sky-map-canvas"), context = canvas.getContext("2d");
    canvas.width = canvas.height = SKY_MAP_PX;
    context.fillStyle = `rgb(${SKY_MAP_BACKGROUND.join(", ")})`;
    context.fillRect(0, 0, SKY_MAP_PX, SKY_MAP_PX);
    context.setTransform(skyMapZoom.factor, 0, 0, skyMapZoom.factor, skyMapZoom.x, skyMapZoom.y);
    if (element("sky-map-photos").checked && photos.some(photo => photo.projection)) {
      skyMapPhotos ??= mergedSkyMapCanvas();
      context.drawImage(skyMapPhotos, 0, 0);
    }
    const markers = COMPASS.map(([label, azimuth]) => {
      const radius = SKY_MAP_RADIUS + 20;
      return { label, x: SKY_MAP_PX / 2 + radius * Math.sin(azimuth * Math.PI / 180), y: SKY_MAP_PX / 2 - radius * Math.cos(azimuth * Math.PI / 180) };
    });
    const skyMapSunPaths = sunPaths.map(({ color, directions }) => ({ color, pieces: [directions.map(skyMapPosition)] }));
    drawPatches(context, patchCorners(skyMapView), displayScale(canvas) / skyMapZoom.factor, { markers, sunPaths: skyMapSunPaths });
    context.setTransform(1, 0, 0, 1, 0, 0);
  }

  function drawPhoto() {
    if (!sky || !selected) return;
    const canvas = element("photo-canvas"), context = canvas.getContext("2d");
    canvas.width = selected.image.width;
    canvas.height = selected.image.height;
    context.drawImage(selected.image, 0, 0);
    drawPatches(context, patchCorners(selected), displayScale(canvas), { markers: selected.projection?.markers, horizon: selected.projection?.horizon, sunPaths: selected.projection?.sun_paths });
  }

  function draw() {
    drawSkyMap();
    drawPhoto();
  }

  // Redraw the sky map a moment after the last change of a photo's view: merging the photos takes a while.
  function scheduleSkyMapRedraw() {
    clearTimeout(skyMapRedrawTimer);
    skyMapRedrawTimer = setTimeout(drawSkyMap, SKY_MAP_REDRAW_DELAY_MS);
  }

  // ---- Photo list and photo settings ----

  // The photos as buttons, each with a trash button to remove it; the selected one is shown below the list.
  function showPhotoList() {
    const list = element("photo-list");
    list.innerHTML = "";
    for (const photo of photos) {
      const item = Object.assign(document.createElement("span"), { className: "photo-item" });
      const button = Object.assign(document.createElement("button"), { textContent: photo.name, className: photo === selected ? "small selected" : "small" });
      button.addEventListener("click", () => select(photo === selected ? null : photo));
      const remove = Object.assign(document.createElement("button"), { textContent: "🗑", className: "small remove", title: `Remove ${photo.name}` });
      remove.setAttribute("aria-label", `Remove ${photo.name}`);
      remove.addEventListener("click", () => removePhoto(photo));
      item.append(button, remove);
      list.appendChild(item);
    }
    element("photo-view").hidden = !selected;
    showPhotoSettings();
  }

  function showPhotoSettings() {
    if (!selected) return;
    for (const key of ["azimuth_deg", "elevation_deg", "roll_deg", "fov_deg"]) element(`photo-${key}`).value = Math.round(selected.view[key] * 10) / 10;
  }

  // Show a photo for marking (null: none).
  function select(photo) {
    selected = photo;
    showPhotoList();
    drawPhoto();
  }

  // Report changed photos once they stop changing for a second (aligning changes them continuously).
  function photosChanged() {
    clearTimeout(photosChangedTimer);
    photosChangedTimer = setTimeout(onPhotosChanged, 1000);
  }

  // Project the sky nodes into a photo (Python); repeated requests while one runs are merged into one. Called for every
  // new or changed camera view, so it also reports changed photos. Before marking started, photos are projected by `reload`.
  async function project(photo) {
    photosChanged();
    if (!sky) return;
    photo.projectionWanted = true;
    if (photo.projecting) return;
    photo.projecting = true;
    try {
      while (photo.projectionWanted) {
        photo.projectionWanted = false;
        photo.projection = await call("projectSky", photo.view);
        skyMapPhotos = null;
      }
    } finally {
      photo.projecting = false;
    }
    if (photo === selected) drawPhoto();
    scheduleSkyMapRedraw();
  }

  function photoSettingChanged() {
    if (!selected) return;
    for (const key of ["azimuth_deg", "elevation_deg", "roll_deg", "fov_deg"]) {
      const value = Number(element(`photo-${key}`).value);
      if (element(`photo-${key}`).value.trim() !== "" && !Number.isNaN(value)) selected.view[key] = value;
    }
    project(selected).catch(fail);
  }

  // A photo: the image as canvas (at most MAX_PHOTO_PX) with its camera view; projected when marking starts.
  function makePhoto(image, view, name, taken) {
    const scale = Math.min(1, MAX_PHOTO_PX / Math.max(image.width, image.height));
    const photoCanvas = document.createElement("canvas");
    photoCanvas.width = Math.round(image.width * scale);
    photoCanvas.height = Math.round(image.height * scale);
    photoCanvas.getContext("2d").drawImage(image, 0, 0, photoCanvas.width, photoCanvas.height);
    return { kind: "photo", name, taken, image: photoCanvas, view: { ...view, width: photoCanvas.width, height: photoCanvas.height } };
  }

  // Add a photo with its camera view; shown for marking unless `selectPhoto` is false (photos taken in a series).
  async function addPhoto(image, view, name, taken, selectPhoto = true) {
    const photo = makePhoto(image, view, name, taken);
    photos.push(photo);
    if (selectPhoto) select(photo);
    else showPhotoList();
    await project(photo);
    report(`Photo ${name}: azimuth ${photo.view.azimuth_deg.toFixed(1)}°, elevation ${photo.view.elevation_deg.toFixed(1)}°, roll ${photo.view.roll_deg.toFixed(1)}°, field of view ${photo.view.fov_deg}°`);
  }

  function removePhoto(photo) {
    photos.splice(photos.indexOf(photo), 1);
    if (selected === photo) selected = null;
    skyMapPhotos = null;
    showPhotoList();
    drawSkyMap();
    photosChanged();
    report(`Removed ${photo.name}`);
  }

  // Send the flags to Python as the new obstructed sky description, with the contributing methods and the photos' views.
  async function apply() {
    const details = photos.map(photo => ({ name: photo.name, taken: photo.taken, ...photo.view }));
    const summary = await call("applyEdits", Array.from(flags), [...methodsUsed], { photos: details });
    await onApplied(summary);
  }

  // ---- Aligning a photo: moving the sky grid ----

  // Change the photo's camera view by a movement of the grid on screen (canvas pixels): the grid follows the pointer,
  // so the camera turns the other way. The movement is split along the level camera axes, which the roll turns.
  function moveGrid(photo, dx, dy) {
    const focalLength = Math.max(photo.view.width, photo.view.height) / 2 / Math.tan(photo.view.fov_deg * Math.PI / 360);
    const roll = photo.view.roll_deg * Math.PI / 180;
    const along = dx * Math.cos(roll) + dy * Math.sin(roll);
    const upwards = dx * Math.sin(roll) - dy * Math.cos(roll);
    const elevation = photo.view.elevation_deg * Math.PI / 180;
    photo.view.azimuth_deg = ((photo.view.azimuth_deg - along / focalLength * 180 / Math.PI / Math.max(Math.cos(elevation), 0.3)) % 360 + 360) % 360;
    photo.view.elevation_deg = Math.max(-90, Math.min(90, photo.view.elevation_deg - upwards / focalLength * 180 / Math.PI));
  }

  // Zoom the grid by a factor (> 1: larger, i.e. a smaller field of view).
  function zoomGrid(photo, factor) {
    const fov = 2 * Math.atan(Math.tan(photo.view.fov_deg * Math.PI / 360) / factor) * 180 / Math.PI;
    photo.view.fov_deg = Math.max(10, Math.min(170, fov));
  }

  function gridChanged(photo) {
    showPhotoSettings();
    project(photo).catch(fail);
  }

  // ---- Pointer input on a canvas: marking (left button, one finger); aligning a photo or zooming the sky map (right button, two fingers) ----

  // `currentView()` gives the view shown on the canvas: the sky map or the selected photo.
  function attachPointerInput(canvas, currentView) {
    const pointers = new Map();          // active pointers (touch: several fingers) by id: {x, y} in canvas pixels
    let stroke = null;                   // {changed, before} while marking
    let alignment = null;                // {touch, last, rotate} while aligning a photo or moving the sky map

    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
    }

    // The point in the view's own coordinates: canvas pixels, undoing the sky map's zoom.
    function viewPoint(event) {
      const [x, y] = canvasPoint(event);
      if (currentView().kind !== "sky_map") return [x, y];
      return [(x - skyMapZoom.x) / skyMapZoom.factor, (y - skyMapZoom.y) / skyMapZoom.factor];
    }

    function markAt(event) {
      const index = patchAt(currentView(), viewPoint(event));
      if (index < 0) return;
      if (flags[index] !== markValue) {
        flags[index] = markValue;
        stroke.changed = true;
        draw();
      }
    }

    // Centre, spread and angle of the active touch points (two fingers), to move, zoom and rotate the grid.
    function touchGesture() {
      const [first, second] = [...pointers.values()];
      return { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2, distance: Math.hypot(second.x - first.x, second.y - first.y), angle: Math.atan2(second.y - first.y, second.x - first.x) };
    }

    function startAlignment(event) {
      alignment = event.pointerType === "touch" ? { touch: true, last: touchGesture() } : { touch: false, last: canvasPoint(event), rotate: event.shiftKey };
    }

    function align(event) {
      const view = currentView();
      if (view.kind === "sky_map") return moveSkyMap(event);
      if (alignment.touch) {
        if (pointers.size < 2) return;
        const gesture = touchGesture(), last = alignment.last;
        moveGrid(view, gesture.x - last.x, gesture.y - last.y);
        if (last.distance > 0) zoomGrid(view, gesture.distance / last.distance);
        view.view.roll_deg += (gesture.angle - last.angle) * 180 / Math.PI;
        alignment.last = gesture;
      } else {
        const [x, y] = canvasPoint(event), [lastX, lastY] = alignment.last;
        if (alignment.rotate) view.view.roll_deg += (x - lastX) * 0.2 / displayScale(canvas);
        else moveGrid(view, x - lastX, y - lastY);
        alignment.last = [x, y];
      }
      gridChanged(view);
    }

    // Zoom and move the sky map with two fingers, or move it with the right mouse button.
    function moveSkyMap(event) {
      if (alignment.touch) {
        if (pointers.size < 2) return;
        const gesture = touchGesture(), last = alignment.last;
        panSkyMap(gesture.x - last.x, gesture.y - last.y);
        if (last.distance > 0) zoomSkyMap(gesture.distance / last.distance, [gesture.x, gesture.y]);
        alignment.last = gesture;
      } else {
        const [x, y] = canvasPoint(event), [lastX, lastY] = alignment.last;
        panSkyMap(x - lastX, y - lastY);
        alignment.last = [x, y];
      }
      drawSkyMap();
    }

    canvas.addEventListener("contextmenu", event => event.preventDefault());
    canvas.addEventListener("pointerdown", event => {
      if (!sky) return;
      canvas.setPointerCapture(event.pointerId);
      const [x, y] = canvasPoint(event);
      pointers.set(event.pointerId, { x, y });
      if (event.pointerType === "touch" && pointers.size === 2) {
        // A second finger: undo what the first one marked and align or zoom instead.
        if (stroke) flags.set(stroke.before);
        stroke = null;
        startAlignment(event);
        draw();
      } else if (event.button === 2) {
        startAlignment(event);
      } else if (event.button === 0 && pointers.size === 1) {
        stroke = { changed: false, before: flags.slice() };
        markAt(event);
      }
    });
    canvas.addEventListener("pointermove", event => {
      if (!pointers.has(event.pointerId)) return;
      const [x, y] = canvasPoint(event);
      pointers.set(event.pointerId, { x, y });
      if (alignment) align(event);
      else if (stroke) markAt(event);
    });
    const endPointer = event => {
      pointers.delete(event.pointerId);
      if (alignment) {
        if (!alignment.touch || pointers.size === 0) alignment = null;
        return;
      }
      if (!stroke) return;
      const changed = stroke.changed;
      stroke = null;
      if (!changed) return;
      methodsUsed.add(currentView().kind);
      apply().catch(fail);
    };
    canvas.addEventListener("pointerup", endPointer);
    canvas.addEventListener("pointercancel", endPointer);
    canvas.addEventListener("wheel", event => {
      if (!sky) return;
      const view = currentView();
      event.preventDefault();
      if (view.kind === "sky_map") {
        zoomSkyMap(Math.exp(-event.deltaY * 0.001), canvasPoint(event));
        drawSkyMap();
      } else {
        zoomGrid(view, Math.exp(-event.deltaY * 0.001));
        gridChanged(view);
      }
    }, { passive: false });
  }

  attachPointerInput(element("sky-map-canvas"), () => skyMapView);
  attachPointerInput(element("photo-canvas"), () => selected);

  // ---- Mode switch: mark or free (one setting, a switch above the sky map and one above the photo) ----

  function setMarkValue(value) {
    markValue = value;
    for (const button of document.querySelectorAll("[data-mark-value]")) {
      button.classList.toggle("selected", Number(button.dataset.markValue) === value);
      button.setAttribute("aria-pressed", String(Number(button.dataset.markValue) === value));
    }
  }
  for (const button of document.querySelectorAll("[data-mark-value]")) button.addEventListener("click", () => setMarkValue(Number(button.dataset.markValue)));

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
      // While aiming, the patches covered by earlier photos matter more than the obstructions marked so far.
      const covered = coveredPatches();
      drawPatches(overlayContext, patchCorners({ kind: "photo", projection: camera.projection }), displayScale(overlay), { markers: camera.projection.markers, horizon: camera.projection.horizon, sunPaths: camera.projection.sun_paths, filled: covered, fillStyle: COVERED_FILL });
      const { azimuth_deg, elevation_deg, roll_deg } = camera.projection.view;
      element("camera-status").textContent = `${camera.photosTaken} photo${camera.photosTaken === 1 ? "" : "s"} taken; camera: azimuth ${azimuth_deg.toFixed(0)}°, elevation ${elevation_deg.toFixed(0)}°, roll ${roll_deg.toFixed(0)}°; blue: covered by earlier photos (${percentCovered(covered)} of the sky)`
        + (camera.orientation.absolute ? "" : "; no compass: correct the azimuth after taking the photo");
    }
    requestAnimationFrame(liveOverlay);
  }

  // Hardware keys (volume, camera key, Enter, space) take the photo, where the browser passes them to the page.
  function onCameraKey(event) {
    if (!SHUTTER_KEYS.includes(event.key)) return;
    event.preventDefault();
    if (!event.repeat) element("camera-shoot").click();
  }

  async function startCamera() {
    try {
      await openCamera();
    } catch (error) {
      stopCamera();       // leave the full-screen layer, e.g. when the camera permission is denied
      throw error;
    }
  }

  async function openCamera() {
    element("camera").hidden = false;
    // Full screen hides the browser's bars; not offered everywhere (e.g. iPhones), where the fixed layer covers the page.
    if (element("camera").requestFullscreen) element("camera").requestFullscreen().catch(error => report(`No full screen: ${error.message}`));
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
    window.addEventListener("keydown", onCameraKey);
    camera.photosTaken = 0;
    camera.running = true;
    element("camera-status").textContent = "Waiting for the motion sensors ... (without them, the photo gets a default orientation to align afterwards)";
    report(`Camera started (${video.videoWidth} x ${video.videoHeight} pixels, orientation from ${camera.eventName})`);
    requestAnimationFrame(liveOverlay);
  }

  function stopCamera() {
    camera.running = false;
    camera.stream?.getTracks().forEach(track => track.stop());
    camera.stream = null;
    if (camera.listener) window.removeEventListener(camera.eventName, camera.listener);
    camera.listener = null;
    window.removeEventListener("keydown", onCameraKey);
    if (document.fullscreenElement) document.exitFullscreen().catch(error => report(`Could not leave full screen: ${error.message}`));
    element("camera").hidden = true;
    // After a series of photos, the sky map shows them all merged.
    if (camera.photosTaken > 0) {
      select(null);
      element("sky-map-canvas").scrollIntoView({ block: "center" });
    }
    camera.photosTaken = 0;
  }

  // A short white flash over the camera image as feedback that a photo was taken.
  function flash() {
    const layer = element("camera-flash");
    layer.classList.remove("flashing");
    void layer.offsetWidth;     // restart the animation
    layer.classList.add("flashing");
  }

  // Take a photo and stay in the camera: the live view then marks the patches the new photo covers, so the next photo
  // can aim at the rest of the sky. "Done" leaves the camera.
  async function takePhoto() {
    if (camera.shooting || !camera.running) return;
    camera.shooting = true;
    try {
      await shoot();
    } finally {
      camera.shooting = false;
    }
  }

  async function shoot() {
    const video = element("camera-video");
    const frame = document.createElement("canvas");
    frame.width = video.videoWidth;
    frame.height = video.videoHeight;
    frame.getContext("2d").drawImage(video, 0, 0);
    // The orientation at the moment of the shot; without sensors, a default view to correct by hand.
    const view = camera.orientation ? (await call("projectSky", liveView())).view : { ...DEFAULT_PHOTO_VIEW, fov_deg: defaultFov() };
    const taken = new Date().toISOString();
    flash();
    camera.photosTaken += 1;
    await addPhoto(frame, view, `Photo ${photos.length + 1}`, taken, false);
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
      await addPhoto(await loadImage(url), { ...DEFAULT_PHOTO_VIEW, fov_deg: defaultFov() }, file.name, new Date(file.lastModified).toISOString());
    } finally {
      URL.revokeObjectURL(url);
    }
    report(`Loaded ${file.name}: move the sky grid (or set the camera orientation) so that the horizon and the compass letters match the photo`);
  }

  function photoSetText() {
    const entries = photos.map(photo => ({ name: photo.name, taken: photo.taken, view: photo.view, image: photo.image.toDataURL("image/jpeg", 0.85) }));
    return JSON.stringify({ format_version: PHOTO_SET_FORMAT_VERSION, kind: "photo_set", photos: entries });
  }

  // The photos as JPEG bytes with their camera views, for a project (#37).
  async function photoFiles() {
    const toJpeg = canvas => new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Could not encode a photo as JPEG")), "image/jpeg", 0.85));
    return Promise.all(photos.map(async photo => ({ name: photo.name, taken: photo.taken, view: photo.view, bytes: new Uint8Array(await (await toJpeg(photo.image)).arrayBuffer()) })));
  }

  // Replace all photos, e.g. by those of a loaded project ([{name, taken, view, bytes}] with JPEG bytes).
  async function setPhotos(newPhotos) {
    const loaded = [];
    for (const { name, taken, view, bytes } of newPhotos) {
      const url = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
      try {
        loaded.push(makePhoto(await loadImage(url), view, name, taken));
      } finally {
        URL.revokeObjectURL(url);
      }
    }
    photos.splice(0, photos.length, ...loaded);
    selected = null;
    skyMapPhotos = null;
    showPhotoList();
    for (const photo of loaded) await project(photo);
    photosChanged();
    drawSkyMap();
  }

  // ---- Start state ----

  // (Re)start marking from the current obstructed sky description (or a free sky), e.g. after loading another one or
  // a change of the site (sun paths).
  async function reload() {
    const { latitude, longitude } = site();
    const start = await call("startEditing", nSkyNodes(), latitude, longitude);
    sky = { nodes: start.nodes, triangles: start.triangles };
    sunPaths = start.sun_paths;
    element("sun-path-legend").innerHTML = sunPaths.length
      ? "Sun paths: " + sunPaths.map(({ label, color }) => `<span style="color: ${color}">━ ${label}</span>`).join(", ")
      : "Sun paths: set the site or load weather data to show them.";
    flags = Uint8Array.from(start.obstructed);
    skyMapPixels = sky.nodes.map(skyMapPosition);
    methodsUsed.clear();
    report(`Marking obstructions on ${start.triangles.length} sky patches; start state: ${start.method === "none" || !start.method ? "free sky" : start.method}`);
    for (const photo of photos) await project(photo);
    skyMapPhotos = null;
    draw();
  }

  element("sky-map-controls").textContent = CONTROLS_TEXT.sky_map;
  element("photo-controls").textContent = CONTROLS_TEXT.photo;
  for (const key of ["azimuth_deg", "elevation_deg", "roll_deg", "fov_deg"]) element(`photo-${key}`).addEventListener("input", photoSettingChanged);
  element("sky-map-photos").addEventListener("change", drawSkyMap);

  return {
    reload, redraw: draw, startCamera, stopCamera, takePhoto, loadPhotoFile, photoSetText, photoFiles, setPhotos,
    get isStarted() { return sky !== null; },
    get hasPhotos() { return photos.length > 0; },
  };
}
