// Map of the site in the panel tab (#38) with the panel as a small rectangle: dragging the rectangle places the panel,
// dragging the round handle at the end of the line from the panel turns it (the line points the way the panel faces;
// the handle sets the azimuth field and follows it). The map pans and zooms independently; a button moves the panel to
// the map's centre. The map starts at the panel's position (else at the site); it doesn't set the site.
// Map services: Google Maps (satellite and map view) if an API key is set (build setting GOOGLE_MAPS_API_KEY, see
// web/README.md, or the URL parameter ?google_maps_key=...), else Leaflet with Esri World Imagery and OpenStreetMap.
// The map library is loaded when the map is first shown; the map needs network access.

import { GOOGLE_MAPS_API_KEY } from "./site_settings.js";

const LEAFLET_URL = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist";
// Start zoom of Google Maps (Leaflet starts at 19, the finest level of Esri's imagery).
const DEFAULT_ZOOM = 20;
// Distance of the turning handle from the panel's centre in screen pixels.
const HANDLE_DISTANCE_PX = 75;

const LEAFLET_LAYERS = {
  "Aerial imagery (Esri)": { url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community", maxNativeZoom: 19 },
  "Map (OpenStreetMap)": { url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", attribution: "© <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors", maxNativeZoom: 19 },
};

// The Google Maps API key: the URL parameter, else the build setting (empty: no Google Maps).
function googleMapsKey() {
  return new URLSearchParams(location.search).get("google_maps_key") ?? GOOGLE_MAPS_API_KEY;
}

// Load a classic script once per URL.
const scriptPromises = new Map();
function loadScript(url) {
  if (!scriptPromises.has(url)) scriptPromises.set(url, new Promise((resolve, reject) => {
    const script = Object.assign(document.createElement("script"), { src: url, async: true });
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Could not load ${url.replace(/key=[^&]*/, "key=…")}`));
    document.head.appendChild(script);
  }));
  return scriptPromises.get(url);
}

// A map backend: create(container, position, onViewChange) makes the map; then toPixel(position) gives the container
// pixel of a position ({x, y}, or null before the map is ready), toPosition(x, y) the reverse, centre() the map's centre,
// setCentre(position) moves the map and resize() adapts it to a changed container size.

// Leaflet with Esri World Imagery and OpenStreetMap.
async function createLeafletMap(container, position, onViewChange) {
  document.head.appendChild(Object.assign(document.createElement("link"), { rel: "stylesheet", href: `${LEAFLET_URL}/leaflet.css` }));
  const leaflet = await import(`${LEAFLET_URL}/leaflet-src.esm.js`);
  const map = leaflet.map(container, { center: [position.latitude, position.longitude], zoom: 19, maxZoom: 21 });   // 19: Esri's finest imagery
  const layers = Object.fromEntries(Object.entries(LEAFLET_LAYERS).map(([name, { url, ...options }]) => [name, leaflet.tileLayer(url, { maxZoom: 21, ...options })]));
  Object.values(layers)[0].addTo(map);
  leaflet.control.layers(layers, null, { collapsed: true }).addTo(map);
  leaflet.control.scale({ imperial: false }).addTo(map);
  map.on("move zoom viewreset resize", onViewChange);
  const asPosition = latLng => ({ latitude: latLng.lat, longitude: leaflet.Util.wrapNum(latLng.lng, [-180, 180], true) });
  return {
    name: "Leaflet (Esri World Imagery, OpenStreetMap)",
    toPixel: ({ latitude, longitude }) => map.latLngToContainerPoint([latitude, longitude]),
    toPosition: (x, y) => asPosition(map.containerPointToLatLng([x, y])),
    centre: () => asPosition(map.getCenter()),
    setCentre: ({ latitude, longitude }) => map.setView([latitude, longitude], map.getZoom(), { animate: false }),
    resize: () => map.invalidateSize(),
    remove: () => map.remove(),
  };
}

// Google Maps, satellite view by default (straight from above), with the switch to the map view.
async function createGoogleMap(container, position, onViewChange, key, onAuthFailure) {
  // Google calls this global function if it rejects the key (e.g. not allowed for this website).
  window.gm_authFailure = onAuthFailure;
  await new Promise((resolve, reject) => {
    window.pvYieldGoogleMapsLoaded = resolve;
    loadScript(`https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}&v=weekly&loading=async&callback=pvYieldGoogleMapsLoaded`).catch(reject);
  });
  const maps = window.google.maps;
  const map = new maps.Map(container, { center: { lat: position.latitude, lng: position.longitude }, zoom: DEFAULT_ZOOM, mapTypeId: "satellite", tilt: 0, heading: 0,
    mapTypeControl: true, streetViewControl: false, rotateControl: false, fullscreenControl: false, cameraControl: false, scaleControl: true });
  // An empty overlay only for its projection (container pixels); its draw() follows zooming, the map's events panning.
  const projectionHelper = new maps.OverlayView();
  projectionHelper.onAdd = projectionHelper.onRemove = () => {};
  projectionHelper.draw = onViewChange;
  projectionHelper.setMap(map);
  for (const event of ["center_changed", "zoom_changed", "bounds_changed"]) map.addListener(event, onViewChange);
  const asPosition = latLng => ({ latitude: latLng.lat(), longitude: latLng.lng() });
  return {
    name: "Google Maps",
    toPixel: ({ latitude, longitude }) => projectionHelper.getProjection()?.fromLatLngToContainerPixel(new maps.LatLng(latitude, longitude)) ?? null,
    toPosition: (x, y) => asPosition(projectionHelper.getProjection().fromContainerPixelToLatLng(new maps.Point(x, y))),
    centre: () => asPosition(map.getCenter()),
    setCentre: ({ latitude, longitude }) => map.setCenter({ lat: latitude, lng: longitude }),
    resize: () => {},
    remove: () => projectionHelper.setMap(null),
  };
}

// The compass azimuth (0 = north, clockwise) of a screen direction (y downwards), in [0, 360).
function azimuthOfScreenDirection(dx, dy) {
  return (Math.atan2(dx, -dy) * 180 / Math.PI + 360) % 360;
}

const samePosition = (a, b) => a && b && Math.abs(a.latitude - b.latitude) < 1e-7 && Math.abs(a.longitude - b.longitude) < 1e-7;
const rounded = ({ latitude, longitude }) => ({ latitude: Number(latitude.toFixed(6)), longitude: Number(longitude.toFixed(6)) });

// The panel map in the element "panel-map". Callbacks: `position()` gives the panel's position ({latitude, longitude},
// or null if there is no site yet), `azimuth()` the panel's azimuth (null: optimized), `onAzimuth(degrees)` and
// `onPosition(latitude, longitude)` report the user's changes.
export function createPanelMap({ element, report, position, azimuth, onAzimuth, onPosition }) {
  const container = element("panel-map");
  const wrap = container.parentElement;
  const note = element("panel-map-note");
  let backend = null;
  let creating = null;
  let useGoogle = Boolean(googleMapsKey());
  // The panel's position as shown, kept while dragging (reported when the drag ends).
  let panelPosition = null;

  // The overlay over the whole map: the panel (rectangle, its long side across the facing direction, the facing edge
  // drawn thicker), the line to the handle and the handle. Only the panel and the handle take pointer events.
  const overlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  overlay.setAttribute("class", "panel-map-overlay");
  overlay.innerHTML = `<g class="rotating">
      <line class="direction" x1="0" y1="-13" x2="0" y2="${-HANDLE_DISTANCE_PX}"></line>
      <rect class="panel" x="-22" y="-13" width="44" height="26"></rect>
      <line class="facing-edge" x1="-22" y1="-13" x2="22" y2="-13"></line>
      <circle class="handle-target" cx="0" cy="${-HANDLE_DISTANCE_PX}" r="22"></circle>
      <circle class="handle" cx="0" cy="${-HANDLE_DISTANCE_PX}" r="10"></circle>
    </g>`;
  const rotating = overlay.querySelector(".rotating");
  const panelShape = overlay.querySelector(".panel");
  const handleTarget = overlay.querySelector(".handle-target");
  wrap.appendChild(overlay);

  // The panel's pixel in the map container, or null if the map isn't ready.
  function panelPixel() {
    return backend && panelPosition ? backend.toPixel(panelPosition) : null;
  }

  // Draw the panel at its position, turned to the azimuth; an empty azimuth (optimized) is shown faded, facing south.
  function draw() {
    const pixel = panelPixel();
    overlay.style.visibility = pixel ? "visible" : "hidden";
    if (!pixel) return;
    const degrees = azimuth();
    rotating.setAttribute("transform", `translate(${pixel.x} ${pixel.y}) rotate(${degrees ?? 180})`);
    overlay.classList.toggle("optimized", degrees === null);
  }
  let drawRequested = false;
  function requestDraw() {
    if (drawRequested) return;
    drawRequested = true;
    requestAnimationFrame(() => { drawRequested = false; draw(); });
  }

  // The pointer's position in the map container's pixels.
  function pointerPixel(event) {
    const box = container.getBoundingClientRect();
    return { x: event.clientX - box.left, y: event.clientY - box.top };
  }

  // Dragging with pointer capture on a shape: `move(event)` on each move, `end()` when released.
  function draggable(shape, { start = () => {}, move, end = () => {} }) {
    shape.addEventListener("pointerdown", event => {
      if (!panelPixel()) return;
      event.preventDefault();
      event.stopPropagation();
      shape.setPointerCapture(event.pointerId);
      overlay.classList.add("dragging");
      start(event);
      move(event);
    });
    shape.addEventListener("pointermove", event => { if (shape.hasPointerCapture(event.pointerId)) move(event); });
    for (const type of ["pointerup", "pointercancel"]) shape.addEventListener(type, event => {
      if (!overlay.classList.contains("dragging")) return;
      overlay.classList.remove("dragging");
      end(event);
    });
  }

  // The handle turns the panel: the azimuth is the direction from the panel's centre to the pointer.
  draggable(handleTarget, { move: event => {
    const pixel = panelPixel(), pointer = pointerPixel(event);
    const dx = pointer.x - pixel.x, dy = pointer.y - pixel.y;
    if (Math.hypot(dx, dy) < 5) return;
    onAzimuth(Math.round(azimuthOfScreenDirection(dx, dy)) % 360);
    draw();
  } });

  // The rectangle moves the panel; the point grabbed stays under the pointer.
  let grabOffset = { x: 0, y: 0 };
  draggable(panelShape, {
    start: event => {
      const pixel = panelPixel(), pointer = pointerPixel(event);
      grabOffset = { x: pointer.x - pixel.x, y: pointer.y - pixel.y };
    },
    move: event => {
      const pointer = pointerPixel(event);
      panelPosition = rounded(backend.toPosition(pointer.x - grabOffset.x, pointer.y - grabOffset.y));
      draw();
    },
    end: () => onPosition(panelPosition.latitude, panelPosition.longitude),
  });

  // Move the panel to the map's centre (after panning to the roof, quicker than dragging the panel there).
  element("panel-map-centre").addEventListener("click", () => {
    if (!backend) return;
    panelPosition = rounded(backend.centre());
    onPosition(panelPosition.latitude, panelPosition.longitude);
    draw();
  });

  function showNote(text) {
    note.textContent = text;
    note.hidden = !text;
  }

  // Google rejected the key: fall back to Leaflet with Esri imagery.
  function googleFailed(reason) {
    report(`Google Maps not available (${reason}); using Esri World Imagery and OpenStreetMap instead`);
    useGoogle = false;
    backend?.remove();
    backend = null;
    creating = null;
    container.replaceChildren();
    container.removeAttribute("class");
    container.removeAttribute("style");
    show().catch(error => report(`Could not show the map: ${error.message ?? error}`));
  }

  async function createBackend(start) {
    if (useGoogle) {
      try {
        return await createGoogleMap(container, start, requestDraw, googleMapsKey(), () => googleFailed("the API key was rejected, see the browser console"));
      } catch (error) {
        report(`Could not load Google Maps: ${error.message ?? error}`);
        useGoogle = false;
      }
    }
    return createLeafletMap(container, start, requestDraw);
  }

  // Show the map (when the panel tab is shown, or inputs changed): created on first use; a panel position changed
  // elsewhere (e.g. a new site) moves the panel and centres the map on it.
  async function show() {
    const start = position();
    wrap.hidden = start === null;
    element("panel-map-centre").hidden = start === null;
    if (start === null) {
      panelPosition = null;
      return showNote("The map shows up once the site is set or weather data is loaded (tab Site & weather).");
    }
    showNote("");
    if (!backend) {
      creating ??= createBackend(start);
      try {
        backend = await creating;
      } catch (error) {
        creating = null;
        wrap.hidden = element("panel-map-centre").hidden = true;
        report(`Could not load the map (offline?): ${error.message ?? error}`);
        return showNote("The map could not be loaded (it needs network access).");
      }
      panelPosition = start;
      report(`Panel map loaded: ${backend.name}`);
    }
    backend.resize();
    if (!samePosition(start, panelPosition)) {
      panelPosition = start;
      backend.setCentre(start);
    }
    draw();
  }

  return { show, drawAzimuth: draw };
}
