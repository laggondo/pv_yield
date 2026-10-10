// Map of the site in the panel tab (#38): aerial imagery (or OpenStreetMap) with the panel as a small rectangle in the
// map's centre. Panning the map moves the panel (it always stays in the centre, zooming keeps it there); the handle at
// the end of the line from the panel shows the direction the panel faces and turns it (the azimuth field). The map
// starts at the panel's stored position, else at the site; it doesn't set the site (a few 100 m don't matter for the
// sun position and the weather). Leaflet is loaded from the CDN when the map is first shown; tiles need network access.

const LEAFLET_URL = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist";
const DEFAULT_ZOOM = 19;
// Distance of the turning handle from the panel's centre in screen pixels.
const HANDLE_DISTANCE_PX = 75;
// Overlay size in pixels (square, centred on the map): panel, line and handle.
const OVERLAY_SIZE_PX = 2 * HANDLE_DISTANCE_PX + 50;

const BASE_LAYERS = {
  "Aerial imagery (Esri)": { url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community", maxNativeZoom: 19 },
  "Map (OpenStreetMap)": { url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", attribution: "© <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors", maxNativeZoom: 19 },
};

// Load Leaflet's stylesheet and module once.
let leafletPromise = null;
function loadLeaflet() {
  leafletPromise ??= (async () => {
    document.head.appendChild(Object.assign(document.createElement("link"), { rel: "stylesheet", href: `${LEAFLET_URL}/leaflet.css` }));
    return import(`${LEAFLET_URL}/leaflet-src.esm.js`);
  })();
  return leafletPromise;
}

// The compass azimuth (0 = north, clockwise) of a screen direction (y downwards), in [0, 360).
function azimuthOfScreenDirection(dx, dy) {
  return (Math.atan2(dx, -dy) * 180 / Math.PI + 360) % 360;
}

// The panel map in the given container. Callbacks: `start()` gives the position to start at ({latitude, longitude} or
// null if there is no site yet), `azimuth()` the panel's azimuth (null: optimized), `onAzimuth(degrees)` and
// `onPosition(latitude, longitude)` report the user's changes.
export function createPanelMap({ element, report, start, azimuth, onAzimuth, onPosition }) {
  const container = element("panel-map");
  const note = element("panel-map-note");
  let map = null;
  let leaflet = null;

  // The overlay over the map's centre: the panel (rectangle, its long side across the facing direction, the facing edge
  // drawn thicker), the line to the handle and the handle. Only the handle takes pointer events; the rest goes to the map.
  const svgNamespace = "http://www.w3.org/2000/svg";
  const overlay = document.createElementNS(svgNamespace, "svg");
  overlay.setAttribute("class", "panel-map-overlay");
  overlay.setAttribute("width", OVERLAY_SIZE_PX);
  overlay.setAttribute("height", OVERLAY_SIZE_PX);
  overlay.setAttribute("viewBox", `${-OVERLAY_SIZE_PX / 2} ${-OVERLAY_SIZE_PX / 2} ${OVERLAY_SIZE_PX} ${OVERLAY_SIZE_PX}`);
  overlay.innerHTML = `<g class="rotating">
      <line class="direction" x1="0" y1="0" x2="0" y2="${-HANDLE_DISTANCE_PX}"></line>
      <rect class="panel" x="-22" y="-13" width="44" height="26"></rect>
      <line class="facing-edge" x1="-22" y1="-13" x2="22" y2="-13"></line>
      <circle class="centre" r="3"></circle>
      <circle class="handle-target" cx="0" cy="${-HANDLE_DISTANCE_PX}" r="22"></circle>
      <circle class="handle" cx="0" cy="${-HANDLE_DISTANCE_PX}" r="10"></circle>
    </g>`;
  const rotating = overlay.querySelector(".rotating");
  const handleTarget = overlay.querySelector(".handle-target");
  container.parentElement.appendChild(overlay);

  // Turn the overlay to the azimuth; an empty azimuth (optimized) is shown faded, pointing south.
  function drawAzimuth() {
    const degrees = azimuth();
    rotating.setAttribute("transform", `rotate(${degrees ?? 180})`);
    overlay.classList.toggle("optimized", degrees === null);
  }

  // Dragging the handle turns the panel: the azimuth is the direction from the panel's centre to the pointer.
  function turnTo(event) {
    const box = overlay.getBoundingClientRect();
    const dx = event.clientX - (box.left + box.width / 2), dy = event.clientY - (box.top + box.height / 2);
    if (Math.hypot(dx, dy) < 5) return;
    onAzimuth(Math.round(azimuthOfScreenDirection(dx, dy)) % 360);
    drawAzimuth();
  }
  handleTarget.addEventListener("pointerdown", event => {
    event.preventDefault();
    handleTarget.setPointerCapture(event.pointerId);
    overlay.classList.add("turning");
    turnTo(event);
  });
  handleTarget.addEventListener("pointermove", event => { if (handleTarget.hasPointerCapture(event.pointerId)) turnTo(event); });
  for (const type of ["pointerup", "pointercancel"]) handleTarget.addEventListener(type, () => overlay.classList.remove("turning"));

  function showNote(text) {
    note.textContent = text;
    note.hidden = !text;
  }

  // Show the map (when the panel tab is shown, or the site changed): created on first use, then moved to the start
  // position if that changed (e.g. a new site) and resized (a hidden map has no size).
  async function show() {
    drawAzimuth();
    const position = start();
    container.parentElement.hidden = position === null;
    if (position === null) return showNote("The map shows up when the site is set (tab Site & weather) or weather data is loaded.");
    showNote("");
    if (!map) {
      try {
        leaflet = await loadLeaflet();
      } catch (error) {
        container.parentElement.hidden = true;
        report(`Could not load the map library (offline?): ${error.message ?? error}`);
        return showNote("The map could not be loaded (it needs network access); the azimuth can still be entered above.");
      }
      // Zooming keeps the map's centre (the panel) in place, also with the mouse wheel and two fingers.
      map = leaflet.map(container, { center: [position.latitude, position.longitude], zoom: DEFAULT_ZOOM, maxZoom: 21, scrollWheelZoom: "center", touchZoom: "center", doubleClickZoom: "center" });
      const layers = Object.fromEntries(Object.entries(BASE_LAYERS).map(([name, { url, ...options }]) => [name, leaflet.tileLayer(url, { maxZoom: 21, ...options })]));
      Object.values(layers)[0].addTo(map);
      leaflet.control.layers(layers, null, { collapsed: true }).addTo(map);
      leaflet.control.scale({ imperial: false }).addTo(map);
      map.on("moveend", () => {
        const centre = map.getCenter();
        const latitude = Number(centre.lat.toFixed(6)), longitude = Number(leaflet.Util.wrapNum(centre.lng, [-180, 180], true).toFixed(6));
        const current = start();
        if (current && current.latitude === latitude && current.longitude === longitude) return;
        onPosition(latitude, longitude);
      });
      report("Panel map loaded");
    }
    map.invalidateSize();
    const centre = map.getCenter();
    if (Math.abs(centre.lat - position.latitude) > 1e-6 || Math.abs(centre.lng - position.longitude) > 1e-6) map.setView([position.latitude, position.longitude], map.getZoom(), { animate: false });
  }

  return { show, drawAzimuth, get isLoaded() { return map !== null; } };
}
