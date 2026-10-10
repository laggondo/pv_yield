// Map of the site in the panel tab (#38): OpenStreetMap over faint aerial imagery (Esri World Imagery), with the panel
// as a small rectangle in the map's centre. Panning the map moves the panel and with it the site (the panel always stays
// in the centre, zooming keeps it there); the handle at the end of the line from the panel shows the direction the panel
// faces and turns it (the azimuth field). Leaflet is loaded from the CDN when the map is first shown; tiles need network access.

const LEAFLET_URL = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist";
const DEFAULT_ZOOM = 19;
// Opacity of the aerial imagery over the map.
const IMAGERY_OPACITY = 0.2;
// Distance of the turning handle from the panel's centre in screen pixels.
const HANDLE_DISTANCE_PX = 75;
// Overlay size in pixels (square, centred on the map): panel, line and handle.
const OVERLAY_SIZE_PX = 2 * HANDLE_DISTANCE_PX + 70;
// Decimals of the coordinates the map reports (about 1 m), as in the site fields.
const COORDINATE_DECIMALS = 5;

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

// The panel map in the element "panel-map". Callbacks: `position()` gives the panel's position, i.e. the site
// ({latitude, longitude}, or null if there is none yet), `azimuth()` the panel's azimuth (null: optimized),
// `onAzimuth(degrees)` and `onPosition(latitude, longitude)` report the user's changes.
export function createPanelMap({ element, report, position, azimuth, onAzimuth, onPosition }) {
  const container = element("panel-map");
  const wrap = container.parentElement;
  const note = element("panel-map-note");
  let map = null;
  let leaflet = null;

  // The overlay over the map's centre: the panel (rectangle, its long side across the facing direction, the facing edge
  // drawn thicker), the line to the handle and the handle. Only the handle takes pointer events; the rest goes to the map.
  const overlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  overlay.setAttribute("class", "panel-map-overlay");
  overlay.setAttribute("width", OVERLAY_SIZE_PX);
  overlay.setAttribute("height", OVERLAY_SIZE_PX);
  overlay.setAttribute("viewBox", `${-OVERLAY_SIZE_PX / 2} ${-OVERLAY_SIZE_PX / 2} ${OVERLAY_SIZE_PX} ${OVERLAY_SIZE_PX}`);
  overlay.innerHTML = `<g class="rotating">
      <line class="direction" x1="0" y1="-13" x2="0" y2="${-HANDLE_DISTANCE_PX}"></line>
      <rect class="panel" x="-22" y="-13" width="44" height="26"></rect>
      <line class="facing-edge" x1="-22" y1="-13" x2="22" y2="-13"></line>
      <circle class="centre" r="3"></circle>
      <circle class="handle-target" cx="0" cy="${-HANDLE_DISTANCE_PX}" r="32"></circle>
      <circle class="handle" cx="0" cy="${-HANDLE_DISTANCE_PX}" r="12"></circle>
    </g>`;
  const rotating = overlay.querySelector(".rotating");
  const handleTarget = overlay.querySelector(".handle-target");
  wrap.appendChild(overlay);

  // Turn the overlay to the azimuth; an empty azimuth (optimized) is shown faded, facing south.
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
  // On phones, a finger on the handle would scroll the page after a few pixels (and cancel the turning): only cancelling
  // the touch events stops that (cancelling the pointer events doesn't, and `touch-action` isn't honoured on SVG shapes everywhere).
  for (const type of ["touchstart", "touchmove"]) handleTarget.addEventListener(type, event => event.preventDefault(), { passive: false });

  function showNote(text) {
    note.textContent = text;
    note.hidden = !text;
  }

  // The map's centre, rounded like the site fields.
  function centre() {
    const { lat, lng } = map.getCenter();
    return { latitude: Number(lat.toFixed(COORDINATE_DECIMALS)), longitude: Number(leaflet.Util.wrapNum(lng, [-180, 180], true).toFixed(COORDINATE_DECIMALS)) };
  }
  const samePosition = (a, b) => Math.abs(a.latitude - b.latitude) < 0.6 * 10 ** -COORDINATE_DECIMALS && Math.abs(a.longitude - b.longitude) < 0.6 * 10 ** -COORDINATE_DECIMALS;

  // Show the map (when the panel tab is shown, or inputs changed): created on first use, then moved to the site if that
  // changed elsewhere, and resized (a hidden map has no size).
  async function show() {
    drawAzimuth();
    const start = position();
    wrap.hidden = start === null;
    if (start === null) return showNote("The map shows up once the site is set or weather data is loaded (tab Site & weather).");
    showNote("");
    if (!map) {
      try {
        leaflet = await loadLeaflet();
      } catch (error) {
        wrap.hidden = true;
        report(`Could not load the map library (offline?): ${error.message ?? error}`);
        return showNote("The map could not be loaded (it needs network access).");
      }
      // Zooming keeps the map's centre (the panel) in place, also with the mouse wheel and two fingers.
      map = leaflet.map(container, { center: [start.latitude, start.longitude], zoom: DEFAULT_ZOOM, maxZoom: 21, scrollWheelZoom: "center", touchZoom: "center", doubleClickZoom: "center" });
      leaflet.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxNativeZoom: 19, maxZoom: 21,
        attribution: "© <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors" }).addTo(map);
      leaflet.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxNativeZoom: 19, maxZoom: 21, opacity: IMAGERY_OPACITY,
        attribution: "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community" }).addTo(map);
      leaflet.control.scale({ imperial: false }).addTo(map);
      map.on("moveend", () => {
        const current = position(), moved = centre();
        if (!current || !samePosition(current, moved)) onPosition(moved.latitude, moved.longitude);
      });
      report("Panel map loaded");
    }
    map.invalidateSize();
    if (!samePosition(centre(), start)) map.setView([start.latitude, start.longitude], map.getZoom(), { animate: false });
  }

  return { show, drawAzimuth };
}
