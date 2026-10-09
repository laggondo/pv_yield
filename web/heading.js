// Combining the phone's two orientation readings for the camera (Chrome on Android): the gyroscope-based one
// (`deviceorientation`: steady when turning, but its heading starts in an arbitrary direction and drifts slowly) and the
// compass-based one (`deviceorientationabsolute`: relative to north, but jumpy and pulled off by uncalibrated compasses
// or metal nearby). Both describe the same device rotation up to a turn about the vertical, the gyroscope's heading
// offset; the camera uses the gyroscope's orientation turned by an estimate of that offset. Before the first photo, the
// estimate is the compass's mean over the last seconds (the shutter waits until it is steady); the first photo fixes it
// as the reference, and from then on the compass only corrects it slowly, so it pulls the view along neither when the
// phone turns nor when the compass jumps. A lasting difference between compass and estimate suggests calibrating the
// compass (figure-8 movement).

const SETTLE_SECONDS = 2;            // the offset must be steady over this long before the first photo
const SETTLE_SPREAD_DEG = 4;         // largest deviation from the mean offset that counts as steady
const CORRECTION_SECONDS = 30;       // time constant of the compass correction after the first photo
const WARNING_DEG = 10;              // compass and estimate differing more than this: calibrate the compass
const PAIR_SECONDS = 0.2;            // a compass reading is paired with a gyroscope reading at most this old

const wrapDeg = angle => ((angle + 180) % 360 + 360) % 360 - 180;
const radians = degrees => degrees * Math.PI / 180;

// Rotation matrix from device to earth coordinates for W3C device orientation angles (Z-X'-Y'' intrinsic), as
// `rotation_from_device_angles` in core/photo.py.
export function rotationFromDeviceAngles({ alpha_deg, beta_deg, gamma_deg }) {
  const [ca, sa, cb, sb, cg, sg] = [alpha_deg, beta_deg, gamma_deg].flatMap(angle => [Math.cos(radians(angle)), Math.sin(radians(angle))]);
  return [
    [ca * cg - sa * sb * sg, -sa * cb, ca * sg + sa * sb * cg],
    [sa * cg + ca * sb * sg, ca * cb, sa * sg - ca * sb * cg],
    [-cb * sg, sb, cb * cg],
  ];
}

// The turn about the vertical (degrees, counterclockwise seen from above) from the gyroscope's orientation to the
// compass's: the yaw of compass × gyroscopeᵀ, robust also where the Euler angles of the two readings differ (phone upright).
function headingOffset(compass, gyroscope) {
  const a = rotationFromDeviceAngles(compass), b = rotationFromDeviceAngles(gyroscope);
  const m = (i, j) => a[i][0] * b[j][0] + a[i][1] * b[j][1] + a[i][2] * b[j][2];
  return Math.atan2(m(1, 0) - m(0, 1), m(0, 0) + m(1, 1)) * 180 / Math.PI;
}

// Mean of angles (degrees) and the largest deviation from it.
function circularMean(angles) {
  const mean = Math.atan2(angles.reduce((sum, angle) => sum + Math.sin(radians(angle)), 0), angles.reduce((sum, angle) => sum + Math.cos(radians(angle)), 0)) * 180 / Math.PI;
  return { mean, spread: Math.max(...angles.map(angle => Math.abs(wrapDeg(angle - mean)))) };
}

export class HeadingFusion {
  // Times are in seconds (e.g. event.timeStamp / 1000).
  constructor() {
    this.gyroscope = null;           // latest gyroscope reading {alpha_deg, beta_deg, gamma_deg, time}
    this.compass = null;             // latest compass reading
    this.offsets = [];               // [{time, offset}] measured over the last SETTLE_SECONDS
    this.estimate = null;            // estimated heading offset (degrees); null until steady once
    this.locked = false;             // true from the first photo on: the compass only corrects slowly
    this.lastCorrection = null;
  }

  addGyroscope(angles, time) {
    this.gyroscope = { ...angles, time };
  }

  addCompass(angles, time) {
    this.compass = { ...angles, time };
    if (!this.gyroscope || time - this.gyroscope.time > PAIR_SECONDS) return;
    const offset = headingOffset(angles, this.gyroscope);
    this.offsets.push({ time, offset });
    while (this.offsets.length && this.offsets[0].time < time - SETTLE_SECONDS) this.offsets.shift();
    if (!this.locked) {
      const { mean, spread } = circularMean(this.offsets.map(entry => entry.offset));
      if (spread <= SETTLE_SPREAD_DEG && this.covers(SETTLE_SECONDS)) this.estimate = mean;
    } else {
      const step = Math.min(1, (time - this.lastCorrection) / CORRECTION_SECONDS);
      this.estimate = wrapDeg(this.estimate + wrapDeg(offset - this.estimate) * step);
      this.lastCorrection = time;
    }
  }

  // True if the measured offsets span at least most of `seconds`.
  covers(seconds) {
    return this.offsets.length >= 5 && this.offsets.at(-1).time - this.offsets[0].time >= 0.8 * seconds;
  }

  // Both readings are coming in (within the last second before `time`).
  active(time) {
    return Boolean(this.gyroscope && this.compass && time - this.gyroscope.time < 1 && time - this.compass.time < 1);
  }

  // Ready for a photo: the offset is known (steady before the first photo).
  get ready() {
    return this.estimate !== null;
  }

  // The current offset measurements: their spread, and how far the compass is from the estimate (degrees).
  get status() {
    if (!this.offsets.length) return { spread: null, difference: null };
    const { mean, spread } = circularMean(this.offsets.map(entry => entry.offset));
    return { spread, difference: this.estimate === null ? null : Math.abs(wrapDeg(mean - this.estimate)), warn: this.estimate !== null && Math.abs(wrapDeg(mean - this.estimate)) > WARNING_DEG };
  }

  // The gyroscope's orientation turned to north, as device orientation angles; null until ready.
  get orientation() {
    if (!this.ready || !this.gyroscope) return null;
    const { alpha_deg, beta_deg, gamma_deg } = this.gyroscope;
    return { alpha_deg: ((alpha_deg + this.estimate) % 360 + 360) % 360, beta_deg, gamma_deg };
  }

  // Fix the estimate as the reference (first photo); later compass readings correct it slowly.
  lock(time) {
    if (this.locked || !this.ready) return;
    this.locked = true;
    this.lastCorrection = time;
  }
}

// Field of view calibration: an object far away is put on one of two lines across the image's longer side, then the
// phone is turned until the object sits on the other line. The lines are `MARKER_FRACTION` of the half image away from
// the centre, so the turn between the two camera directions is 2 atan(MARKER_FRACTION tan(fov / 2)) (pinhole camera).
export const MARKER_FRACTION = 0.8;

// The back camera's viewing direction (earth coordinates) for device orientation angles: the device's -z axis.
export function cameraForward(angles) {
  const rotation = rotationFromDeviceAngles(angles);
  return rotation.map(row => -row[2]);
}

// Field of view (degrees, across the longer image side) from the camera directions with the object on either line.
export function fieldOfViewFromTurn(firstForward, secondForward) {
  const cosine = Math.max(-1, Math.min(1, firstForward.reduce((sum, value, index) => sum + value * secondForward[index], 0)));
  const turn = Math.acos(cosine);
  return 2 * Math.atan(Math.tan(turn / 2) / MARKER_FRACTION) * 180 / Math.PI;
}
