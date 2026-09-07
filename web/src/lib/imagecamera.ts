/**
 * Line the reconstructed mesh up with the photograph VGGT actually saw.
 *
 * The point of this is to let someone place the cut on a picture of the limb
 * rather than on a reconstruction of it. The marker band is *visible* in the
 * photo and only *inferred* in the mesh, so clicking the band is a far more
 * direct action than judging where a translucent disc meets a surface.
 *
 * The chain from a mesh vertex to a pixel is exact, with nothing fitted:
 *
 *     scene  --(undo the view transform)-->  levelled
 *     levelled  --(R_totalᵀ)-->             VGGT world
 *     VGGT world  --(extrinsic)-->          camera
 *     camera  --(intrinsic)-->              pixel
 *
 * Rather than move the mesh, this moves the CAMERA into the scene space the
 * mesh is already loaded in — same picture, and the geometry never has to be
 * reuploaded when the frame changes.
 *
 * Two conventions differ and both matter. VGGT's extrinsic is OpenCV-style:
 * it maps world to camera, the camera looks down +Z, and +Y points down.
 * three.js looks down -Z with +Y up. The basis built below flips the two axes
 * that disagree, which keeps the result right-handed.
 *
 * Measured on inputs/champ: fx/fy is 1.0000–1.0006 and the principal point sits
 * at exactly (259, 259) on a 518px frame, so square pixels and a centred
 * principal point are safe assumptions and a plain PerspectiveCamera is enough.
 * The focal length is NOT constant across frames — 55.00°, 54.40° and 53.32° on
 * three frames of one capture — so the field of view has to be taken from the
 * frame being shown, never set once.
 */
import * as THREE from "three";

/** VGGT's per-frame cameras, as `01_inference/raw/cameras.json` stores them. */
export interface CameraData {
  /** Per frame, a 3x4 world-to-camera matrix, row major. */
  extrinsic: number[][][];
  /** Per frame, a 3x3 pinhole intrinsic matrix, row major. */
  intrinsic: number[][][];
}

/** Stage 3's levelling, as `03_clean/debug/levelling.json` stores it. */
export interface LevellingData {
  /** 3x3 rotation, row major. The file's own note: levelled = R_total @ pointmap. */
  R_total: number[][];
}

/** How the mesh was placed in scene space by `transformToScene`. */
export interface ScenePlacement {
  scale: number;
  offset: THREE.Vector3;
}

/** The pose and lens of one frame, ready to drive a three.js camera. */
export interface FrameCamera {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  verticalFieldOfViewDeg: number;
  aspect: number;
}

// Builds a Matrix3 from row-major rows, which is how both cameras.json and
// levelling.json store their matrices.
function matrixFromRows(rows: number[][]): THREE.Matrix3 {
  const matrix = new THREE.Matrix3();
  matrix.set(
    rows[0][0], rows[0][1], rows[0][2],
    rows[1][0], rows[1][1], rows[1][2],
    rows[2][0], rows[2][1], rows[2][2],
  );
  return matrix;
}

/**
 * Builds the rotation taking a VGGT world direction into scene space.
 *
 * `transformToScene` rotates the levelled mesh -90° about X to go from the
 * pipeline's Z-up to three.js's Y-up, then scales and translates it. Only the
 * rotations matter for a direction, and they compose as: rotate by R_total to
 * reach levelled space, then by -90° about X to reach scene space.
 */
function worldToSceneRotation(levelling: LevellingData): THREE.Matrix4 {
  const rotationToLevelled = matrixFromRows(levelling.R_total);
  const toLevelled = new THREE.Matrix4().setFromMatrix3(rotationToLevelled);
  const zUpToYUp = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
  return zUpToYUp.multiply(toLevelled);
}

/**
 * Places a three.js camera so it sees the mesh exactly as this frame's photo does.
 *
 * Returns null when the frame index is out of range for the loaded cameras.
 */
export function cameraForFrame(
  cameras: CameraData,
  levelling: LevellingData,
  placement: ScenePlacement,
  frameIndex: number,
  imageSize: number,
): FrameCamera | null {
  if (frameIndex < 0 || frameIndex >= cameras.extrinsic.length) return null;

  const extrinsic = cameras.extrinsic[frameIndex];
  const intrinsic = cameras.intrinsic[frameIndex];

  // The rotation and translation halves of the 3x4 world-to-camera matrix.
  const rotationRows = [
    [extrinsic[0][0], extrinsic[0][1], extrinsic[0][2]],
    [extrinsic[1][0], extrinsic[1][1], extrinsic[1][2]],
    [extrinsic[2][0], extrinsic[2][1], extrinsic[2][2]],
  ];
  const translation = new THREE.Vector3(
    extrinsic[0][3], extrinsic[1][3], extrinsic[2][3],
  );

  // p_camera = R p_world + t, so the camera sits at -Rᵀt in world space.
  const worldToCamera = matrixFromRows(rotationRows);
  const cameraToWorld = worldToCamera.clone().transpose();
  const positionWorld = translation.clone()
    .applyMatrix3(cameraToWorld)
    .multiplyScalar(-1);

  const worldToScene = worldToSceneRotation(levelling);

  // The mesh was rotated, then scaled, then translated. A point follows all
  // three; a direction follows only the rotation.
  const positionScene = positionWorld.clone()
    .applyMatrix4(worldToScene)
    .multiplyScalar(placement.scale)
    .add(placement.offset);

  // The rows of R are the camera's own axes expressed in world coordinates.
  const axisInScene = (row: number[]) =>
    new THREE.Vector3(row[0], row[1], row[2])
      .applyMatrix4(worldToScene)
      .normalize();

  const cameraRight = axisInScene(rotationRows[0]);
  const cameraDown = axisInScene(rotationRows[1]);
  const cameraForward = axisInScene(rotationRows[2]);

  // three.js looks down -Z with +Y up; OpenCV looks down +Z with +Y down. Flip
  // the two that disagree, which leaves the basis right-handed.
  const basis = new THREE.Matrix4().makeBasis(
    cameraRight,
    cameraDown.clone().multiplyScalar(-1),
    cameraForward.clone().multiplyScalar(-1),
  );
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(basis);

  const focalY = intrinsic[1][1];
  const verticalFieldOfViewDeg =
    (2 * Math.atan(imageSize / (2 * focalY)) * 180) / Math.PI;

  return {
    position: positionScene,
    quaternion,
    verticalFieldOfViewDeg,
    aspect: 1,
  };
}

/**
 * Turns a click on the photo into a ray in scene space.
 *
 * `x` and `y` are fractions of the image, 0..1, with the origin top-left the
 * way an image is addressed. The ray starts at the camera and passes through
 * that pixel.
 */
export function rayThroughImagePoint(
  frameCamera: FrameCamera,
  x: number,
  y: number,
): THREE.Ray {
  // Normalised device coordinates: -1..1, with +Y up, hence the flip on y.
  const ndc = new THREE.Vector2(x * 2 - 1, -(y * 2 - 1));
  const halfHeight = Math.tan(
    (frameCamera.verticalFieldOfViewDeg * Math.PI) / 360,
  );
  const halfWidth = halfHeight * frameCamera.aspect;

  // A direction in the camera's own frame, then rotated into scene space.
  const direction = new THREE.Vector3(
    ndc.x * halfWidth,
    ndc.y * halfHeight,
    -1,
  )
    .applyQuaternion(frameCamera.quaternion)
    .normalize();

  return new THREE.Ray(frameCamera.position.clone(), direction);
}
