// ============================================================================
// Existir como estructura vs. existir como acontecimiento
// 
//   Realidad 1: "Existir como estructura" (MediaPipe Pose + Hand Tracking)
//     - Lectura corporal adaptativa de espacio y proximidad.
//     - Lectura extendida de manos: dedos abiertos, falanges y palma articulada.
//     - Malla geométrica continua en constelación de luz.
//     - Huellas fósiles que persisten ~20s en el estanque.
//
//   Realidad 2: "Existir como acontecimiento" (OpenCV Espejismo Espectral)
//     - Reacción cromática por velocidad y dirección del movimiento:
//         * Movimiento lento/suave -> Cian o azul profundo (agua tranquila)
//         * Movimiento rápido/brusco -> Naranja, rojo intenso y magenta (calor y fricción)
//         * Hacia arriba -> Verde lima
//         * Hacia abajo -> Violeta eléctrico
//         * Movimientos laterales -> Cian y amarillo solar
//     - "El Espejismo": Descomposición espectral en los colores primarios de la luz
//         con dispersión prismática en agua turbulenta.
//     - Rápida evaporación en quietud (cero memoria).
// ============================================================================

const VISION_BUNDLE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// ---------------------------------------------------------------------------
// Referencias DOM
// ---------------------------------------------------------------------------

const video = document.getElementById("video");
const cameraToggle = document.getElementById("cameraToggle");
const statusMsg = document.getElementById("statusMsg");

const canvasStructure = document.getElementById("canvasStructure");
const ctxStructure = canvasStructure.getContext("2d");

const canvasEvent = document.getElementById("canvasEvent");
const ctxEvent = canvasEvent.getContext("2d");

const hiddenDiffSample = document.getElementById("hiddenDiffSample");
const ctxDiff = hiddenDiffSample.getContext("2d", { willReadFrequently: true });

const statStructure = document.getElementById("statStructure");
const statEvent = document.getElementById("statEvent");

const mouseRippleCanvas = document.getElementById("mouseRippleCanvas");
const ctxRipple = mouseRippleCanvas ? mouseRippleCanvas.getContext("2d") : null;

// ---------------------------------------------------------------------------
// Estado Global
// ---------------------------------------------------------------------------

let poseLandmarker = null;
let handLandmarker = null;
let running = false;
let mediaStream = null;
let lastVideoTime = -1;
let frameCount = 0;

// Estado Sistema 1 (Estructura - MediaPipe Cuerpo + Manos)
let structureFootprints = [];
let lastStructureCentroid = null;
const FOOTPRINT_LIFETIME = 20000; // 20 segundos de persistencia como fósil
let smoothLandmarks = null;
let smoothHands = [];
const LERP_FACTOR = 0.42;

// Estado Sistema 2 (Acontecimiento - OpenCV Espejismo Espectral)
const DIFF_W = 160;
const DIFF_H = 120;
let prevGrayFrame = null;
let liquidMiragePuffs = [];
const MAX_MIRAGE_PUFFS = 220;
let disturbanceIntensity = 0;

// ---------------------------------------------------------------------------
// Ayuda de Texto de Estado
// ---------------------------------------------------------------------------

function setStatus(line1, line2) {
  statusMsg.innerHTML = `<span class="status-line-1">${line1}</span><span class="status-line-2">${line2}</span>`;
}

// ---------------------------------------------------------------------------
// Control del Sensor Óptico
// ---------------------------------------------------------------------------

cameraToggle.addEventListener("click", toggleCamera);

async function toggleCamera() {
  if (running) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  cameraToggle.setAttribute("aria-checked", "true");
  setStatus("Interpretando la escena…", "Dándole sentido a cosas que probablemente no lo necesitan.");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = mediaStream;
    await video.play();

    await new Promise((resolve) => {
      if (video.readyState >= 2) return resolve();
      video.onloadedmetadata = () => resolve();
    });

    running = true;
    requestAnimationFrame(renderLoop);

    if (!poseLandmarker || !handLandmarker) {
      await initModels();
    }

    setStatus("Hola tú…", "Ahora sí tenemos algo con qué jugar.");
  } catch (err) {
    console.error("Fallo de acceso a cámara:", err);
    setStatus("No podemos conocerte aún…", "La tecnología también tiene límites. Qué decepción.");
    stopCamera(false);
  }
}

function stopCamera(updateStatus = true) {
  running = false;
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
  cameraToggle.setAttribute("aria-checked", "false");

  if (updateStatus) {
    setStatus("Se perdió la señal.", "Y con ella, nuestras versiones de la realidad.");
  }

  clearCanvas(ctxStructure, canvasStructure.width, canvasStructure.height);
  clearCanvas(ctxEvent, canvasEvent.width, canvasEvent.height);

  structureFootprints = [];
  lastStructureCentroid = null;
  smoothLandmarks = null;
  smoothHands = [];
  prevGrayFrame = null;
  liquidMiragePuffs = [];

  statStructure.textContent = "Estructura en reposo";
  statEvent.textContent = "Superficie neutra // Sin alteración";
}

async function initModels() {
  const { PoseLandmarker, HandLandmarker, FilesetResolver } = await import(VISION_BUNDLE_URL);
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

  // 1. Pose Landmarker (Estructura corporal)
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1
    });
  } catch (err) {
    console.warn("Pose GPU no disponible, usando CPU:", err);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 1
    });
  }

  // 2. Hand Landmarker (Dedos abiertos y articulaciones finas de manos)
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2
    });
  } catch (err) {
    console.warn("Hand GPU no disponible, usando CPU:", err);
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "CPU" },
        runningMode: "VIDEO",
        numHands: 2
      });
    } catch (err2) {
      console.warn("HandLandmarker no disponible:", err2);
    }
  }
}

function clearCanvas(ctx, w, h) {
  ctx.fillStyle = "#040907";
  ctx.fillRect(0, 0, w, h);
}

clearCanvas(ctxStructure, canvasStructure.width, canvasStructure.height);
clearCanvas(ctxEvent, canvasEvent.width, canvasEvent.height);

// ---------------------------------------------------------------------------
// Loop Principal de Renderizado
// ---------------------------------------------------------------------------

function renderLoop(timestampMs) {
  if (!running) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    frameCount++;

    // Inferencia corporal con PoseLandmarker
    const poseResult = poseLandmarker ? poseLandmarker.detectForVideo(video, timestampMs) : null;
    const rawLandmarks = poseResult && poseResult.landmarks && poseResult.landmarks[0] ? poseResult.landmarks[0] : null;

    // Inferencia de manos finas con HandLandmarker
    const handResult = handLandmarker ? handLandmarker.detectForVideo(video, timestampMs) : null;
    const rawHands = handResult && handResult.landmarks ? handResult.landmarks : [];

    if (!rawLandmarks && rawHands.length === 0) {
      smoothLandmarks = null;
      smoothHands = [];
      if (frameCount % 60 === 0) {
        setStatus("No hay nadie a la vista.", "O eso creemos.");
      }
    } else {
      if (statusMsg.textContent.includes("No hay nadie")) {
        setStatus("Hola tú…", "Ahora sí tenemos algo con qué jugar.");
      }

      // Suavizado EMA de cuerpo (Pose)
      if (rawLandmarks) {
        if (!smoothLandmarks) {
          smoothLandmarks = rawLandmarks.map(p => ({ x: p.x, y: p.y, z: p.z || 0, v: p.visibility ?? 1 }));
        } else {
          for (let i = 0; i < rawLandmarks.length; i++) {
            const raw = rawLandmarks[i];
            const sm = smoothLandmarks[i];
            sm.x += (raw.x - sm.x) * LERP_FACTOR;
            sm.y += (raw.y - sm.y) * LERP_FACTOR;
            sm.z += ((raw.z || 0) - sm.z) * LERP_FACTOR;
            sm.v += ((raw.visibility ?? 1) - sm.v) * LERP_FACTOR;
          }
        }
      }

      // Suavizado EMA de manos finas (Hands)
      if (rawHands.length > 0) {
        if (smoothHands.length !== rawHands.length) {
          smoothHands = rawHands.map(h => h.map(p => ({ x: p.x, y: p.y, z: p.z || 0 })));
        } else {
          for (let hIdx = 0; hIdx < rawHands.length; hIdx++) {
            const rawHand = rawHands[hIdx];
            const smHand = smoothHands[hIdx];
            for (let i = 0; i < rawHand.length; i++) {
              smHand[i].x += (rawHand[i].x - smHand[i].x) * (LERP_FACTOR * 1.1);
              smHand[i].y += (rawHand[i].y - smHand[i].y) * (LERP_FACTOR * 1.1);
            }
          }
        }
      } else {
        smoothHands = [];
      }
    }

    // Renderizar Sistema 1: Existir como estructura (MediaPipe Cuerpo + Manos abiertas)
    drawSystem1_Structure(smoothLandmarks, smoothHands, timestampMs);

    // Renderizar Sistema 2: Existir como acontecimiento (OpenCV Espejismo Espectral)
    drawSystem2_Event(timestampMs);
  }

  requestAnimationFrame(renderLoop);
}

// ---------------------------------------------------------------------------
// SISTEMA 1: EXISTIR COMO ESTRUCTURA (MediaPipe)
// - Lectura real de distancia y proximidad (cuerpo crece al acercarse).
// - Malla facial en plano cercano.
// - Lectura de manos con dedos abiertos (21 hitos por mano: falanges y palma).
// ---------------------------------------------------------------------------

const FACIAL_TRIANGLES = [
  [0, 1, 2], [0, 4, 5], [1, 2, 3], [4, 5, 6],
  [2, 7, 9], [5, 8, 10], [0, 2, 7], [0, 5, 8],
  [0, 9, 10], [9, 10, 11], [9, 10, 12]
];

const TORSO_TRIANGLES = [
  [0, 11, 12], [11, 12, 23], [12, 23, 24],
  [11, 13, 23], [12, 14, 24], [13, 15, 11], [14, 16, 12]
];

// Triangulación anatómica de dedos abiertos y palma para cada mano (21 puntos)
const HAND_MESH_TRIANGLES = [
  // Palma
  [0, 1, 5], [0, 5, 9], [0, 9, 13], [0, 13, 17], [1, 5, 9],
  // Pulgar
  [1, 2, 3], [2, 3, 4], [1, 2, 5],
  // Índice
  [5, 6, 7], [6, 7, 8], [5, 6, 9],
  // Medio
  [9, 10, 11], [10, 11, 12], [9, 10, 13],
  // Anular
  [13, 14, 15], [14, 15, 16], [13, 14, 17],
  // Meñique
  [17, 18, 19], [18, 19, 20]
];

function drawSystem1_Structure(bodyLandmarks, handsLandmarks, timestampMs) {
  const w = canvasStructure.width;
  const h = canvasStructure.height;

  ctxStructure.fillStyle = "#040907";
  ctxStructure.fillRect(0, 0, w, h);

  structureFootprints = structureFootprints.filter(fp => (timestampMs - fp.createdAt) < FOOTPRINT_LIFETIME);

  // Proyectar puntos de cuerpo a píxeles directos
  let bodyPoints = {};
  let isCloseUp = false;
  let centroid = null;

  if (bodyLandmarks) {
    for (let i = 0; i < bodyLandmarks.length; i++) {
      const p = bodyLandmarks[i];
      if (p && (p.v ?? 1) > 0.35) {
        bodyPoints[i] = { x: p.x * w, y: p.y * h };
      }
    }

    const eyeL = bodyPoints[2], eyeR = bodyPoints[5];
    const shL = bodyPoints[11], shR = bodyPoints[12];
    const nose = bodyPoints[0];

    let apparentScale = 0;
    if (eyeL && eyeR) {
      apparentScale = Math.hypot(eyeR.x - eyeL.x, eyeR.y - eyeL.y);
    } else if (shL && shR) {
      apparentScale = Math.hypot(shR.x - shL.x, shR.y - shL.y) * 0.4;
    }

    isCloseUp = apparentScale > 72;

    const cx = nose ? nose.x : (shL && shR ? (shL.x + shR.x) / 2 : w / 2);
    const cy = nose ? nose.y : (shL && shR ? (shL.y + shR.y) / 2 : h / 2);
    centroid = { x: cx, y: cy };
  }

  // Proyectar puntos de manos finas (si están abiertas / presentes)
  let handsPoints = [];
  if (handsLandmarks && handsLandmarks.length > 0) {
    for (let hData of handsLandmarks) {
      const hMap = {};
      for (let i = 0; i < hData.length; i++) {
        hMap[i] = { x: hData[i].x * w, y: hData[i].y * h };
      }
      handsPoints.push(hMap);
    }
  }

  // Grabar nueva huella fósil en movimiento
  if (centroid) {
    let distMoved = 100;
    if (lastStructureCentroid) {
      distMoved = Math.hypot(centroid.x - lastStructureCentroid.x, centroid.y - lastStructureCentroid.y);
    }

    if (distMoved >= 16 || !lastStructureCentroid) {
      lastStructureCentroid = centroid;
      structureFootprints.push({
        bodyPoints: { ...bodyPoints },
        handsPoints: handsPoints.map(h => ({ ...h })),
        createdAt: timestampMs,
        isCloseUp: isCloseUp
      });
    }
  }

  // 1. DIBUJAR HUELLAS ACUMULADAS (Fósiles de luz congelados)
  for (let fp of structureFootprints) {
    const age = timestampMs - fp.createdAt;
    const progress = age / FOOTPRINT_LIFETIME;
    const alpha = (1 - progress) * 0.28;

    ctxStructure.save();
    ctxStructure.fillStyle = `rgba(45, 179, 153, ${alpha * 0.35})`;
    ctxStructure.strokeStyle = `rgba(63, 226, 197, ${alpha * 0.85})`;
    ctxStructure.lineWidth = 1.0;

    // Cuerpo
    const bPts = fp.bodyPoints;
    const activeTriangles = [
      ...FACIAL_TRIANGLES,
      ...(fp.isCloseUp ? [] : TORSO_TRIANGLES)
    ];

    for (let [iA, iB, iC] of activeTriangles) {
      const pA = bPts[iA], pB = bPts[iB], pC = bPts[iC];
      if (pA && pB && pC) {
        ctxStructure.beginPath();
        ctxStructure.moveTo(pA.x, pA.y);
        ctxStructure.lineTo(pB.x, pB.y);
        ctxStructure.lineTo(pC.x, pC.y);
        ctxStructure.closePath();
        ctxStructure.fill();
        ctxStructure.stroke();
      }
    }

    // Manos en huella
    if (fp.handsPoints) {
      for (let hMap of fp.handsPoints) {
        for (let [iA, iB, iC] of HAND_MESH_TRIANGLES) {
          const pA = hMap[iA], pB = hMap[iB], pC = hMap[iC];
          if (pA && pB && pC) {
            ctxStructure.beginPath();
            ctxStructure.moveTo(pA.x, pA.y);
            ctxStructure.lineTo(pB.x, pB.y);
            ctxStructure.lineTo(pC.x, pC.y);
            ctxStructure.closePath();
            ctxStructure.fill();
            ctxStructure.stroke();
          }
        }
      }
    }

    ctxStructure.restore();
  }

  // 2. DIBUJAR LA ESTRUCTURA VIVA ACTUAL
  const hasBody = Object.keys(bodyPoints).length >= 4;
  const hasHands = handsPoints.length > 0;

  if (hasBody || hasHands) {
    ctxStructure.save();

    const glowIntensity = isCloseUp ? 14 : 8;
    const strokeWidth = isCloseUp ? 1.8 : 1.3;

    ctxStructure.fillStyle = isCloseUp ? "rgba(45, 179, 153, 0.22)" : "rgba(45, 179, 153, 0.15)";
    ctxStructure.strokeStyle = "rgba(180, 255, 240, 0.75)";
    ctxStructure.lineWidth = strokeWidth;
    ctxStructure.shadowColor = "#3fe2c5";
    ctxStructure.shadowBlur = glowIntensity;

    // A. Triángulos del cuerpo
    if (hasBody) {
      const activeTriangles = [
        ...FACIAL_TRIANGLES,
        ...TORSO_TRIANGLES
      ];

      for (let [iA, iB, iC] of activeTriangles) {
        const pA = bodyPoints[iA], pB = bodyPoints[iB], pC = bodyPoints[iC];
        if (pA && pB && pC && pA.y <= h + 20 && pB.y <= h + 20 && pC.y <= h + 20) {
          ctxStructure.beginPath();
          ctxStructure.moveTo(pA.x, pA.y);
          ctxStructure.lineTo(pB.x, pB.y);
          ctxStructure.lineTo(pC.x, pC.y);
          ctxStructure.closePath();
          ctxStructure.fill();
          ctxStructure.stroke();
        }
      }

      // Nodos corporales
      ctxStructure.fillStyle = "#ffffff";
      for (let idx in bodyPoints) {
        const p = bodyPoints[idx];
        if (p && p.y <= h) {
          ctxStructure.beginPath();
          ctxStructure.arc(p.x, p.y, isCloseUp ? 3.5 : 2.4, 0, Math.PI * 2);
          ctxStructure.fill();
        }
      }
    }

    // B. Malla fina de manos (dedos abiertos y palma)
    if (hasHands) {
      ctxStructure.fillStyle = "rgba(63, 226, 197, 0.20)";
      ctxStructure.strokeStyle = "rgba(220, 255, 250, 0.85)";
      ctxStructure.lineWidth = 1.2;

      for (let hMap of handsPoints) {
        // Facetas de dedos y palma
        for (let [iA, iB, iC] of HAND_MESH_TRIANGLES) {
          const pA = hMap[iA], pB = hMap[iB], pC = hMap[iC];
          if (pA && pB && pC) {
            ctxStructure.beginPath();
            ctxStructure.moveTo(pA.x, pA.y);
            ctxStructure.lineTo(pB.x, pB.y);
            ctxStructure.lineTo(pC.x, pC.y);
            ctxStructure.closePath();
            ctxStructure.fill();
            ctxStructure.stroke();
          }
        }

        // Nodos en articulaciones y puntas de los dedos
        for (let i = 0; i <= 20; i++) {
          const p = hMap[i];
          if (p) {
            const isTip = [4, 8, 12, 16, 20].includes(i);
            ctxStructure.beginPath();
            ctxStructure.arc(p.x, p.y, isTip ? 3.8 : 2.0, 0, Math.PI * 2);
            ctxStructure.fillStyle = isTip ? "#ffffff" : "rgba(180, 255, 240, 0.9)";
            ctxStructure.fill();
          }
        }
      }
    }

    ctxStructure.restore();

    let statusText = isCloseUp ? "Primer plano cercano // Detalle facial" : "Estructura activa";
    if (hasHands) statusText += ` // Manos articuladas (${handsPoints.length})`;
    statStructure.textContent = `${statusText} // ${structureFootprints.length} huellas`;
  } else {
    statStructure.textContent = `Cuerpo ausente // ${structureFootprints.length} huellas preservadas`;
  }
}

// ---------------------------------------------------------------------------
// SISTEMA 2: EXISTIR COMO ACONTECIMIENTO (OpenCV Espejismo Espectral)
// Reacción cromática por velocidad y dirección:
// - Lento/Suave: Cian o azul profundo
// - Rápido/Brusco: Naranja, rojo intenso o magenta (calor y fricción)
// - Hacia arriba: Verde lima
// - Hacia abajo: Violeta eléctrico
// - Laterales: Cian o amarillo solar
// - "El Espejismo": Descomposición de la estela en colores primarios de la luz
// ---------------------------------------------------------------------------

function drawSystem2_Event(timestampMs) {
  const w = canvasEvent.width;
  const h = canvasEvent.height;

  // 1. Muestreo de video en baja resolución
  ctxDiff.drawImage(video, 0, 0, DIFF_W, DIFF_H);
  const frameData = ctxDiff.getImageData(0, 0, DIFF_W, DIFF_H).data;

  const totalPixels = DIFF_W * DIFF_H;
  const currentGray = new Float32Array(totalPixels);
  for (let i = 0; i < totalPixels; i++) {
    const px = i * 4;
    currentGray[i] = 0.299 * frameData[px] + 0.587 * frameData[px + 1] + 0.114 * frameData[px + 2];
  }

  // 2. Análisis de flujo óptico y vectores de movimiento (dirección + velocidad)
  let activeMotionPoints = [];
  if (prevGrayFrame) {
    for (let y = 2; y < DIFF_H - 2; y += 2) {
      for (let x = 2; x < DIFF_W - 2; x += 2) {
        const i = y * DIFF_W + x;
        const diff = Math.abs(currentGray[i] - prevGrayFrame[i]);

        if (diff > 16) {
          // Gradientes espaciales y temporal
          const gx = (currentGray[y * DIFF_W + x + 1] - currentGray[y * DIFF_W + x - 1]) * 0.5;
          const gy = (currentGray[(y + 1) * DIFF_W + x] - currentGray[(y - 1) * DIFF_W + x]) * 0.5;
          const gt = currentGray[i] - prevGrayFrame[i];

          const gradMagSq = gx * gx + gy * gy + 0.08;
          // Vector de velocidad óptica
          const vx = -(gt * gx) / gradMagSq;
          const vy = -(gt * gy) / gradMagSq;

          activeMotionPoints.push({
            x, y, diff,
            vx: Math.max(-4, Math.min(4, vx)),
            vy: Math.max(-4, Math.min(4, vy))
          });
        }
      }
    }

    const motionRatio = activeMotionPoints.length / (totalPixels / 4);
    disturbanceIntensity = disturbanceIntensity * 0.85 + motionRatio * 0.15;

    // Generación del espejismo espectral
    if (activeMotionPoints.length > 0) {
      activeMotionPoints.sort((a, b) => b.diff - a.diff);
      const spawnCount = Math.min(30, Math.floor(activeMotionPoints.length * 0.38) + 3);

      for (let k = 0; k < spawnCount; k++) {
        if (liquidMiragePuffs.length >= MAX_MIRAGE_PUFFS) break;
        const pt = activeMotionPoints[k % activeMotionPoints.length];

        const px = (pt.x / DIFF_W) * w;
        const py = (pt.y / DIFF_H) * h;

        const speed = Math.hypot(pt.vx, pt.vy);
        const speedNorm = Math.min(1.0, (speed + (pt.diff / 40.0)) * 0.5);

        // DETERMINACIÓN DE COLOR SEGÚN REGLA PERCEPTUAL:
        // 1. Velocidad:
        //    - Lento -> Cian o azul profundo
        //    - Rápido -> Naranja, rojo o magenta
        // 2. Dirección:
        //    - Hacia arriba -> Verde lima
        //    - Hacia abajo -> Violeta
        //    - Lateral izquierdo -> Cian
        //    - Lateral derecho -> Amarillo
        let primaryHue = 195; // base azul profundo
        let isFast = speedNorm > 0.55;

        if (isFast) {
          // Fricción violenta / calor rápido
          const heatOptions = [20, 5, 335]; // Naranja, Rojo intenso, Magenta
          primaryHue = heatOptions[Math.floor(Math.random() * heatOptions.length)];
        } else {
          // Dirección dominante
          const absVx = Math.abs(pt.vx);
          const absVy = Math.abs(pt.vy);

          if (absVy > absVx * 0.8 && absVy > 0.25) {
            if (pt.vy < 0) {
              // Mano hacia arriba: VERDE LIMA
              primaryHue = 98;
            } else {
              // Mano hacia abajo: VIOLETA
              primaryHue = 280;
            }
          } else if (absVx > 0.25) {
            if (pt.vx > 0) {
              // Movimiento derecha: AMARILLO
              primaryHue = 52;
            } else {
              // Movimiento izquierda: CIAN
              primaryHue = 180;
            }
          } else {
            // Movimiento suave neutro: Azul profundo / Cian
            primaryHue = 205;
          }
        }

        liquidMiragePuffs.push({
          x: px + (Math.random() - 0.5) * 14,
          y: py + (Math.random() - 0.5) * 14,
          vx: pt.vx * 1.5 + (Math.random() - 0.5) * 0.8,
          vy: pt.vy * 1.5 - 0.2, // ligera flotabilidad
          radius: 16 + Math.random() * 26,
          maxLife: 30 + Math.random() * 25, // ~0.8s a 1.1s de disolución
          age: 0,
          primaryHue: primaryHue,
          speedNorm: speedNorm
        });
      }
    }
  }
  prevGrayFrame = currentGray;

  // 3. Fondo con evaporación activa en quietud
  ctxEvent.fillStyle = "rgba(4, 9, 7, 0.20)";
  ctxEvent.fillRect(0, 0, w, h);

  // 4. Renderizado: El Espejismo (Descomposición en colores primarios de la luz)
  ctxEvent.save();
  ctxEvent.globalCompositeOperation = "screen";

  liquidMiragePuffs = liquidMiragePuffs.filter(p => p.age < p.maxLife);

  for (let p of liquidMiragePuffs) {
    p.age++;
    p.x += p.vx;
    p.y += p.vy;
    p.vx *= 0.93;
    p.vy *= 0.93;

    const progress = p.age / p.maxLife;
    const alpha = Math.sin(progress * Math.PI) * 0.65;
    const curRad = p.radius * (0.85 + progress * 0.65);

    // Separación espectral (Prismatic aberration offset)
    const splitDist = Math.max(2.5, p.speedNorm * 7.5);

    // CAPA 1: Componente Rojo/Cálido (ligeramente adelantada)
    const redHue = (p.primaryHue < 90 || p.primaryHue > 290) ? p.primaryHue : 15;
    const gradR = ctxEvent.createRadialGradient(p.x + splitDist, p.y - splitDist * 0.4, 0, p.x + splitDist, p.y - splitDist * 0.4, curRad);
    gradR.addColorStop(0, `hsla(${redHue}, 100%, 60%, ${alpha * 0.85})`);
    gradR.addColorStop(0.5, `hsla(${redHue}, 95%, 48%, ${alpha * 0.4})`);
    gradR.addColorStop(1, `hsla(${redHue}, 90%, 30%, 0)`);
    ctxEvent.beginPath();
    ctxEvent.arc(p.x + splitDist, p.y - splitDist * 0.4, curRad, 0, Math.PI * 2);
    ctxEvent.fillStyle = gradR;
    ctxEvent.fill();

    // CAPA 2: Componente Central Direccional (Verde/Amarillo/Tono principal)
    const gradG = ctxEvent.createRadialGradient(p.x, p.y, 0, p.x, p.y, curRad);
    gradG.addColorStop(0, `hsla(${p.primaryHue}, 100%, 65%, ${alpha * 0.95})`);
    gradG.addColorStop(0.4, `hsla(${p.primaryHue}, 95%, 52%, ${alpha * 0.6})`);
    gradG.addColorStop(1, `hsla(${p.primaryHue}, 90%, 35%, 0)`);
    ctxEvent.beginPath();
    ctxEvent.arc(p.x, p.y, curRad, 0, Math.PI * 2);
    ctxEvent.fillStyle = gradG;
    ctxEvent.fill();

    // CAPA 3: Componente Azul/Violeta (estela rezagada)
    const blueHue = (p.primaryHue >= 170 && p.primaryHue <= 290) ? p.primaryHue : 220;
    const gradB = ctxEvent.createRadialGradient(p.x - splitDist, p.y + splitDist * 0.4, 0, p.x - splitDist, p.y + splitDist * 0.4, curRad);
    gradB.addColorStop(0, `hsla(${blueHue}, 100%, 62%, ${alpha * 0.85})`);
    gradB.addColorStop(0.5, `hsla(${blueHue}, 95%, 50%, ${alpha * 0.4})`);
    gradB.addColorStop(1, `hsla(${blueHue}, 90%, 30%, 0)`);
    ctxEvent.beginPath();
    ctxEvent.arc(p.x - splitDist, p.y + splitDist * 0.4, curRad, 0, Math.PI * 2);
    ctxEvent.fillStyle = gradB;
    ctxEvent.fill();
  }

  ctxEvent.restore();

  // Subtítulo con lectura perceptual en vivo
  if (liquidMiragePuffs.length === 0 && disturbanceIntensity < 0.005) {
    statEvent.textContent = "Quietud absoluta // El espectro se ha disuelto";
  } else if (disturbanceIntensity < 0.04) {
    statEvent.textContent = `Estelas suaves en agua // ${liquidMiragePuffs.length} fragmentos prismáticos`;
  } else {
    statEvent.textContent = `Espejismo espectral activo // Descomposición cromática (${liquidMiragePuffs.length} vórtices)`;
  }
}

// ---------------------------------------------------------------------------
// Capa de Perturbación de Agua con Mouse (Lienzo 2D sobre la Fotografía)
// ---------------------------------------------------------------------------
let mouseRipples = [];
let lastPointerPos = { x: -999, y: -999 };
let lastPointerTime = 0;
let hadRipplesToClear = false;

function resizeRippleCanvas() {
  if (!mouseRippleCanvas) return;
  mouseRippleCanvas.width = window.innerWidth;
  mouseRippleCanvas.height = window.innerHeight;
}

function addMouseRipple(x, y, power = 1.0) {
  if (mouseRipples.length > 30) {
    mouseRipples.shift();
  }
  mouseRipples.push({
    x,
    y,
    radius: 3,
    maxRadius: Math.min(110, 60 + power * 35),
    speed: 1.8 + Math.min(power, 2.0) * 0.9,
    life: 1.0,
    decay: 0.016,
    power: Math.min(power, 2.5)
  });
}

function onPointerMove(e) {
  const now = performance.now();
  const dx = e.clientX - lastPointerPos.x;
  const dy = e.clientY - lastPointerPos.y;
  const dist = Math.hypot(dx, dy);

  // Detecta el desplazamiento del cursor para generar ondas líquidas
  if (dist > 14 && now - lastPointerTime > 30) {
    lastPointerPos = { x: e.clientX, y: e.clientY };
    lastPointerTime = now;
    addMouseRipple(e.clientX, e.clientY, dist / 22);
  }
}

function onPointerDown(e) {
  addMouseRipple(e.clientX, e.clientY, 2.2);
}

function renderMouseRipples() {
  if (ctxRipple && mouseRippleCanvas) {
    if (mouseRipples.length > 0) {
      ctxRipple.clearRect(0, 0, mouseRippleCanvas.width, mouseRippleCanvas.height);
      hadRipplesToClear = true;

      for (let i = mouseRipples.length - 1; i >= 0; i--) {
        const r = mouseRipples[i];
        r.radius += r.speed;
        r.life -= r.decay;

        if (r.life <= 0 || r.radius >= r.maxRadius) {
          mouseRipples.splice(i, 1);
          continue;
        }

        const progress = 1 - r.life;
        const alpha = Math.sin(r.life * Math.PI) * 0.38 * Math.min(r.power, 1.8);
        const waveWidth = 3.5 + progress * 7;

        ctxRipple.save();

        // 1. Cresta luminosa de refracción (cian agua pura)
        ctxRipple.beginPath();
        ctxRipple.arc(r.x, r.y, r.radius, 0, Math.PI * 2);
        ctxRipple.lineWidth = waveWidth;
        ctxRipple.strokeStyle = `rgba(160, 248, 230, ${alpha * 0.8})`;
        ctxRipple.stroke();

        // 2. Depresión o sombra líquida interna (profundidad refractiva)
        if (r.radius > 5) {
          ctxRipple.beginPath();
          ctxRipple.arc(r.x, r.y, Math.max(1, r.radius - waveWidth * 0.55), 0, Math.PI * 2);
          ctxRipple.lineWidth = waveWidth * 0.7;
          ctxRipple.strokeStyle = `rgba(1, 14, 10, ${alpha * 0.6})`;
          ctxRipple.stroke();
        }

        // 3. Sutil destello de perturbación en el epicentro inicial
        if (r.radius < 26) {
          const epicenterAlpha = (1 - r.radius / 26) * alpha * 0.45;
          ctxRipple.beginPath();
          ctxRipple.arc(r.x, r.y, r.radius * 0.45, 0, Math.PI * 2);
          ctxRipple.fillStyle = `rgba(220, 255, 250, ${epicenterAlpha})`;
          ctxRipple.fill();
        }

        ctxRipple.restore();
      }
    } else if (hadRipplesToClear) {
      // Cuando todas las ondas se apagan por completo, deja el lienzo en transparencia absoluta
      ctxRipple.clearRect(0, 0, mouseRippleCanvas.width, mouseRippleCanvas.height);
      hadRipplesToClear = false;
    }
  }

  requestAnimationFrame(renderMouseRipples);
}

// Inicialización de la capa interactiva del estanque
if (mouseRippleCanvas) {
  resizeRippleCanvas();
  window.addEventListener("resize", resizeRippleCanvas);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerdown", onPointerDown);
  requestAnimationFrame(renderMouseRipples);
}

