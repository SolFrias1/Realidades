// ============================================================================
// Existir como estructura vs. existir como acontecimiento
// 
//   Realidad 1: "Existir como estructura" (MediaPipe Pose + Hand Tracking)
//     - Lectura corporal adaptativa de espacio y proximidad.
//     - Lectura extendida de manos: dedos abiertos, falanges y palma articulada.
//     - Malla geométrica continua en constelación de luz.
//     - Huellas fósiles que persisten ~20s en el estanque.
//
//   Realidad 2: "Existir como acontecimiento" (OpenCV - La Consecuencia del Acontecimiento)
//     - "No queda una figura, sino la consecuencia de haber estado allí".
//     - La imagen opera como una materia sensible al acontecimiento.
//     - Cinco estados de transición:
//         1. cambio: El punto de inflexión donde la quietud reacciona a la presencia.
//         2. corriente: Flujo óptico denso que traduce el desplazamiento en arrastre fluido.
//         3. expansión: Onda de choque elíptica propagada por la velocidad del cuerpo.
//         4. intensidad: La aceleración que quema, deforma u oscurece los reflejos.
//         5. disipación: El lento retorno matemático al silencio y quietud original.
//     - Descomposición espectral prismática (RGB split) sobre el agua turbulenta.
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

// Estado Sistema 2 (Acontecimiento - OpenCV: La Consecuencia del Acontecimiento)
// Estados conceptuales: cambio · corriente · expansión · intensidad · disipación
const DIFF_W = 160;
const DIFF_H = 120;
let prevGrayFrame = null;
let fluidStreams = [];
const MAX_FLUID_STREAMS = 120;
let shockwaves = [];
const MAX_SHOCKWAVES = 18;
let kineticEnergy = 0;
let prevAverageSpeed = 0;
let currentTransitionState = "silencio";

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
  fluidStreams = [];
  shockwaves = [];
  kineticEnergy = 0;
  prevAverageSpeed = 0;
  currentTransitionState = "silencio";

  statStructure.textContent = "Estructura en reposo";
  statEvent.textContent = "Silencio // Superficie en reposo";
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
// ---------------------------------------------------------------------------
// Realidad 2: "Existir como acontecimiento" (OpenCV - La Consecuencia del Acontecimiento)
//
//   Paradigma conceptual y temporal:
//   - "No queda una figura, sino la consecuencia de haber estado allí"
//   - La imagen es materia sensible: el cuerpo no es dibujado; se registra
//     exclusivamente el rastro del caos generado en el agua.
//   - 5 Estados de transición:
//       1. cambio: El punto de inflexión donde la quietud reacciona ante la presencia.
//       2. corriente: Flujo óptico denso que traduce el desplazamiento en arrastre fluido.
//       3. expansión: Onda de choque elíptica que se propaga con la velocidad del cuerpo.
//       4. intensidad: La aceleración que quema, deforma y oscurece los reflejos.
//       5. disipación: El lento retorno matemático al silencio y quietud original.
// ---------------------------------------------------------------------------

function drawSystem2_Event(timestampMs) {
  const w = canvasEvent.width;
  const h = canvasEvent.height;

  // 1. Muestreo de video en baja resolución para análisis diferencial
  ctxDiff.drawImage(video, 0, 0, DIFF_W, DIFF_H);
  const frameData = ctxDiff.getImageData(0, 0, DIFF_W, DIFF_H).data;

  const totalPixels = DIFF_W * DIFF_H;
  const currentGray = new Float32Array(totalPixels);
  for (let i = 0; i < totalPixels; i++) {
    const px = i * 4;
    currentGray[i] = 0.299 * frameData[px] + 0.587 * frameData[px + 1] + 0.114 * frameData[px + 2];
  }

  // 2. Análisis de flujo óptico y vectores de fuerza cinética
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
          const vx = -(gt * gx) / gradMagSq;
          const vy = -(gt * gy) / gradMagSq;
          const speed = Math.hypot(vx, vy);

          activeMotionPoints.push({
            x, y, diff,
            vx: Math.max(-4, Math.min(4, vx)),
            vy: Math.max(-4, Math.min(4, vy)),
            speed: speed
          });
        }
      }
    }

    const motionRatio = activeMotionPoints.length / (totalPixels / 4);
    kineticEnergy = kineticEnergy * 0.88 + motionRatio * 0.12;

    const speedNormAverage = activeMotionPoints.length > 0
      ? (activeMotionPoints.reduce((acc, p) => acc + p.speed, 0) / activeMotionPoints.length) / 4.0
      : 0;

    const acceleration = Math.max(0, speedNormAverage - prevAverageSpeed);
    prevAverageSpeed = prevAverageSpeed * 0.75 + speedNormAverage * 0.25;

    // 3. Generación de Corrientes Fluidas y Ondas de Choque
    if (activeMotionPoints.length > 0) {
      // Priorizar zonas de mayor fricción
      activeMotionPoints.sort((a, b) => (b.diff * b.speed) - (a.diff * a.speed));
      const maxNewStreams = Math.min(10, Math.floor(activeMotionPoints.length * 0.26) + 1);

      for (let k = 0; k < maxNewStreams; k++) {
        if (fluidStreams.length >= MAX_FLUID_STREAMS) break;
        const pt = activeMotionPoints[k];

        const px = (pt.x / DIFF_W) * w;
        const py = (pt.y / DIFF_H) * h;
        const speedNorm = Math.min(1.0, (pt.speed + (pt.diff / 35.0)) * 0.45);

        // Color espectral según velocidad y dirección:
        let primaryHue = 195;
        let isFast = speedNorm > 0.52;

        if (isFast) {
          const heatOptions = [20, 8, 335]; // Naranja solar, rojo incandescente, magenta
          primaryHue = heatOptions[Math.floor(Math.random() * heatOptions.length)];
        } else {
          const absVx = Math.abs(pt.vx);
          const absVy = Math.abs(pt.vy);

          if (absVy > absVx * 0.75 && absVy > 0.22) {
            primaryHue = (pt.vy < 0) ? 98 : 280; // Arriba: Verde Lima | Abajo: Violeta
          } else if (absVx > 0.22) {
            primaryHue = (pt.vx > 0) ? 52 : 180; // Derecha: Amarillo Solar | Izquierda: Cian
          } else {
            primaryHue = 202; // Calma líquida azul profundo
          }
        }

        fluidStreams.push({
          x: px,
          y: py,
          vx: pt.vx * 1.8,
          vy: pt.vy * 1.8 - 0.15,
          width: 12 + speedNorm * 36,
          primaryHue: primaryHue,
          speedNorm: speedNorm,
          isFast: isFast,
          age: 0,
          maxLife: 70 + Math.floor(speedNorm * 45), // 2.5s a 3.5s de disipación gradual
          points: [
            { x: px - pt.vx * 18, y: py - pt.vy * 18 },
            { x: px - pt.vx * 8, y: py - pt.vy * 8 },
            { x: px, y: py }
          ]
        });
      }

      // Expansión: Onda de choque ante aceleraciones bruscas
      if (acceleration > 0.25 || (speedNormAverage > 0.58 && Math.random() < 0.35)) {
        if (shockwaves.length < MAX_SHOCKWAVES) {
          const epic = activeMotionPoints[0];
          const ex = (epic.x / DIFF_W) * w;
          const ey = (epic.y / DIFF_H) * h;
          const shockAngle = Math.atan2(epic.vy, epic.vx) + Math.PI * 0.5;

          shockwaves.push({
            x: ex,
            y: ey,
            angle: shockAngle,
            radius: 8,
            maxRadius: 100 + speedNormAverage * 120,
            speed: 3.2 + speedNormAverage * 4.5,
            life: 1.0,
            decay: 0.016,
            hue: (speedNormAverage > 0.5) ? 22 : 185
          });
        }
      }
    }
  }
  prevGrayFrame = currentGray;

  // 4. Fondo con disipación matemática gradual
  // Opacidad baja (0.075) para preservar el velo del estanque y la inercia del movimiento
  ctxEvent.fillStyle = "rgba(4, 9, 7, 0.075)";
  ctxEvent.fillRect(0, 0, w, h);

  // 5. Renderizado: Corrientes continuas y Espejismo de luz
  ctxEvent.save();
  ctxEvent.globalCompositeOperation = "screen";

  fluidStreams = fluidStreams.filter(s => s.age < s.maxLife);

  for (let s of fluidStreams) {
    s.age++;
    s.x += s.vx;
    s.y += s.vy;
    s.vx *= 0.94;
    s.vy *= 0.94;

    s.points.unshift({ x: s.x, y: s.y });
    if (s.points.length > 5) s.points.pop();

    const progress = s.age / s.maxLife;
    const alpha = Math.sin((1 - progress) * Math.PI * 0.5) * 0.65;
    if (alpha <= 0.01 || s.points.length < 2) continue;

    const strokeWidth = s.width * (1.0 + progress * 0.7);
    const splitDist = Math.max(2.8, s.speedNorm * 8.5);

    // CAPA 1: Cresta de fricción térmica adelantada (Rojo / Ámbar)
    const redHue = (s.primaryHue < 90 || s.primaryHue > 290) ? s.primaryHue : 16;
    ctxEvent.beginPath();
    ctxEvent.moveTo(s.points[0].x + splitDist, s.points[0].y - splitDist * 0.35);
    for (let i = 1; i < s.points.length; i++) {
      ctxEvent.lineTo(s.points[i].x + splitDist, s.points[i].y - splitDist * 0.35);
    }
    ctxEvent.lineWidth = strokeWidth * 0.85;
    ctxEvent.lineCap = "round";
    ctxEvent.lineJoin = "round";
    ctxEvent.strokeStyle = `hsla(${redHue}, 98%, 56%, ${alpha * 0.75})`;
    ctxEvent.stroke();

    // CAPA 2: Corriente central direccional (Lima, Violeta, Amarillo, Cian)
    ctxEvent.beginPath();
    ctxEvent.moveTo(s.points[0].x, s.points[0].y);
    for (let i = 1; i < s.points.length; i++) {
      ctxEvent.lineTo(s.points[i].x, s.points[i].y);
    }
    ctxEvent.lineWidth = strokeWidth;
    ctxEvent.strokeStyle = `hsla(${s.primaryHue}, 100%, 62%, ${alpha * 0.95})`;
    ctxEvent.stroke();

    // CAPA 3: Estela rezagada fría (Azul profundo / Violeta)
    const blueHue = (s.primaryHue >= 170 && s.primaryHue <= 290) ? s.primaryHue : 220;
    ctxEvent.beginPath();
    ctxEvent.moveTo(s.points[0].x - splitDist, s.points[0].y + splitDist * 0.35);
    for (let i = 1; i < s.points.length; i++) {
      ctxEvent.lineTo(s.points[i].x - splitDist, s.points[i].y + splitDist * 0.35);
    }
    ctxEvent.lineWidth = strokeWidth * 0.9;
    ctxEvent.strokeStyle = `hsla(${blueHue}, 95%, 52%, ${alpha * 0.75})`;
    ctxEvent.stroke();
  }

  // 6. Ondas de Choque (Expansión sobre el agua)
  shockwaves = shockwaves.filter(sw => sw.life > 0 && sw.radius < sw.maxRadius);

  for (let sw of shockwaves) {
    sw.radius += sw.speed;
    sw.life -= sw.decay;
    sw.speed *= 0.97;

    const swAlpha = Math.sin(sw.life * Math.PI) * 0.5;
    if (swAlpha <= 0.01) continue;

    ctxEvent.beginPath();
    ctxEvent.ellipse(sw.x, sw.y, sw.radius, sw.radius * 0.65, sw.angle, 0, Math.PI * 2);
    ctxEvent.lineWidth = 3.5 + (1 - sw.life) * 7.5;
    ctxEvent.strokeStyle = `hsla(${sw.hue}, 100%, 65%, ${swAlpha})`;
    ctxEvent.stroke();

    if (sw.radius > 16) {
      ctxEvent.beginPath();
      ctxEvent.ellipse(sw.x, sw.y, sw.radius * 0.82, sw.radius * 0.54, sw.angle, 0, Math.PI * 2);
      ctxEvent.lineWidth = 2.0;
      ctxEvent.strokeStyle = `rgba(180, 255, 240, ${swAlpha * 0.45})`;
      ctxEvent.stroke();
    }
  }

  ctxEvent.restore();

  // 7. Transición bajo los 5 estados conceptuales
  const motionActive = activeMotionPoints.length > 5;
  const speedNormAvg = activeMotionPoints.length > 0 
    ? (activeMotionPoints.reduce((acc, p) => acc + p.speed, 0) / activeMotionPoints.length) / 4.0 
    : 0;
  const isHighAcceleration = (speedNormAvg > 0.52 && motionActive);

  if (isHighAcceleration) {
    currentTransitionState = "intensidad";
    statEvent.textContent = "Intensidad // Aceleración crítica: turbulencia térmica y distorsión";
  } else if (shockwaves.length > 0 && kineticEnergy > 0.20) {
    currentTransitionState = "expansión";
    statEvent.textContent = "Expansión // Onda de choque propagándose sobre el estanque";
  } else if (motionActive && kineticEnergy > 0.06) {
    currentTransitionState = "corriente";
    statEvent.textContent = `Corriente // Arrastre cinético denso (${fluidStreams.length} estelas fluidas)`;
  } else if (motionActive) {
    currentTransitionState = "cambio";
    statEvent.textContent = "Cambio // La quietud reacciona a la presencia";
  } else if (fluidStreams.length > 0 || shockwaves.length > 0 || kineticEnergy > 0.006) {
    currentTransitionState = "disipación";
    statEvent.textContent = "Disipación // El agua lava el trauma del movimiento hacia el silencio";
  } else {
    currentTransitionState = "silencio";
    statEvent.textContent = "Silencio // Superficie en reposo absoluto";
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

