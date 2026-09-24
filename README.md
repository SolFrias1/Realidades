# Existir como estructura vs. existir como acontecimiento

> *¿Qué ocurre cuando una máquina recuerda tu forma, mientras otra solo recuerda tus cambios?*

**Ejercicio 02 — Percepción Artificial y Representación (DPPI 2026)**  
Escuela de Diseño UDP  
Autor: Felipe Roa R.

---

## Reflexión Conceptual

La obra explora cómo una misma realidad puede producir distintas formas de existencia dependiendo de aquello que se decide observar. Porque percibir no es simplemente recibir información: también es seleccionar, reducir e interpretar.

Frente a una misma cámara, cada sistema construye una versión posible de lo que ocurre a partir de aquello que puede reconocer; uno encuentra una estructura corporal, mientras el otro registra únicamente aquello que cambia. Lo observado permanece, pero la lectura lo transforma.

Estas versiones no buscan establecer una única verdad, sino hacer visible que toda interpretación tiene un punto de vista y, por lo tanto, un límite. Cada sistema revela algo y, al mismo tiempo, deja otras cosas fuera. Tal vez comprender una realidad no consiste en reducirla a una sola explicación, sino en permitir que pueda existir de distintas maneras al mismo tiempo. Nunca dejas de ser tú frente a la cámara; simplemente aparece una versión de ti que quizás todavía no habías conocido aún.

---

## Dos Criterios de Existencia

### 01. El cuerpo como forma suspendida (MediaPipe Pose + Hand Tracking)
* **Criterio de existencia**: La constelación anatómica, la geometría del cuerpo y la persistencia de la memoria.
* **Malla de luz articulada**: Una constelación geométrica translúcida de facetas que conectan el torso, hombros, cuello, cabeza y brazos con suavizado exponencial anti-jitter.
* **Extensión a manos y dedos abiertos**: Detección en tiempo real de 21 hitos articulares por mano (muñeca, falanges de los cinco dedos, membranas de dedos abiertos y nodos estelares de alta luminosidad en las yemas).
* **Adaptación a proximidad y lejanía**: Al acercarse a la cámara, el plano enfoca la topografía facial y torácica; al retroceder o desplazarse, se amplía la figura completa.
* **Huellas fósiles en el estanque**: Cada desplazamiento estampa siluetas minerales congeladas en el agua. La velocidad modula la distancia espacial entre huellas, creando una arqueología estructurada de formas etéreas que persisten flotando durante 20 segundos.

### 02. El cuerpo como consecuencia del cambio (OpenCV Espejismo Espectral)
* **Criterio de existencia**: La densidad cinética, la fricción y la disolución activa. Ignora por completo la identidad y la anatomía humana.
* **Reacción cromática por velocidad**:
  * *Movimiento lento / suave*: Estelas en tonos cian eléctrico y azul profundo (agua tranquila y calma líquida).
  * *Movimiento rápido / brusco*: Estallido en naranja solar, rojo incandescente y magenta (fricción térmica violenta y calor).
* **Reacción cromática por dirección cinemática**:
  * *Hacia arriba*: Verde lima brillante.
  * *Hacia abajo*: Violeta eléctrico.
  * *Movimientos laterales*: Cian puro y amarillo solar.
* **"El Espejismo" (Descomposición Prismática de la Luz)**: Cada perturbación se disgrega en sus colores primarios con dispersión prismática (RGB split) en capas offset que simulan la refracción en agua turbulenta.
* **Cero memoria**: En ausencia de movimiento, el espejismo se disipa y evapora en menos de 1 segundo, devolviendo el espacio a la quietud y oscuridad absoluta.

---

## Atmósfera Global: Estanque de Agua Digital

* **Fondo Prístino**: Fotografía real en alta resolución (`Imagen/FondoAguajpg.jpg`) conservada con su textura y color natural.
* **Capa Interactiva de Perturbación con el Mouse**: Lienzo 2D transparente superpuesto que genera ondas concéntricas de refracción líquida (crestas luminosas cian y sombras de depresión acuática) al desplazar o hacer clic con el cursor. Al detener el cursor, las ondas se amortiguan y el lienzo regresa al 100% de transparencia y reposo.

---

## Cómo Ejecutar el Proyecto

### Opción 1: Experiencia Web Completa (Recomendado)

Inicia un servidor local simple para servir los módulos ES y modelos de visión:

```bash
# Con Python 3:
python3 -m http.server 8000
```

1. Abre en tu navegador (Chrome, Edge, Safari o Firefox): `http://localhost:8000`
2. Presiona el interruptor interactivo **Sensor Óptico**.
3. Concede el permiso de acceso a la cámara cuando el navegador lo solicite.

### Opción 2: Script Independiente de Escritorio (Python + OpenCV nativo)

Para explorar la realidad de acontecimiento de forma aislada en una ventana nativa de OpenCV:

```bash
# Instalar dependencias:
pip install opencv-python numpy

# Ejecutar el script:
python3 acontecimiento.py
```

* **Controles en la ventana de Python**:
  * `ESC` o `q`: Salir.
  * `+` / `-`: Aumentar o disminuir la sensibilidad de detección.
  * `d` / `f`: Modificar la persistencia / inercia de las ondas.
  * `c`: Alternar paleta cromática (Acuática profunda, Ámbar otoñal, Espectral).
  * `r`: Limpiar el acumulador de perturbación.

---

## Estructura de Archivos del Repositorio

```text
├── index.html              # Interfaz web principal, marco dual y manifiesto
├── style.css               # Estilos visuales, atmósfera del estanque y tipografías
├── script.js               # Motores en tiempo real (MediaPipe Pose+Hands, OpenCV Espejismo, Ondas Mouse)
├── acontecimiento.py       # Versión de escritorio en Python OpenCV nativo
├── README.md               # Memoria conceptual y documentación técnica
└── Imagen/
    └── FondoAguajpg.jpg    # Fotografía de fondo del estanque digital
```

---

*Felipe Roa R. // DPPI 2026 — Escuela de Diseño UDP*
   
     