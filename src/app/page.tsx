"use client";

import { compile, MathNode } from "mathjs";
import { useEffect, useMemo, useRef, useState } from "react";

type Scope = {
  x: number;
  y: number;
  a: number;
  b: number;
  c: number;
  d: number;
};

type Trajectory = {
  x0: number;
  y0: number;
  color: string;
  startedAtMs: number;
};

type Preset = {
  name: string;
  fx: string;
  fy: string;
  params: Pick<Scope, "a" | "b" | "c" | "d">;
  domain: number;
};

type EquilibriumPoint = {
  x: number;
  y: number;
};

type EquilibriumAnalysis = {
  jacobian: {
    j11: number;
    j12: number;
    j21: number;
    j22: number;
  };
  eigenvalue1: string;
  eigenvalue2: string;
  classification: string;
  eigenDirections: Array<{
    x: number;
    y: number;
    eigenvalue: number;
  }>;
};

type TimeSeriesSample = {
  t: number;
  x: number;
  y: number;
};

type ManualModeState = {
  fxExpr: string;
  fyExpr: string;
  a: number;
  b: number;
  c: number;
  d: number;
  aInput: string;
  bInput: string;
  cInput: string;
  dInput: string;
  domain: number;
};

type PersistedAppState = {
  fxExpr: string;
  fyExpr: string;
  a: number;
  b: number;
  c: number;
  d: number;
  aInput: string;
  bInput: string;
  cInput: string;
  dInput: string;
  domain: number;
  showEquilibria: boolean;
  showEigenSpaces: boolean;
  showField: boolean;
  showTrajectoryAnimation: boolean;
  vectorDensity: number;
  stepSize: number;
  steps: number;
  selectedPreset: string;
  manualModeState: ManualModeState;
};

const PRESETS: Preset[] = [
  {
    name: "Lotka-Volterra",
    fx: "a*x - b*x*y",
    fy: "-c*y + d*x*y",
    params: { a: 1.2, b: 0.8, c: 1.0, d: 0.6 },
    domain: 5,
  },
  {
    name: "Sattel (linear)",
    fx: "a*x",
    fy: "-b*y",
    params: { a: 1, b: 1, c: 0, d: 0 },
    domain: 4,
  },
  {
    name: "Spirale (gedämpft)",
    fx: "a*x - y",
    fy: "x + a*y",
    params: { a: -0.2, b: 0, c: 0, d: 0 },
    domain: 4,
  },
  {
    name: "Van der Pol",
    fx: "y",
    fy: "a*(1-x^2)*y - x",
    params: { a: 1.5, b: 0, c: 0, d: 0 },
    domain: 4,
  },
  {
    name: "Gedämpftes Pendel",
    fx: "y",
    fy: "-sin(x) - a*y",
    params: { a: 0.2, b: 0, c: 0, d: 0 },
    domain: 7,
  },
  {
    name: "RLC-Schwingkreis (2. Ordnung)",
    fx: "a*y",
    fy: "-c*x - b*y",
    params: { a: 1, b: 0.25, c: 1, d: 0 },
    domain: 5,
  },
];

const TRAJECTORY_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"];
const NO_PRESET = "__none__";
const SESSION_STORAGE_KEY = "pplane-modern-state-v1";
const TRAJECTORY_TIME_SCALE = 1.4;
const MAX_SIMULTANEOUS_ANIMATIONS = 5;

function dedupePoints(points: Array<{ x: number; y: number }>, tolerance = 0.2) {
  const result: Array<{ x: number; y: number }> = [];
  for (const point of points) {
    const exists = result.some((candidate) => {
      const dx = candidate.x - point.x;
      const dy = candidate.y - point.y;
      return Math.hypot(dx, dy) < tolerance;
    });
    if (!exists) {
      result.push(point);
    }
  }
  return result;
}

function formatNumber(value: number, digits = 3) {
  if (!Number.isFinite(value)) {
    return "NaN";
  }
  return value.toFixed(digits);
}

function formatAxisTick(value: number) {
  const absValue = Math.abs(value);
  if (absValue > 0 && (absValue >= 10000 || absValue < 0.001)) {
    return value.toExponential(1);
  }
  if (absValue >= 100) {
    return value.toFixed(1);
  }
  if (absValue >= 10) {
    return value.toFixed(2);
  }
  return value.toFixed(3);
}

function computeEigenDirection(
  j11: number,
  j12: number,
  j21: number,
  j22: number,
  lambda: number,
) {
  const eps = 1e-10;

  let vx = 0;
  let vy = 0;
  if (Math.abs(j12) > Math.abs(j21) && Math.abs(j12) > eps) {
    vx = 1;
    vy = -((j11 - lambda) / j12);
  } else if (Math.abs(j21) > eps) {
    vx = -((j22 - lambda) / j21);
    vy = 1;
  } else if (Math.abs(j12) > eps) {
    vx = 1;
    vy = -((j11 - lambda) / j12);
  } else if (Math.abs(lambda - j11) <= Math.abs(lambda - j22)) {
    vx = 0;
    vy = 1;
  } else {
    vx = 1;
    vy = 0;
  }

  const length = Math.hypot(vx, vy);
  if (!Number.isFinite(length) || length <= eps) {
    return null;
  }

  return { x: vx / length, y: vy / length };
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const xTimeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const yTimeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const hasLoadedSessionRef = useRef(false);
  const holdStartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const holdRepeatTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const manualModeStateRef = useRef<ManualModeState>({
    fxExpr: PRESETS[0].fx,
    fyExpr: PRESETS[0].fy,
    a: PRESETS[0].params.a,
    b: PRESETS[0].params.b,
    c: PRESETS[0].params.c,
    d: PRESETS[0].params.d,
    aInput: String(PRESETS[0].params.a),
    bInput: String(PRESETS[0].params.b),
    cInput: String(PRESETS[0].params.c),
    dInput: String(PRESETS[0].params.d),
    domain: PRESETS[0].domain,
  });

  const [fxExpr, setFxExpr] = useState(PRESETS[0].fx);
  const [fyExpr, setFyExpr] = useState(PRESETS[0].fy);
  const [a, setA] = useState(PRESETS[0].params.a);
  const [b, setB] = useState(PRESETS[0].params.b);
  const [c, setC] = useState(PRESETS[0].params.c);
  const [d, setD] = useState(PRESETS[0].params.d);
  const [aInput, setAInput] = useState(String(PRESETS[0].params.a));
  const [bInput, setBInput] = useState(String(PRESETS[0].params.b));
  const [cInput, setCInput] = useState(String(PRESETS[0].params.c));
  const [dInput, setDInput] = useState(String(PRESETS[0].params.d));
  const [domain, setDomain] = useState(PRESETS[0].domain);
  const [showEquilibria, setShowEquilibria] = useState(true);
  const [showEigenSpaces, setShowEigenSpaces] = useState(true);
  const [showField, setShowField] = useState(true);
  const [showTrajectoryAnimation, setShowTrajectoryAnimation] = useState(true);
  const [vectorDensity, setVectorDensity] = useState(23);
  const [stepSize, setStepSize] = useState(0.03);
  const [steps, setSteps] = useState(1200);
  const [trajectories, setTrajectories] = useState<Trajectory[]>([]);
  const [selectedPreset, setSelectedPreset] = useState(PRESETS[0].name);
  const [hoveredEquilibrium, setHoveredEquilibrium] = useState<EquilibriumPoint | null>(null);
  const [selectedEquilibrium, setSelectedEquilibrium] = useState<EquilibriumPoint | null>(null);

  const compiled = useMemo(() => {
    try {
      const fx = compile(fxExpr) as MathNode;
      const fy = compile(fyExpr) as MathNode;
      return { fx, fy, error: "" };
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unbekannter Parserfehler";
      return { fx: null, fy: null, error: message };
    }
  }, [fxExpr, fyExpr]);

  const field = useMemo(() => {
    return (x: number, y: number) => {
      if (!compiled.fx || !compiled.fy) {
        return { dx: 0, dy: 0, valid: false };
      }
      const scope: Scope = { x, y, a, b, c, d };
      try {
        const dx = Number(compiled.fx.evaluate(scope));
        const dy = Number(compiled.fy.evaluate(scope));
        const valid = Number.isFinite(dx) && Number.isFinite(dy);
        return { dx: valid ? dx : 0, dy: valid ? dy : 0, valid };
      } catch {
        return { dx: 0, dy: 0, valid: false };
      }
    };
  }, [a, b, c, d, compiled.fx, compiled.fy]);

  const equilibria = useMemo(() => {
    const xMin = -domain;
    const xMax = domain;
    const yMin = -domain;
    const yMax = domain;
    const candidates: EquilibriumPoint[] = [];
    const samples = 70;
    const equilibriumThreshold = 0.06;
    for (let row = 0; row <= samples; row += 1) {
      for (let col = 0; col <= samples; col += 1) {
        const x = xMin + (col / samples) * (xMax - xMin);
        const y = yMin + (row / samples) * (yMax - yMin);
        const { dx, dy, valid } = field(x, y);
        if (valid && Math.abs(dx) < equilibriumThreshold && Math.abs(dy) < equilibriumThreshold) {
          candidates.push({ x, y });
        }
      }
    }
    return dedupePoints(candidates, (xMax - xMin) * 0.03);
  }, [domain, field]);

  const activeEquilibrium = hoveredEquilibrium ?? selectedEquilibrium;

  const activeTrajectory = useMemo(() => {
    if (trajectories.length === 0) {
      return null;
    }
    return trajectories[trajectories.length - 1];
  }, [trajectories]);

  const timeSeries = useMemo<TimeSeriesSample[]>(() => {
    if (!activeTrajectory) {
      return [];
    }

    const samples: TimeSeriesSample[] = [{ t: 0, x: activeTrajectory.x0, y: activeTrajectory.y0 }];
    let x = activeTrajectory.x0;
    let y = activeTrajectory.y0;
    let t = 0;

    for (let index = 0; index < steps; index += 1) {
      const h = stepSize;
      const k1 = field(x, y);
      const k2 = field(x + (h * k1.dx) / 2, y + (h * k1.dy) / 2);
      const k3 = field(x + (h * k2.dx) / 2, y + (h * k2.dy) / 2);
      const k4 = field(x + h * k3.dx, y + h * k3.dy);

      if (!k1.valid || !k2.valid || !k3.valid || !k4.valid) {
        break;
      }

      x += (h / 6) * (k1.dx + 2 * k2.dx + 2 * k3.dx + k4.dx);
      y += (h / 6) * (k1.dy + 2 * k2.dy + 2 * k3.dy + k4.dy);
      t += h;

      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e6 || Math.abs(y) > 1e6) {
        break;
      }

      samples.push({ t, x, y });
    }

    return samples;
  }, [activeTrajectory, field, stepSize, steps]);

  const equilibriumAnalysis = useMemo<EquilibriumAnalysis | null>(() => {
    if (!activeEquilibrium) {
      return null;
    }

    const x0 = activeEquilibrium.x;
    const y0 = activeEquilibrium.y;
    const h = Math.max(domain * 1e-4, 1e-5);

    const fxPlus = field(x0 + h, y0);
    const fxMinus = field(x0 - h, y0);
    const fyPlus = field(x0, y0 + h);
    const fyMinus = field(x0, y0 - h);

    if (!fxPlus.valid || !fxMinus.valid || !fyPlus.valid || !fyMinus.valid) {
      return null;
    }

    const j11 = (fxPlus.dx - fxMinus.dx) / (2 * h);
    const j21 = (fxPlus.dy - fxMinus.dy) / (2 * h);
    const j12 = (fyPlus.dx - fyMinus.dx) / (2 * h);
    const j22 = (fyPlus.dy - fyMinus.dy) / (2 * h);

    const trace = j11 + j22;
    const determinant = j11 * j22 - j12 * j21;
    const discriminant = trace * trace - 4 * determinant;
    const eps = 1e-8;

    let eigenvalue1 = "";
    let eigenvalue2 = "";
    let classification = "";
    const eigenDirections: EquilibriumAnalysis["eigenDirections"] = [];

    if (discriminant >= -eps) {
      const sqrtDisc = Math.sqrt(Math.max(0, discriminant));
      const lambda1 = (trace + sqrtDisc) / 2;
      const lambda2 = (trace - sqrtDisc) / 2;
      eigenvalue1 = formatNumber(lambda1);
      eigenvalue2 = formatNumber(lambda2);

      if (determinant < -eps) {
        classification = "Sattelpunkt";
      } else if (Math.abs(discriminant) <= eps) {
        if (trace < -eps) {
          classification = "Stabiler (degenerierter) Knoten";
        } else if (trace > eps) {
          classification = "Instabiler (degenerierter) Knoten";
        } else {
          classification = "Nicht-hyperbolischer Gleichgewichtspunkt";
        }
      } else if (lambda1 < -eps && lambda2 < -eps) {
        classification = "Stabiler Knoten";
      } else if (lambda1 > eps && lambda2 > eps) {
        classification = "Instabiler Knoten";
      } else if ((lambda1 < -eps && Math.abs(lambda2) <= eps) || (lambda2 < -eps && Math.abs(lambda1) <= eps)) {
        classification = "Semistabiler Knoten";
      } else if ((lambda1 > eps && Math.abs(lambda2) <= eps) || (lambda2 > eps && Math.abs(lambda1) <= eps)) {
        classification = "Semistabiler Knoten";
      } else {
        classification = "Nicht-hyperbolischer Gleichgewichtspunkt";
      }

      const direction1 = computeEigenDirection(j11, j12, j21, j22, lambda1);
      if (direction1) {
        eigenDirections.push({ x: direction1.x, y: direction1.y, eigenvalue: lambda1 });
      }

      if (Math.abs(lambda2 - lambda1) > eps) {
        const direction2 = computeEigenDirection(j11, j12, j21, j22, lambda2);
        if (direction2) {
          const duplicate = eigenDirections.some(
            (candidate) => Math.abs(candidate.x * direction2.y - candidate.y * direction2.x) < 1e-5,
          );
          if (!duplicate) {
            eigenDirections.push({ x: direction2.x, y: direction2.y, eigenvalue: lambda2 });
          }
        }
      }
    } else {
      const realPart = trace / 2;
      const imaginaryPart = Math.sqrt(-discriminant) / 2;
      eigenvalue1 = `${formatNumber(realPart)} + ${formatNumber(imaginaryPart)}i`;
      eigenvalue2 = `${formatNumber(realPart)} - ${formatNumber(imaginaryPart)}i`;

      if (Math.abs(realPart) <= eps) {
        classification = "Zentrum";
      } else if (realPart < 0) {
        classification = "Stabiler Strudel";
      } else {
        classification = "Instabiler Strudel";
      }
    }

    return {
      jacobian: { j11, j12, j21, j22 },
      eigenvalue1,
      eigenvalue2,
      classification,
      eigenDirections,
    };
  }, [activeEquilibrium, domain, field]);

  const handleSavePng = () => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const link = document.createElement("a");
    link.download = `pplane-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  useEffect(() => {
    const drawTimePlot = (
      canvas: HTMLCanvasElement | null,
      valueSelector: (sample: TimeSeriesSample) => number,
      label: string,
      color: string,
    ) => {
      if (!canvas) {
        return;
      }
      const context = canvas.getContext("2d");
      if (!context) {
        return;
      }

      const width = canvas.width;
      const height = canvas.height;
      const padLeft = 66;
      const padRight = 14;
      const padTop = 14;
      const padBottom = 30;
      const plotWidth = width - padLeft - padRight;
      const plotHeight = height - padTop - padBottom;

      context.clearRect(0, 0, width, height);
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, width, height);

      const tMax = timeSeries.length > 0 ? timeSeries[timeSeries.length - 1].t : 1;
      const values = timeSeries.map(valueSelector);
      let yMin = values.length > 0 ? Math.min(...values) : -1;
      let yMax = values.length > 0 ? Math.max(...values) : 1;

      if (Math.abs(yMax - yMin) < 1e-9) {
        yMin -= 1;
        yMax += 1;
      }

      const yPadding = (yMax - yMin) * 0.12;
      yMin -= yPadding;
      yMax += yPadding;

      const toX = (t: number) => padLeft + (t / Math.max(tMax, 1e-9)) * plotWidth;
      const toY = (value: number) => padTop + ((yMax - value) / (yMax - yMin)) * plotHeight;

      const yTickCount = 4;
      const yTicks = Array.from({ length: yTickCount + 1 }, (_, index) => {
        const ratio = index / yTickCount;
        const value = yMax - ratio * (yMax - yMin);
        return { ratio, value };
      });

      context.strokeStyle = "#e5e7eb";
      context.lineWidth = 1;
      for (let index = 0; index <= 6; index += 1) {
        const x = padLeft + (index / 6) * plotWidth;
        context.beginPath();
        context.moveTo(x, padTop);
        context.lineTo(x, padTop + plotHeight);
        context.stroke();
      }

      for (const tick of yTicks) {
        const y = padTop + tick.ratio * plotHeight;
        context.beginPath();
        context.moveTo(padLeft, y);
        context.lineTo(padLeft + plotWidth, y);
        context.stroke();
      }

      context.strokeStyle = "#9ca3af";
      context.lineWidth = 1.2;
      context.beginPath();
      context.moveTo(padLeft, padTop + plotHeight);
      context.lineTo(padLeft + plotWidth, padTop + plotHeight);
      context.stroke();
      context.beginPath();
      context.moveTo(padLeft, padTop);
      context.lineTo(padLeft, padTop + plotHeight);
      context.stroke();

      context.fillStyle = "#6b7280";
      context.font = "12px sans-serif";
      context.textAlign = "left";
      context.textBaseline = "top";
      context.fillText(label, padLeft + 4, 2);

      context.textAlign = "right";
      context.textBaseline = "middle";
      for (const tick of yTicks) {
        const y = padTop + tick.ratio * plotHeight;
        context.fillText(formatAxisTick(tick.value), padLeft - 8, y);
      }

      context.textAlign = "left";
      context.textBaseline = "top";
      context.fillText("0", padLeft, padTop + plotHeight + 6);
      context.textAlign = "right";
      context.fillText(formatAxisTick(tMax), padLeft + plotWidth, padTop + plotHeight + 6);
      context.textAlign = "left";
      context.fillText("t", padLeft + plotWidth + 6, padTop + plotHeight + 6);

      if (timeSeries.length > 1) {
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.beginPath();
        timeSeries.forEach((sample, index) => {
          const px = toX(sample.t);
          const py = toY(valueSelector(sample));
          if (index === 0) {
            context.moveTo(px, py);
          } else {
            context.lineTo(px, py);
          }
        });
        context.stroke();
      }
    };

    drawTimePlot(xTimeCanvasRef.current, (sample) => sample.x, "x(t)", "#2563eb");
    drawTimePlot(yTimeCanvasRef.current, (sample) => sample.y, "y(t)", "#dc2626");
  }, [timeSeries]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const xMin = -domain;
    const xMax = domain;
    const yMin = -domain;
    const yMax = domain;

    const toCanvasX = (x: number) => ((x - xMin) / (xMax - xMin)) * width;
    const toCanvasY = (y: number) => height - ((y - yMin) / (yMax - yMin)) * height;
    const toWorldX = (px: number) => xMin + (px / width) * (xMax - xMin);
    const toWorldY = (py: number) => yMin + ((height - py) / height) * (yMax - yMin);

    const integrate = (x0: number, y0: number, direction: 1 | -1) => {
      const points: Array<{ x: number; y: number; t: number }> = [{ x: x0, y: y0, t: 0 }];
      let x = x0;
      let y = y0;
      let t = 0;

      for (let index = 0; index < steps; index += 1) {
        const hSigned = stepSize * direction;

        const k1 = field(x, y);
        const k2 = field(x + (hSigned * k1.dx) / 2, y + (hSigned * k1.dy) / 2);
        const k3 = field(x + (hSigned * k2.dx) / 2, y + (hSigned * k2.dy) / 2);
        const k4 = field(x + hSigned * k3.dx, y + hSigned * k3.dy);

        if (!k1.valid || !k2.valid || !k3.valid || !k4.valid) {
          break;
        }

        x += (hSigned / 6) * (k1.dx + 2 * k2.dx + 2 * k3.dx + k4.dx);
        y += (hSigned / 6) * (k1.dy + 2 * k2.dy + 2 * k3.dy + k4.dy);
        t += stepSize;

        if (x < xMin - 1 || x > xMax + 1 || y < yMin - 1 || y > yMax + 1) {
          break;
        }

        points.push({ x, y, t });
      }

      return points;
    };

    const interpolateAtTime = (
      points: Array<{ x: number; y: number; t: number }>,
      targetTime: number,
    ) => {
      if (points.length === 0) {
        return null;
      }

      if (targetTime <= 0 || points.length === 1) {
        return points[0];
      }

      const last = points[points.length - 1];
      if (targetTime >= last.t) {
        return last;
      }

      for (let index = 1; index < points.length; index += 1) {
        const previous = points[index - 1];
        const current = points[index];
        if (targetTime <= current.t) {
          const dt = current.t - previous.t;
          const ratio = dt <= 1e-12 ? 0 : (targetTime - previous.t) / dt;
          return {
            x: previous.x + (current.x - previous.x) * ratio,
            y: previous.y + (current.y - previous.y) * ratio,
            t: targetTime,
          };
        }
      }

      return last;
    };

    let animationFrameId: number | null = null;
    let disposed = false;

    const drawScene = () => {
      if (disposed) {
        return;
      }

      const nowMs = performance.now();

      context.clearRect(0, 0, width, height);
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, width, height);

      context.strokeStyle = "#e5e7eb";
      context.lineWidth = 1;
      const integerLimit = Math.ceil(domain);
      for (let value = -integerLimit; value <= integerLimit; value += 1) {
        const x = toCanvasX(value);
        const y = toCanvasY(value);
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
      }

      context.strokeStyle = "#9ca3af";
      context.lineWidth = 1.2;
      context.beginPath();
      context.moveTo(toCanvasX(0), 0);
      context.lineTo(toCanvasX(0), height);
      context.stroke();
      context.beginPath();
      context.moveTo(0, toCanvasY(0));
      context.lineTo(width, toCanvasY(0));
      context.stroke();

      if (showField) {
        context.strokeStyle = "rgba(31, 41, 55, 0.65)";
        context.fillStyle = "rgba(31, 41, 55, 0.65)";
        for (let row = 0; row <= vectorDensity; row += 1) {
          for (let col = 0; col <= vectorDensity; col += 1) {
            const x = xMin + (col / vectorDensity) * (xMax - xMin);
            const y = yMin + (row / vectorDensity) * (yMax - yMin);
            const { dx, dy, valid } = field(x, y);
            if (!valid) {
              continue;
            }
            const length = Math.hypot(dx, dy);
            if (length < 1e-7) {
              continue;
            }
            const ux = dx / length;
            const uy = dy / length;
            const arrowScale = ((xMax - xMin) / vectorDensity) * 0.35;
            const x1 = x - (ux * arrowScale) / 2;
            const y1 = y - (uy * arrowScale) / 2;
            const x2 = x + (ux * arrowScale) / 2;
            const y2 = y + (uy * arrowScale) / 2;

            const px1 = toCanvasX(x1);
            const py1 = toCanvasY(y1);
            const px2 = toCanvasX(x2);
            const py2 = toCanvasY(y2);

            context.lineWidth = 1;
            context.beginPath();
            context.moveTo(px1, py1);
            context.lineTo(px2, py2);
            context.stroke();

            const arrowHead = 3.5;
            const angle = Math.atan2(py2 - py1, px2 - px1);
            context.beginPath();
            context.moveTo(px2, py2);
            context.lineTo(
              px2 - arrowHead * Math.cos(angle - Math.PI / 6),
              py2 - arrowHead * Math.sin(angle - Math.PI / 6),
            );
            context.lineTo(
              px2 - arrowHead * Math.cos(angle + Math.PI / 6),
              py2 - arrowHead * Math.sin(angle + Math.PI / 6),
            );
            context.closePath();
            context.fill();
          }
        }
      }

      if (showEquilibria) {
        for (const point of equilibria) {
          const isHovered =
            hoveredEquilibrium !== null && Math.hypot(point.x - hoveredEquilibrium.x, point.y - hoveredEquilibrium.y) < 1e-9;
          const isSelected =
            selectedEquilibrium !== null && Math.hypot(point.x - selectedEquilibrium.x, point.y - selectedEquilibrium.y) < 1e-9;

          if (isSelected || isHovered) {
            context.fillStyle = isHovered ? "#ef4444" : "#991b1b";
            context.beginPath();
            context.arc(toCanvasX(point.x), toCanvasY(point.y), 6, 0, 2 * Math.PI);
            context.fill();
          }

          context.fillStyle = "#dc2626";
          context.beginPath();
          context.arc(toCanvasX(point.x), toCanvasY(point.y), 4, 0, 2 * Math.PI);
          context.fill();
        }
      }

      if (showEigenSpaces && equilibria.length > 0) {
        const span = (xMax - xMin) * 2;
        const eps = 1e-8;
        const h = Math.max(domain * 1e-4, 1e-5);

        const getEigenDirectionsForPoint = (point: EquilibriumPoint) => {
          const fxPlus = field(point.x + h, point.y);
          const fxMinus = field(point.x - h, point.y);
          const fyPlus = field(point.x, point.y + h);
          const fyMinus = field(point.x, point.y - h);

          if (!fxPlus.valid || !fxMinus.valid || !fyPlus.valid || !fyMinus.valid) {
            return [] as Array<{ x: number; y: number; eigenvalue: number }>;
          }

          const j11 = (fxPlus.dx - fxMinus.dx) / (2 * h);
          const j21 = (fxPlus.dy - fxMinus.dy) / (2 * h);
          const j12 = (fyPlus.dx - fyMinus.dx) / (2 * h);
          const j22 = (fyPlus.dy - fyMinus.dy) / (2 * h);

          const trace = j11 + j22;
          const determinant = j11 * j22 - j12 * j21;
          const discriminant = trace * trace - 4 * determinant;
          if (discriminant < -eps) {
            return [] as Array<{ x: number; y: number; eigenvalue: number }>;
          }

          const sqrtDisc = Math.sqrt(Math.max(0, discriminant));
          const lambda1 = (trace + sqrtDisc) / 2;
          const lambda2 = (trace - sqrtDisc) / 2;
          const directions: Array<{ x: number; y: number; eigenvalue: number }> = [];

          const direction1 = computeEigenDirection(j11, j12, j21, j22, lambda1);
          if (direction1) {
            directions.push({ x: direction1.x, y: direction1.y, eigenvalue: lambda1 });
          }

          if (Math.abs(lambda2 - lambda1) > eps) {
            const direction2 = computeEigenDirection(j11, j12, j21, j22, lambda2);
            if (direction2) {
              const duplicate = directions.some(
                (candidate) => Math.abs(candidate.x * direction2.y - candidate.y * direction2.x) < 1e-5,
              );
              if (!duplicate) {
                directions.push({ x: direction2.x, y: direction2.y, eigenvalue: lambda2 });
              }
            }
          }

          return directions;
        };

        for (const point of equilibria) {
          const directions = getEigenDirectionsForPoint(point);
          const isActivePoint =
            activeEquilibrium !== null && Math.hypot(point.x - activeEquilibrium.x, point.y - activeEquilibrium.y) < 1e-9;

          directions.forEach((direction, index) => {
            context.strokeStyle = index % 2 === 0 ? "rgba(59, 130, 246, 0.9)" : "rgba(234, 88, 12, 0.9)";
            context.lineWidth = isActivePoint ? 2.1 : 1.4;
            context.setLineDash([6, 4]);
            context.beginPath();
            context.moveTo(
              toCanvasX(point.x - direction.x * span),
              toCanvasY(point.y - direction.y * span),
            );
            context.lineTo(
              toCanvasX(point.x + direction.x * span),
              toCanvasY(point.y + direction.y * span),
            );
            context.stroke();
            context.setLineDash([]);

            if (isActivePoint) {
              const labelX = toCanvasX(point.x + direction.x * (span * 0.25));
              const labelY = toCanvasY(point.y + direction.y * (span * 0.25));
              context.fillStyle = "#1f2937";
              context.font = "11px sans-serif";
              context.fillText(`λ=${formatNumber(direction.eigenvalue, 2)}`, labelX + 4, labelY - 4);
            }
          });
        }
      }

      let hasActiveAnimation = false;
      trajectories.forEach((trajectory) => {
        const forward = integrate(trajectory.x0, trajectory.y0, 1);
        if (forward.length === 0) {
          return;
        }

        if (!showTrajectoryAnimation) {
          const backward = integrate(trajectory.x0, trajectory.y0, -1);
          const fullPath = [...backward.reverse(), ...forward];

          context.strokeStyle = trajectory.color;
          context.lineWidth = 2.4;
          context.beginPath();
          fullPath.forEach((point, index) => {
            const cx = toCanvasX(point.x);
            const cy = toCanvasY(point.y);
            if (index === 0) {
              context.moveTo(cx, cy);
            } else {
              context.lineTo(cx, cy);
            }
          });
          context.stroke();

          context.fillStyle = trajectory.color;
          context.beginPath();
          context.arc(toCanvasX(trajectory.x0), toCanvasY(trajectory.y0), 3.8, 0, 2 * Math.PI);
          context.fill();
          return;
        }

        const lastPoint = forward[forward.length - 1];
        const elapsedModelTime = Math.max(0, ((nowMs - trajectory.startedAtMs) / 1000) * TRAJECTORY_TIME_SCALE);
        const visibleModelTime = Math.min(elapsedModelTime, lastPoint.t);
        if (visibleModelTime < lastPoint.t - 1e-6) {
          hasActiveAnimation = true;
        }

        const visiblePoints: Array<{ x: number; y: number }> = [];
        for (let index = 0; index < forward.length; index += 1) {
          const point = forward[index];
          if (point.t <= visibleModelTime) {
            visiblePoints.push({ x: point.x, y: point.y });
          } else {
            break;
          }
        }

        const movingPoint = interpolateAtTime(forward, visibleModelTime);
        if (movingPoint) {
          const lastVisiblePoint = visiblePoints[visiblePoints.length - 1];
          if (
            !lastVisiblePoint ||
            Math.hypot(lastVisiblePoint.x - movingPoint.x, lastVisiblePoint.y - movingPoint.y) > 1e-8
          ) {
            visiblePoints.push({ x: movingPoint.x, y: movingPoint.y });
          }
        }

        if (visiblePoints.length >= 2) {
          context.strokeStyle = trajectory.color;
          context.lineWidth = 2.4;
          context.beginPath();
          visiblePoints.forEach((point, index) => {
            const cx = toCanvasX(point.x);
            const cy = toCanvasY(point.y);
            if (index === 0) {
              context.moveTo(cx, cy);
            } else {
              context.lineTo(cx, cy);
            }
          });
          context.stroke();
        }

        if (movingPoint) {
          const velocity = field(movingPoint.x, movingPoint.y);
          if (velocity.valid) {
            const speed = Math.hypot(velocity.dx, velocity.dy);
            if (speed > 1e-10) {
              const ux = velocity.dx / speed;
              const uy = velocity.dy / speed;
              const dynamicScale = Math.min(3.4, 0.22 + speed * 0.55);
              const arrowLengthWorld = ((xMax - xMin) / 22) * dynamicScale;
              const arrowTailX = movingPoint.x;
              const arrowTailY = movingPoint.y;
              const arrowHeadX = movingPoint.x + ux * arrowLengthWorld;
              const arrowHeadY = movingPoint.y + uy * arrowLengthWorld;

              const pxTail = toCanvasX(arrowTailX);
              const pyTail = toCanvasY(arrowTailY);
              const pxHead = toCanvasX(arrowHeadX);
              const pyHead = toCanvasY(arrowHeadY);

              context.strokeStyle = trajectory.color;
              context.fillStyle = trajectory.color;
              context.lineWidth = 2;
              context.beginPath();
              context.moveTo(pxTail, pyTail);
              context.lineTo(pxHead, pyHead);
              context.stroke();

              const headSize = 9;
              const angle = Math.atan2(pyHead - pyTail, pxHead - pxTail);
              context.beginPath();
              context.moveTo(pxHead, pyHead);
              context.lineTo(
                pxHead - headSize * Math.cos(angle - Math.PI / 7),
                pyHead - headSize * Math.sin(angle - Math.PI / 7),
              );
              context.lineTo(
                pxHead - headSize * Math.cos(angle + Math.PI / 7),
                pyHead - headSize * Math.sin(angle + Math.PI / 7),
              );
              context.closePath();
              context.fill();
            }
          }

          context.fillStyle = trajectory.color;
          context.beginPath();
          context.arc(toCanvasX(movingPoint.x), toCanvasY(movingPoint.y), 6, 0, 2 * Math.PI);
          context.fill();

          context.fillStyle = "#ffffff";
          context.beginPath();
          context.arc(toCanvasX(movingPoint.x), toCanvasY(movingPoint.y), 2.2, 0, 2 * Math.PI);
          context.fill();
        }
      });

      if (hasActiveAnimation) {
        animationFrameId = window.requestAnimationFrame(drawScene);
      }
    };

    drawScene();

    const getNearestEquilibrium = (event: MouseEvent) => {
      if (!showEquilibria || equilibria.length === 0) {
        return null;
      }
      const rect = canvas.getBoundingClientRect();
      const px = ((event.clientX - rect.left) / rect.width) * width;
      const py = ((event.clientY - rect.top) / rect.height) * height;

      let nearest: EquilibriumPoint | null = null;
      let minDistance = Number.POSITIVE_INFINITY;
      for (const point of equilibria) {
        const dx = toCanvasX(point.x) - px;
        const dy = toCanvasY(point.y) - py;
        const distance = Math.hypot(dx, dy);
        if (distance < minDistance) {
          minDistance = distance;
          nearest = point;
        }
      }

      return minDistance <= 11 ? nearest : null;
    };

    const moveHandler = (event: MouseEvent) => {
      const nearest = getNearestEquilibrium(event);
      setHoveredEquilibrium(nearest);
      canvas.style.cursor = nearest ? "pointer" : "crosshair";
    };

    const leaveHandler = () => {
      setHoveredEquilibrium(null);
      canvas.style.cursor = "crosshair";
    };

    const clickHandler = (event: MouseEvent) => {
      const nearest = getNearestEquilibrium(event);
      if (nearest) {
        setSelectedEquilibrium(nearest);
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const x = toWorldX((px / rect.width) * width);
      const y = toWorldY((py / rect.height) * height);
      setTrajectories((previous) => {
        if (showTrajectoryAnimation) {
          const nowMs = performance.now();
          const maxAnimationDurationMs = ((steps * stepSize) / TRAJECTORY_TIME_SCALE) * 1000;
          const activeAnimationCount = previous.filter(
            (trajectory) => nowMs - trajectory.startedAtMs < maxAnimationDurationMs,
          ).length;
          if (activeAnimationCount >= MAX_SIMULTANEOUS_ANIMATIONS) {
            return previous;
          }
        }

        const color = TRAJECTORY_COLORS[previous.length % TRAJECTORY_COLORS.length];
        return [...previous, { x0: x, y0: y, color, startedAtMs: performance.now() }];
      });
    };

    canvas.style.cursor = "crosshair";
    canvas.addEventListener("mousemove", moveHandler);
    canvas.addEventListener("mouseleave", leaveHandler);
    canvas.addEventListener("click", clickHandler);
    return () => {
      disposed = true;
      if (animationFrameId !== null) {
        window.cancelAnimationFrame(animationFrameId);
      }
      canvas.removeEventListener("mousemove", moveHandler);
      canvas.removeEventListener("mouseleave", leaveHandler);
      canvas.removeEventListener("click", clickHandler);
    };
  }, [
    compiled.error,
    domain,
    activeEquilibrium,
    equilibria,
    field,
    hoveredEquilibrium,
    selectedEquilibrium,
    showEigenSpaces,
    showEquilibria,
    showField,
    showTrajectoryAnimation,
    stepSize,
    steps,
    trajectories,
    vectorDensity,
  ]);

  const applyPreset = (presetName: string) => {
    if (presetName === NO_PRESET) {
      const manual = manualModeStateRef.current;
      setFxExpr(manual.fxExpr);
      setFyExpr(manual.fyExpr);
      setA(manual.a);
      setB(manual.b);
      setC(manual.c);
      setD(manual.d);
      setAInput(manual.aInput);
      setBInput(manual.bInput);
      setCInput(manual.cInput);
      setDInput(manual.dInput);
      setDomain(manual.domain);
      setSelectedPreset(NO_PRESET);
      return;
    }

    if (selectedPreset === NO_PRESET) {
      manualModeStateRef.current = {
        fxExpr,
        fyExpr,
        a,
        b,
        c,
        d,
        aInput,
        bInput,
        cInput,
        dInput,
        domain,
      };
    }

    const preset = PRESETS.find((entry) => entry.name === presetName);
    if (!preset) {
      return;
    }
    setSelectedPreset(presetName);
    setFxExpr(preset.fx);
    setFyExpr(preset.fy);
    setA(preset.params.a);
    setB(preset.params.b);
    setC(preset.params.c);
    setD(preset.params.d);
    setAInput(String(preset.params.a));
    setBInput(String(preset.params.b));
    setCInput(String(preset.params.c));
    setDInput(String(preset.params.d));
    setDomain(preset.domain);
    setTrajectories([]);
  };

  const parseParamValue = (value: string) => {
    const normalized = value.trim().replace(/,/g, ".");
    if (normalized === "") {
      return null;
    }
    const parsed = Number(normalized);
    return Number.isFinite(parsed) ? parsed : null;
  };

  const handleParamChange = (
    rawValue: string,
    setInput: (value: string) => void,
    setNumeric: (value: number) => void,
  ) => {
    setSelectedPreset(NO_PRESET);
    setInput(rawValue);
    const parsed = parseParamValue(rawValue);
    if (parsed !== null) {
      setNumeric(parsed);
    }
  };

  const handleParamBlur = (
    inputValue: string,
    numericValue: number,
    setInput: (value: string) => void,
  ) => {
    const parsed = parseParamValue(inputValue);
    if (parsed === null) {
      setInput(String(numericValue));
      return;
    }
    setInput(String(parsed));
  };

  const adjustParamValue = (
    delta: number,
    inputValue: string,
    numericValue: number,
    setInput: (value: string) => void,
    setNumeric: (value: number) => void,
  ) => {
    const parsed = parseParamValue(inputValue);
    const base = parsed ?? numericValue;
    const nextValue = Math.round((base + delta) * 1000) / 1000;
    setSelectedPreset(NO_PRESET);
    setNumeric(nextValue);
    setInput(String(nextValue));
  };

  const stopParamHold = () => {
    if (holdStartTimeoutRef.current) {
      clearTimeout(holdStartTimeoutRef.current);
      holdStartTimeoutRef.current = null;
    }
    if (holdRepeatTimeoutRef.current) {
      clearTimeout(holdRepeatTimeoutRef.current);
      holdRepeatTimeoutRef.current = null;
    }
  };

  const startParamHold = (
    delta: number,
    inputValue: string,
    numericValue: number,
    setInput: (value: string) => void,
    setNumeric: (value: number) => void,
  ) => {
    stopParamHold();

    const parsedBase = parseParamValue(inputValue);
    const base = parsedBase ?? numericValue;
    let stepCount = 1;

    const applyStep = () => {
      const nextValue = Math.round((base + delta * stepCount) * 1000) / 1000;
      setSelectedPreset(NO_PRESET);
      setNumeric(nextValue);
      setInput(String(nextValue));
    };

    applyStep();

    holdStartTimeoutRef.current = setTimeout(() => {
      let tick = 0;
      const run = () => {
        tick += 1;
        stepCount += 1;
        applyStep();
        const delay = Math.max(28, 120 - tick * 6);
        holdRepeatTimeoutRef.current = setTimeout(run, delay);
      };
      run();
    }, 280);
  };

  useEffect(() => {
    return () => {
      stopParamHold();
    };
  }, []);

  useEffect(() => {
    if (selectedPreset !== NO_PRESET) {
      return;
    }

    manualModeStateRef.current = {
      fxExpr,
      fyExpr,
      a,
      b,
      c,
      d,
      aInput,
      bInput,
      cInput,
      dInput,
      domain,
    };
  }, [selectedPreset, fxExpr, fyExpr, a, b, c, d, aInput, bInput, cInput, dInput, domain]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      try {
        const raw = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
        if (!raw) {
          return;
        }
        const parsed = JSON.parse(raw) as Partial<PersistedAppState>;

        if (typeof parsed.fxExpr === "string") {
          setFxExpr(parsed.fxExpr);
        }
        if (typeof parsed.fyExpr === "string") {
          setFyExpr(parsed.fyExpr);
        }
        if (typeof parsed.a === "number" && Number.isFinite(parsed.a)) {
          setA(parsed.a);
          setAInput(typeof parsed.aInput === "string" ? parsed.aInput : String(parsed.a));
        }
        if (typeof parsed.b === "number" && Number.isFinite(parsed.b)) {
          setB(parsed.b);
          setBInput(typeof parsed.bInput === "string" ? parsed.bInput : String(parsed.b));
        }
        if (typeof parsed.c === "number" && Number.isFinite(parsed.c)) {
          setC(parsed.c);
          setCInput(typeof parsed.cInput === "string" ? parsed.cInput : String(parsed.c));
        }
        if (typeof parsed.d === "number" && Number.isFinite(parsed.d)) {
          setD(parsed.d);
          setDInput(typeof parsed.dInput === "string" ? parsed.dInput : String(parsed.d));
        }
        if (typeof parsed.domain === "number" && Number.isFinite(parsed.domain)) {
          setDomain(parsed.domain);
        }
        if (typeof parsed.showEquilibria === "boolean") {
          setShowEquilibria(parsed.showEquilibria);
        }
        if (typeof parsed.showEigenSpaces === "boolean") {
          setShowEigenSpaces(parsed.showEigenSpaces);
        }
        if (typeof parsed.showField === "boolean") {
          setShowField(parsed.showField);
        }
        if (typeof parsed.showTrajectoryAnimation === "boolean") {
          setShowTrajectoryAnimation(parsed.showTrajectoryAnimation);
        }
        if (typeof parsed.vectorDensity === "number" && Number.isFinite(parsed.vectorDensity)) {
          setVectorDensity(parsed.vectorDensity);
        }
        if (typeof parsed.stepSize === "number" && Number.isFinite(parsed.stepSize)) {
          setStepSize(parsed.stepSize);
        }
        if (typeof parsed.steps === "number" && Number.isFinite(parsed.steps)) {
          setSteps(parsed.steps);
        }
        if (typeof parsed.selectedPreset === "string") {
          setSelectedPreset(parsed.selectedPreset);
        }
        if (parsed.manualModeState && typeof parsed.manualModeState === "object") {
          const manual = parsed.manualModeState as Partial<ManualModeState>;
          if (
            typeof manual.fxExpr === "string" &&
            typeof manual.fyExpr === "string" &&
            typeof manual.a === "number" &&
            typeof manual.b === "number" &&
            typeof manual.c === "number" &&
            typeof manual.d === "number" &&
            typeof manual.aInput === "string" &&
            typeof manual.bInput === "string" &&
            typeof manual.cInput === "string" &&
            typeof manual.dInput === "string" &&
            typeof manual.domain === "number"
          ) {
            manualModeStateRef.current = {
              fxExpr: manual.fxExpr,
              fyExpr: manual.fyExpr,
              a: manual.a,
              b: manual.b,
              c: manual.c,
              d: manual.d,
              aInput: manual.aInput,
              bInput: manual.bInput,
              cInput: manual.cInput,
              dInput: manual.dInput,
              domain: manual.domain,
            };
          }
        }
      } catch {
      } finally {
        hasLoadedSessionRef.current = true;
      }
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !hasLoadedSessionRef.current) {
      return;
    }

    const stateToPersist: PersistedAppState = {
      fxExpr,
      fyExpr,
      a,
      b,
      c,
      d,
      aInput,
      bInput,
      cInput,
      dInput,
      domain,
      showEquilibria,
      showEigenSpaces,
      showField,
      showTrajectoryAnimation,
      vectorDensity,
      stepSize,
      steps,
      selectedPreset,
      manualModeState: manualModeStateRef.current,
    };

    window.sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(stateToPersist));
  }, [
    a,
    aInput,
    b,
    bInput,
    c,
    cInput,
    d,
    dInput,
    domain,
    fxExpr,
    fyExpr,
    selectedPreset,
    showEquilibria,
    showEigenSpaces,
    showField,
    showTrajectoryAnimation,
    stepSize,
    steps,
    vectorDensity,
  ]);

  return (
    <main className="min-h-screen bg-zinc-100 text-zinc-900">
      <div className="mx-auto grid w-full max-w-7xl gap-4 p-4 lg:grid-cols-[360px_1fr]">
        <section className="rounded-2xl bg-white p-4 shadow-sm ring-1 ring-zinc-200">
          <h1 className="text-2xl font-semibold tracking-tight">PPlane Modern</h1>
          <p className="mt-1 text-sm text-zinc-600">
            Interaktives Phasenporträt für 2D-Systeme. Klick im Plot startet eine animierte Trajektorie mit Live-Geschwindigkeitsvektor.
          </p>

          <div className="mt-4 grid gap-3">
            <label className="grid gap-1 text-sm">
              <span className="font-medium">Preset</span>
              <select
                className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 outline-none transition focus:border-zinc-400 focus:ring-2 focus:ring-zinc-900/10"
                onChange={(event) => applyPreset(event.target.value)}
                value={selectedPreset}
              >
                <option value={NO_PRESET}>Kein Preset (manuell)</option>
                {PRESETS.map((preset) => (
                  <option key={preset.name} value={preset.name}>
                    {preset.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-1 text-sm">
              <span className="font-medium">dx/dt = f(x, y)</span>
              <input
                value={fxExpr}
                onChange={(event) => {
                  setSelectedPreset(NO_PRESET);
                  setFxExpr(event.target.value);
                }}
                className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 outline-none transition focus:border-zinc-400 focus:ring-2 focus:ring-zinc-900/10"
                spellCheck={false}
              />
            </label>

            <label className="grid gap-1 text-sm">
              <span className="font-medium">dy/dt = g(x, y)</span>
              <input
                value={fyExpr}
                onChange={(event) => {
                  setSelectedPreset(NO_PRESET);
                  setFyExpr(event.target.value);
                }}
                className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 outline-none transition focus:border-zinc-400 focus:ring-2 focus:ring-zinc-900/10"
                spellCheck={false}
              />
            </label>

            <div className="grid grid-cols-2 gap-2">
              {[
                {
                  label: "a",
                  value: a,
                  inputValue: aInput,
                  setValue: setA,
                  setInputValue: setAInput,
                },
                {
                  label: "b",
                  value: b,
                  inputValue: bInput,
                  setValue: setB,
                  setInputValue: setBInput,
                },
                {
                  label: "c",
                  value: c,
                  inputValue: cInput,
                  setValue: setC,
                  setInputValue: setCInput,
                },
                {
                  label: "d",
                  value: d,
                  inputValue: dInput,
                  setValue: setD,
                  setInputValue: setDInput,
                },
              ].map((parameter) => (
                <label key={parameter.label} className="grid gap-1 text-sm">
                  <span className="font-medium">{parameter.label}</span>
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onPointerDown={() =>
                        startParamHold(
                          -0.1,
                          parameter.inputValue,
                          parameter.value,
                          parameter.setInputValue,
                          parameter.setValue,
                        )
                      }
                      onPointerUp={stopParamHold}
                      onPointerLeave={stopParamHold}
                      onPointerCancel={stopParamHold}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          adjustParamValue(
                            -0.1,
                            parameter.inputValue,
                            parameter.value,
                            parameter.setInputValue,
                            parameter.setValue,
                          );
                        }
                      }}
                      className="rounded-lg border border-zinc-300 bg-white px-2 py-2 text-sm leading-none text-zinc-700 transition hover:bg-zinc-50"
                      aria-label={`${parameter.label} verringern`}
                    >
                      −
                    </button>
                    <input
                      type="text"
                      inputMode="decimal"
                      value={parameter.inputValue}
                      onChange={(event) =>
                        handleParamChange(
                          event.target.value,
                          parameter.setInputValue,
                          parameter.setValue,
                        )
                      }
                      onBlur={() =>
                        handleParamBlur(
                          parameter.inputValue,
                          parameter.value,
                          parameter.setInputValue,
                        )
                      }
                      className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 outline-none transition focus:border-zinc-400 focus:ring-2 focus:ring-zinc-900/10"
                    />
                    <button
                      type="button"
                      onPointerDown={() =>
                        startParamHold(
                          0.1,
                          parameter.inputValue,
                          parameter.value,
                          parameter.setInputValue,
                          parameter.setValue,
                        )
                      }
                      onPointerUp={stopParamHold}
                      onPointerLeave={stopParamHold}
                      onPointerCancel={stopParamHold}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          adjustParamValue(
                            0.1,
                            parameter.inputValue,
                            parameter.value,
                            parameter.setInputValue,
                            parameter.setValue,
                          );
                        }
                      }}
                      className="rounded-lg border border-zinc-300 bg-white px-2 py-2 text-sm leading-none text-zinc-700 transition hover:bg-zinc-50"
                      aria-label={`${parameter.label} erhöhen`}
                    >
                      +
                    </button>
                  </div>
                </label>
              ))}
            </div>

            <label className="grid gap-1 text-sm">
              <span className="font-medium">Plotbereich ±{domain.toFixed(1)}</span>
              <input
                type="range"
                min="2"
                max="12"
                step="0.5"
                value={domain}
                onChange={(event) => {
                  setSelectedPreset(NO_PRESET);
                  setDomain(Number(event.target.value));
                }}
              />
            </label>

            <label className="grid gap-1 text-sm">
              <span className="font-medium">Vektordichte {vectorDensity}</span>
              <input
                type="range"
                min="12"
                max="35"
                step="1"
                value={vectorDensity}
                onChange={(event) => setVectorDensity(Number(event.target.value))}
              />
            </label>

            <label className="grid gap-1 text-sm">
              <span className="font-medium">Integrationsschritt {stepSize.toFixed(3)}</span>
              <input
                type="range"
                min="0.005"
                max="0.08"
                step="0.001"
                value={stepSize}
                onChange={(event) => setStepSize(Number(event.target.value))}
              />
            </label>

            <label className="grid gap-1 text-sm">
              <span className="font-medium">Integrationsschritte {steps}</span>
              <input
                type="range"
                min="300"
                max="2400"
                step="50"
                value={steps}
                onChange={(event) => setSteps(Number(event.target.value))}
              />
            </label>

            <div className="flex flex-wrap gap-3 text-sm">
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showField}
                  onChange={(event) => setShowField(event.target.checked)}
                />
                Richtungsfeld
              </label>
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showEquilibria}
                  onChange={(event) => setShowEquilibria(event.target.checked)}
                />
                Gleichgewichte
              </label>
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showEigenSpaces}
                  onChange={(event) => setShowEigenSpaces(event.target.checked)}
                />
                Eigenräume
              </label>
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showTrajectoryAnimation}
                  onChange={(event) => setShowTrajectoryAnimation(event.target.checked)}
                />
                Trajektorie animieren
              </label>
            </div>

            <button
              onClick={() => setTrajectories([])}
              className="rounded-lg bg-zinc-900 px-3 py-2 text-sm font-medium text-white transition hover:bg-zinc-700"
              type="button"
            >
              Trajektorien löschen
            </button>
          </div>

          <div className="mt-4 rounded-lg bg-zinc-50 p-3 text-xs text-zinc-700">
            <p>Variablen: x, y, a, b, c, d</p>
            <p>Funktionen: sin, cos, tan, exp, log, sqrt, abs, ...</p>
            {compiled.error && <p className="mt-2 text-rose-600">Parserfehler: {compiled.error}</p>}
          </div>

          <div className="mt-3 rounded-lg bg-zinc-50 p-3 text-xs text-zinc-700">
            <p className="font-semibold text-zinc-900">Gleichgewichtsanalyse</p>
            {!activeEquilibrium && (
              <p className="mt-1">Mit der Maus über einen Gleichgewichtspunkt fahren oder ihn anklicken.</p>
            )}
            {activeEquilibrium && !equilibriumAnalysis && (
              <p className="mt-1 text-rose-600">Analyse konnte für diesen Punkt nicht berechnet werden.</p>
            )}
            {activeEquilibrium && equilibriumAnalysis && (
              <div className="mt-1 space-y-1">
                <p>
                  Punkt: ({formatNumber(activeEquilibrium.x)}, {formatNumber(activeEquilibrium.y)})
                </p>
                <div>
                  <p>Jacobi-Matrix:</p>
                  <div className="mt-1 inline-grid grid-cols-[auto_1fr_auto] items-stretch gap-x-1 font-mono text-[11px] leading-5 text-zinc-800">
                    <div className="grid grid-rows-2 text-sm text-zinc-500">
                      <span>⎡</span>
                      <span>⎣</span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-3 rounded-md border border-zinc-200 bg-white px-2 py-1">
                      <span>{formatNumber(equilibriumAnalysis.jacobian.j11)}</span>
                      <span>{formatNumber(equilibriumAnalysis.jacobian.j12)}</span>
                      <span>{formatNumber(equilibriumAnalysis.jacobian.j21)}</span>
                      <span>{formatNumber(equilibriumAnalysis.jacobian.j22)}</span>
                    </div>
                    <div className="grid grid-rows-2 text-sm text-zinc-500">
                      <span>⎤</span>
                      <span>⎦</span>
                    </div>
                  </div>
                </div>
                <p>
                  λ₁ = {equilibriumAnalysis.eigenvalue1}, λ₂ = {equilibriumAnalysis.eigenvalue2}
                </p>
                <p className="font-semibold text-zinc-900">Typ: {equilibriumAnalysis.classification}</p>
              </div>
            )}
          </div>
        </section>

        <section className="rounded-2xl bg-white p-4 shadow-sm ring-1 ring-zinc-200">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-lg font-semibold">Phasenebene</h2>
            <div className="flex items-center gap-3">
              <span className="text-xs text-zinc-500">
                Hover/Klick auf Gleichgewicht = Analyse, sonst Klick = animierte Trajektorie
              </span>
              <button
                type="button"
                onClick={handleSavePng}
                className="rounded-lg bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-zinc-700"
              >
                Als PNG speichern
              </button>
            </div>
          </div>
          <div className="overflow-hidden rounded-xl border border-zinc-200">
            <canvas ref={canvasRef} width={960} height={960} className="h-auto w-full bg-white" />
          </div>

          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            <div className="rounded-xl border border-zinc-200 bg-white p-2">
              <p className="mb-2 text-xs text-zinc-600">Zeitverlauf x(t)</p>
              <canvas ref={xTimeCanvasRef} width={560} height={200} className="h-auto w-full" />
            </div>
            <div className="rounded-xl border border-zinc-200 bg-white p-2">
              <p className="mb-2 text-xs text-zinc-600">Zeitverlauf y(t)</p>
              <canvas ref={yTimeCanvasRef} width={560} height={200} className="h-auto w-full" />
            </div>
          </div>
          <p className="mt-2 text-xs text-zinc-500">
            Zeitplots basieren auf der zuletzt gesetzten Trajektorie im Hauptplot.
          </p>
          <p className="mt-1 text-xs text-zinc-500">
            Bei aktiver Animation sind maximal {MAX_SIMULTANEOUS_ANIMATIONS} gleichzeitig laufende Trajektorien erlaubt.
          </p>
        </section>
      </div>
    </main>
  );
}
