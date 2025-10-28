export function shallowEqual<T extends object>(a?: T, b?: T): boolean {
    if (a === b) return true;
    if (a === undefined || b === undefined) return false;

    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);

    if (aKeys.length !== bKeys.length) return false;

    for (const key of aKeys) {
        if ((a as Record<string, unknown>)[key] !== (b as Record<string, unknown>)[key]) return false;
    }

    return true;
}

export function deepEqual<T>(obj1: T, obj2: T): boolean {
    if (obj1 === obj2) return true;

    if (
        typeof obj1 !== "object" || obj1 === null ||
        typeof obj2 !== "object" || obj2 === null
    ) {
        return false;
    }

    const keys1 = Object.keys(obj1 as Record<string, unknown>);
    const keys2 = Object.keys(obj2 as Record<string, unknown>);

    if (keys1.length !== keys2.length) return false;

    for (const key of keys1) {
        const val1 = (obj1 as Record<string, unknown>)[key];
        const val2 = (obj2 as Record<string, unknown>)[key];

        if (!keys2.includes(key) || !deepEqual(val1, val2)) {
            return false;
        }
    }

    return true;
}


export function drawClosedShape(ctx: CanvasRenderingContext2D, points: number[][]): void {
    if (points.length === 0) 
        return;

    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        const [x, y] = points[i];
        ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
}

export function distance(p1: number[], p2: number[]) {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  return Math.hypot(dx, dy); // same as Math.sqrt(dx*dx + dy*dy)
}


/**
 * Tests whether a polygon (given as an array of [x, y] points) is convex.
 * Works for CW or CCW vertex order and tolerates an optional repeated first point.
 */
export function isConvexPolygon(
  points: number[][],
  options: { strict?: boolean; allowCollinearAll?: boolean } = {}
): boolean {
  const { strict = true, allowCollinearAll = false } = options;

  if (points.length < 3) return false;

  // Remove closing duplicate point if present
  const n0 = points.length;
  const [fx, fy] = points[0];
  const [lx, ly] = points[n0 - 1];
  const same = fx === lx && fy === ly;
  const pts = same ? points.slice(0, -1) : points.slice();
  const n = pts.length;
  if (n < 3) return false;

  const cross = (a: number[], b: number[], c: number[]): number => {
    const abx = b[0] - a[0];
    const aby = b[1] - a[1];
    const bcx = c[0] - b[0];
    const bcy = c[1] - b[1];
    return abx * bcy - aby * bcx;
  };

  let sign = 0;
  let allZero = true;

  for (let i = 0; i < n; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % n];
    const c = pts[(i + 2) % n];
    const z = cross(a, b, c);

    if (z !== 0) {
      allZero = false;
      const s = z > 0 ? 1 : -1;
      if (sign === 0) sign = s;
      else if (s !== sign) return false;
    } else if (strict) {
      return false; // disallow collinear triples if strict
    }
  }

  if (allZero) return allowCollinearAll;
  return true;
}
