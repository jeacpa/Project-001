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
