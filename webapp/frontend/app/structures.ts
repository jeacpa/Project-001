export type ServerState = {
    showBoxes: boolean;
    showLight: boolean;
    showText: boolean;
    showZones: boolean;
    paused: boolean;
    countZone: number[][];
}

export type ServerControlResponse = {
    state: ServerState;
}
