export type ServerState = {
    showBoxes: boolean;
    showLight: boolean;
    showText: boolean;
    showZones: boolean;
    paused: boolean;
}

export type ServerControlResponse = {
    state: ServerState;
}
