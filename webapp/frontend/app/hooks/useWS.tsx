import React, { useCallback } from "react";

// Simple websocket hook that takes an optional callback for when
// a message is received and when the connection is opened
// Takes care of setting up and tearing down the connection as well as serializing/deserializing messages
// Current protocol is simply JSON strings
export default function useWS(onMessage?: (data: unknown) => void, onOpen?: () => unknown) {
    const ws = React.useRef<WebSocket | null>(null);

    const sendMessage = useCallback((message: unknown) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(message));
        }
    }, [ws]);


    React.useEffect(() => {
        const socket = new WebSocket(process.env.NEXT_PUBLIC_CONTROL_URL!);
        ws.current = socket;

        socket.onopen = () => {
            const res = onOpen?.();

            if (res !== undefined) { 
                sendMessage(res); 
            }

        };
        socket.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            onMessage?.(msg);
        };
        socket.onclose = () => {
            console.log("WebSocket disconnected");
        };

        return () => socket.close();
    }, [sendMessage, onMessage, onOpen]);

    return { sendMessage };
}