"use client";

import * as React from "react";
import { Button, Modal, Backdrop, Box, Typography, Stack } from "@mui/material";
import { distance, drawClosedShape, isConvexPolygon } from "../util";

interface ZoneResizeDialogProps {
    open: boolean;
    onClose?: () => void;
    onSetZone?: (zone?: number[][]) => void;
    zone?: number[][];
}

export default function ZoneResizeDialog({ open, onClose, onSetZone, zone }: ZoneResizeDialogProps) {
    const currentZone = React.useRef<number[][] | undefined>(zone);
    const [hoverPoint, setHoverPoint] = React.useState<number | null>(null);
    const [dragAnchor, setDragAnchor] = React.useState<number[] | null>(null);
    const [dragDelta, setDragDelta] = React.useState<number[] | null>(null);
    currentZone.current = zone;

    const canvasRef = React.useRef<HTMLCanvasElement>(null);

    let feedX = 0;
    let feedY = 0;
    let feedWidth = 0;
    let feedHeight = 0;
    if (typeof document !== "undefined") {
        const el = document.getElementById('videoFeed');
        if (el) {
            const rect = el.getBoundingClientRect();

            feedX = (rect.x);
            feedY = (rect.y);
            feedWidth = rect.width;
            feedHeight = rect.height;
        }
    }
    const getPosition = React.useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
        if (!canvasRef.current)
            return [0, 0];

        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        return [x, y];

    }, []);

    const drawZone = React.useCallback(() => {
        if (!canvasRef.current || !currentZone.current)
            return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;

        const cssW = canvasRef.current.clientWidth;
        const cssH = canvasRef.current.clientHeight;
        const dpr = window.devicePixelRatio || 1;

        canvas.width = Math.max(1, Math.round(cssW * dpr));
        canvas.height = Math.max(1, Math.round(cssH * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // ctx?.clearRect(0, 0, cssW, cssH);
        ctx.strokeStyle = 'yellow';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'yellow';

        const xf = canvasRef.current.clientWidth / 1920;
        const yf = canvasRef.current.clientHeight / 1080;

        const scaledZone = currentZone.current.map(pt => [pt[0] * xf, pt[1] * yf]);

        // If we are dragging a point, apply delta
        if (dragDelta !== null && dragAnchor !== null && hoverPoint !== null) {
            const [dx, dy] = dragDelta;
            scaledZone[hoverPoint] = [
                Math.max(0, Math.min(scaledZone[hoverPoint][0] + dx, canvasRef.current.clientWidth)),
                Math.max(0, Math.min(scaledZone[hoverPoint][1] + dy, canvasRef.current.clientHeight)),
            ];
        }

        drawClosedShape(ctx, scaledZone);

        const w = 11;
        const h = 11;
        const r = 3;

        let idx = 0
        for (const pt of scaledZone) {
            const [x, y] = pt;

            if (idx === hoverPoint) {
                ctx.fillStyle = 'blue';
                ctx.strokeStyle = 'blue';
            } else {
                ctx.fillStyle = 'yellow';
                ctx.strokeStyle = 'yellow';
            }

            ctx.beginPath();
            ctx.roundRect(x - w / 2, y - h / 2, w, h, r);
            ctx.fill();
            ctx.stroke();

            idx++;
        }

    }, [hoverPoint, dragDelta, dragAnchor]);

    const handleHover = React.useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
        if (!canvasRef.current || !currentZone.current)
            return;

        const [x, y] = getPosition(e);

        if (dragAnchor !== null && hoverPoint !== null) {

            // we are being dragged, calulate delta
            const deltaX = x - dragAnchor[0];
            const deltaY = y - dragAnchor[1];

            setDragDelta([deltaX, deltaY]);

        } else {

            const xf = canvasRef.current.clientWidth / 1920;
            const yf = canvasRef.current.clientHeight / 1080;

            const scaledZone = currentZone.current.map(pt => [pt[0] * xf, pt[1] * yf]);

            let idx = 0;
            for (const pt of scaledZone) {
                if (distance(pt, [x, y]) <= 10) {
                    setHoverPoint(idx);
                    return;
                }
                idx++;
            }
            if (hoverPoint !== null) {
                setHoverPoint(null);
            }
        }

    }, [hoverPoint, getPosition, dragAnchor]);

    const handleDown = React.useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
        if (!canvasRef.current)
            return;

        canvasRef.current.setPointerCapture(e.pointerId);

        if (hoverPoint === null)
            return;

        setDragAnchor(getPosition(e));

    }, [hoverPoint, getPosition]);

    const handleUp = React.useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
        if (!canvasRef.current || !currentZone.current)
            return;

        canvasRef.current.releasePointerCapture(e.pointerId);

        // If we were dragging, apply the delta to the zone
        if (dragDelta !== null && dragAnchor !== null && hoverPoint !== null) {
            const [dx, dy] = dragDelta;

            currentZone.current[hoverPoint] = [
                Math.min(1920, Math.max(0, Math.round(currentZone.current[hoverPoint][0] + dx * (1920 / canvasRef.current.clientWidth)))),
                Math.min(1080, Math.max(0, Math.round(currentZone.current[hoverPoint][1] + dy * (1080 / canvasRef.current.clientHeight)))),
            ];
        }

        setDragAnchor(null);
        setDragDelta(null);

    }, [dragDelta, dragAnchor, hoverPoint]);

    React.useEffect(() => {
        drawZone();
    }, [drawZone, zone]);

    const isConvex = currentZone.current ? isConvexPolygon(currentZone.current) : false;

    return (<Modal
        open={open}
        onClose={() => onClose?.()}
        // Backdrop already uses theme.zIndex.modal (1300) which is above AppBar (1100)
        closeAfterTransition
        slots={{ backdrop: Backdrop }}
        slotProps={{ backdrop: { timeout: 200 } }}
        // Ensure portal goes to <body> so it isn't clipped by any ancestor
        container={typeof document !== "undefined" ? document.body : undefined}
        keepMounted
    >
        <Box
            sx={{
                position: "fixed",
                inset: 0,               // top:0 right:0 bottom:0 left:0
                // width: "100vw",
                // height: "100vh",
                bgcolor: "rgba(0,0,0,0.15)", // dim background
                display: "grid",
                placeItems: "start center",
                p: 2,
            }}
        >
            <Box
                sx={{
                    bgcolor: "background.paper",
                    borderRadius: 2,
                    boxShadow: 24,
                    p: 3,
                    maxWidth: 520,
                    width: "100%",
                }}
            >
                <Typography variant="h6" gutterBottom>
                    Change automobile detection zone
                </Typography>
                <Typography variant="body2" gutterBottom>
                    Drag the corners of the detection zone then click Accept.
                </Typography>
                {
                    !isConvex &&
                    <Typography variant="body2" color="error" gutterBottom>
                        The zone must be convex, please adjust the points.
                    </Typography>
                }
                <Stack direction="row" gap={1} justifyContent="flex-end" mt={2}>
                    <Button onClick={() => onClose?.()}>Cancel</Button>
                    <Button disabled={!isConvex} onClick={() => {
                        if (!currentZone.current)
                            return;
                        onSetZone?.(currentZone.current)}
                    }>Accept</Button>
                </Stack>
            </Box>
            <Box sx={{
                position: 'absolute',
                top: feedY,
                left: feedX,
                width: feedWidth,
                height: feedHeight,
                border: '2px dashed white',
            }} >
                <canvas
                    ref={canvasRef}
                    style={{
                        width: '100%',
                        height: '100%',
                        display: 'block',
                        background: 'transparent',
                    }}
                    aria-label="Image overlay"
                    onPointerMove={handleHover}
                    onPointerDown={handleDown}
                    onPointerUp={handleUp}
                />
            </Box>
        </Box>
    </Modal>
    )
}