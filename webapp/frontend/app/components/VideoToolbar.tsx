"use client";

import { Divider, Paper, styled, ToggleButtonGroup, toggleButtonGroupClasses } from "@mui/material";
import React, { useCallback } from "react";
import { ServerControlResponse, ServerState } from "../structures";
import CropPortraitIcon from '@mui/icons-material/CropPortrait';
import TrafficIcon from '@mui/icons-material/Traffic';
import StatsIcon from '@mui/icons-material/QueryStats';
import BoxesIcon from '@mui/icons-material/DirectionsCar';
import PauseIcon from '@mui/icons-material/Pause';
import PlayIcon from '@mui/icons-material/PlayArrow';
import FastForwardIcon from "@mui/icons-material/FastForward";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import RestartIcon from "@mui/icons-material/RestartAlt";
import InfoJumpIcon from "@mui/icons-material/ErrorOutline";
import SimpleToggleButton from "../simpleWrappers/ToggleButton";
import useWS from "../hooks/useWS";
import { shallowEqual } from "../util";

interface VideoToolbarProps {
    onRestart?: () => void;
}

export default function VideoToolbar({ onRestart }: VideoToolbarProps) {

    // const ws = React.useRef<WebSocket | null>(null);
    const [serverState, setServerState] = React.useState<ServerState | undefined>();
    const ws = useWS((data) => {

            // const msg = JSON.parse(data) as ServerControlResponse;
            // Update the server state with the received message
            // setServerState(msg.state);
            if (shallowEqual((data as ServerControlResponse).state, serverState)) 
                return;

            setServerState((data as ServerControlResponse).state);
    }, () => {
        // Send an inital blank action to get the initial state
        return { action: "" };
    });

    const sendAction = useCallback((action: string) => {
        ws.sendMessage({ action });

        // if (ws.current?.readyState === WebSocket.OPEN) {
        //     ws.current.send(JSON.stringify({ action }));
        // }
    }, [ws]);

    // React.useEffect(() => {
    //     const socket = new WebSocket(process.env.NEXT_PUBLIC_CONTROL_URL!);
    //     ws.current = socket;

    //     socket.onopen = () => {
    //         // Send an inital blank action to get the initial state
    //         sendAction("");

    //     };
    //     socket.onmessage = (event) => {
    //         const msg = JSON.parse(event.data) as ServerControlResponse;
    //         // Update the server state with the received message
    //         setServerState(msg.state);
    //     };
    //     socket.onclose = () => {
    //         console.log("WebSocket disconnected");
    //     };

    //     return () => socket.close();
    // }, [sendAction]);


    const StyledToggleButtonGroup = styled(ToggleButtonGroup)(({ theme }) => ({
        [`& .${toggleButtonGroupClasses.grouped}`]: {
            margin: theme.spacing(0.5),
            border: 0,
            borderRadius: theme.shape.borderRadius,
            [`&.${toggleButtonGroupClasses.disabled}`]: {
                border: 0,
            },
        },
        [`& .${toggleButtonGroupClasses.middleButton},& .${toggleButtonGroupClasses.lastButton}`]:
        {
            marginLeft: -1,
            borderLeft: '1px solid transparent',
        },
    }));

    return (
        <Paper
            elevation={0}
            sx={(theme) => ({
                display: 'flex',
                border: `1px solid ${theme.palette.divider}`,
                flexWrap: 'wrap',
                mb: '10px'
            })}
        >
            <StyledToggleButtonGroup
                size="small"
                aria-label="text alignment"
            >
                <SimpleToggleButton
                    selected={!!serverState?.showZones}
                    onChange={() => sendAction("toggle_zones")}
                    icon={<CropPortraitIcon />}
                    tooltip={serverState?.showZones ? "Hide zone" : "Show zone"}
                />
                <SimpleToggleButton
                    selected={!!serverState?.showLight}
                    onChange={() => sendAction("toggle_light")}
                    icon={<TrafficIcon />}
                    tooltip={serverState?.showLight ? "Hide traffic light" : "Show traffic light"}
                />
                <SimpleToggleButton
                    selected={!!serverState?.showText}
                    onChange={() => sendAction("toggle_text")}
                    icon={<StatsIcon />}
                    tooltip={serverState?.showText ? "Hide stats" : "Show stats"}
                />
                <SimpleToggleButton
                    selected={!!serverState?.showBoxes}
                    onChange={() => sendAction("toggle_boxes")}
                    icon={<BoxesIcon />}
                    tooltip={serverState?.showBoxes ? "Hide boxes" : "Show boxes"}
                />
            </StyledToggleButtonGroup>
            <Divider flexItem orientation="vertical" sx={{ mx: 0.5, my: 1 }} />
            <StyledToggleButtonGroup
                size="small"
                aria-label="text alignment"
            >
                <SimpleToggleButton
                    onChange={() => sendAction("rewind")}
                    icon={<FastRewindIcon />}
                    tooltip={"Rewind 1 second"}
                />
                <SimpleToggleButton
                    onChange={() => sendAction("toggle_pause")}
                    icon={serverState?.paused ? <PlayIcon /> : <PauseIcon />}
                    tooltip={serverState?.paused ? "Play" : "Pause"}
                />
                <SimpleToggleButton
                    onChange={() => sendAction("fast_forward")}
                    icon={<FastForwardIcon />}
                    tooltip={"Fast forward 1 second"}
                />
            </StyledToggleButtonGroup>
            <Divider flexItem orientation="vertical" sx={{ mx: 0.5, my: 1 }} />
            <StyledToggleButtonGroup
                size="small"
                aria-label="text alignment"
            >
                <SimpleToggleButton
                    onChange={() => { 
                        sendAction("infojump");
                        onRestart?.();
                    }}

                    icon={<InfoJumpIcon />}
                    tooltip={"Jump to traffic suggestion scenario"}
                />
                <SimpleToggleButton
                    onChange={() => { 
                        sendAction("restart");
                        onRestart?.();
                    }}

                    icon={<RestartIcon />}
                    tooltip={"Restart video"}
                />
            </StyledToggleButtonGroup>
        </Paper>
    )
}