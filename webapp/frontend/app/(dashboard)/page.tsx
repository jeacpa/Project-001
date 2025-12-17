"use client";

import * as React from 'react';
import { PageContainer } from '@toolpad/core/PageContainer';
import { Paper } from '@mui/material';
import VideoToolbar from '../components/VideoToolbar';
import Image from 'next/image';
import useWS from '../hooks/useWS';
import ZoneResizeDialog from '../dialogs/ZoneResizeDialog';
import { ServerState } from '../structures';

interface State {
  zoneResizeOpen: boolean;
  state?: ServerState;
}
export default function HomePage() {
  const [state, setState] = React.useState<State>({ zoneResizeOpen: false });

  // const [zoneResizeOpen, setZoneResizeOpen] = React.useState(false);

  const ws = useWS();
  const handleMouseMove = (e: React.MouseEvent<HTMLImageElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * process.env.NEXT_PUBLIC_VIDEO_WIDTH; //x position within the element.
    const y = ((e.clientY - rect.top) / rect.height) * process.env.NEXT_PUBLIC_VIDEO_HEIGHT; //y position within the element.

    ws.sendMessage({ action: "cursor_pos", x: Math.round(x), y: Math.round(y) });
  };

  const handleMouseClick = () => {
    ws.sendMessage({ action: "select_box" });
  };


  return (
    <PageContainer breadcrumbs={[]}>
      <Paper
        elevation={1}
        sx={{ p: '10px' }}
      >

        <VideoToolbar onZoneResize={(state?: ServerState) => {

          setState({ zoneResizeOpen: true, state })
        }} />
        <Image
          id="videoFeed"
          src={process.env.NEXT_PUBLIC_VIDEO_URL ?? ''}
          alt="Video Stream"
          width={1920}
          height={1080}
          style={{ width: '100%', height: 'auto', display: 'block' }}
          unoptimized
          onMouseMove={handleMouseMove}
          onClick={handleMouseClick}
        />
      </Paper >
      <ZoneResizeDialog
        open={state.zoneResizeOpen}
        zone={state.state?.countZone}
        onClose={() => setState({ zoneResizeOpen: false })}
        onSetZone={(zone?: number[][]) => {
          ws.sendMessage({ action: "set_zone", countZone: zone });
          setState({ zoneResizeOpen: false });
        }}

      />

    </PageContainer>
  );
}

