"use client";

import * as React from 'react';
import { PageContainer } from '@toolpad/core/PageContainer';
import { Paper } from '@mui/material';
import VideoToolbar from '../components/VideoToolbar';
import Image from 'next/image';
import useWS from '../hooks/useWS';

export default function HomePage() {
  const ws = useWS();
  const handleMouseMove = (e: React.MouseEvent<HTMLImageElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 1920; //x position within the element.
    const y = ((e.clientY - rect.top) / rect.height) * 1080; //y position within the element.

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

        <VideoToolbar />
        <Image
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
    </PageContainer>
  );
}

