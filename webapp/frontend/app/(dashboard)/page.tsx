"use client";

import * as React from 'react';
import { PageContainer } from '@toolpad/core/PageContainer';
import { Paper } from '@mui/material';
import VideoToolbar from '../components/VideoToolbar';
import Image from 'next/image';

export default function HomePage() {

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
        />
        {/* <img src={process.env.NEXT_PUBLIC_VIDEO_URL}
          alt='Video Stream'
          // width="1920" 
          // height="1080" 
          style={{ width: '100%', height: 'auto', display: 'block' }} /> */}
      </Paper >
    </PageContainer>
  );
}

