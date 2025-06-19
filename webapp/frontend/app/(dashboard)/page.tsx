"use client";

import * as React from 'react';
import Typography from '@mui/material/Typography';
import { PageContainer } from '@toolpad/core/PageContainer';
import { auth } from '../../auth';
import { useEffect } from 'react';

export default  function HomePage() {
  // const session = await auth();

  useEffect(() => {
    fetch("http://localhost:8000/")
     .then((response) => {
        if (!response.ok) throw new Error("Failed to fetch");
        return response.json();
      })
      .then((json) => console.log(json))
      .catch((error) => console.error("Error fetching data:", error));
  }, []);
  return (    
    <PageContainer breadcrumbs={[]}>
      <Typography>
        Live video here
      </Typography>
      <img src="http://localhost:8000/video" width="1920" height="1080"/>
    </PageContainer>
  );
}
