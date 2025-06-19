import * as React from 'react';
import Typography from '@mui/material/Typography';
import { PageContainer } from '@toolpad/core/PageContainer';
import { auth } from '../../auth';

export default async function HomePage() {
  const session = await auth();

  return (    
    <PageContainer breadcrumbs={[]}>
      <Typography>
        Live video here
      </Typography>
    </PageContainer>
  );
}
