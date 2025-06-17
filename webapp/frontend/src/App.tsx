import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import { Button, Container, Typography } from '@mui/material';

function App() {
  const [data, setData] = useState<{ message: string } | null>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/data')
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch((err) => console.error('Error fetching data:', err));
  }, []);

  return (

    <Container maxWidth="sm" style={{ marginTop: '2rem' }}>
      <Typography variant="h4" gutterBottom>
        Hello Material UI with TypeScript!
      </Typography>
      <p>{data ? data.message : 'Loading...'}</p>
      <Button variant="contained" color="primary">
        Click Me
      </Button>
    </Container>
  );
}

export default App;
