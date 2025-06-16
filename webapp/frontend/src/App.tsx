import React from 'react';
import logo from './logo.svg';
import './App.css';
import { Button, Container, Typography } from '@mui/material';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload!
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
      <Container maxWidth="sm" style={{ marginTop: '2rem' }}>
        <Typography variant="h4" gutterBottom>
          Hello Material UI with TypeScript!
        </Typography>
        <Button variant="contained" color="primary">
          Click Me
        </Button>
      </Container>
    </div>
  );
}

export default App;
