"use client";

import { useSearchParams } from 'next/navigation';

export default function AuthErrorPage() {
  const searchParams = useSearchParams();
  const error = searchParams.get('error');

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>Access Denied</h1>
      {error === 'AccessDenied' ? (
        <p>Your account is not on the approved user list.</p>
      ) : (
        <p>An unknown error occurred: {error}</p>
      )}
      <a href="/api/auth/signin">Try signing in again</a>
    </div>
  );
}
