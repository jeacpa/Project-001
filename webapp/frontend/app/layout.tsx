import * as React from 'react';
import { NextAppProvider } from '@toolpad/core/nextjs';
import { AppRouterCacheProvider } from '@mui/material-nextjs/v15-appRouter';

import type { Navigation } from '@toolpad/core/AppProvider';
import { SessionProvider, signIn, signOut } from 'next-auth/react';
import { auth } from '../auth';
import theme from '../theme';
import VideocamIcon from '@mui/icons-material/Videocam';
import SearchIcon from '@mui/icons-material/Search';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import NewspaperIcon from '@mui/icons-material/Newspaper';
import ErrorIcon from '@mui/icons-material/Error';
import SettingsIcon from '@mui/icons-material/Settings';
import InfoIcon from '@mui/icons-material/Info';

const AUTH_DISABLED =
  process.env.NEXT_PUBLIC_DISABLE_AUTH === 'true';

  const FAKE_SESSION = {
  user: {
    name: 'Dev User',
    email: 'dev@local',
    image: null,
  },
  expires: '2099-01-01T00:00:00.000Z',
};

export const metadata = {
  title: 'AI Controlled Traffic Signals',
  description: 'This is a sample app built with Toolpad Core and Next.js',
};

const NAVIGATION: Navigation = [
  {
    segment: '',
    title: 'Live Video',
    icon: <VideocamIcon />,
  },
  {
    segment: 'search',
    title: 'Search',
    icon: <SearchIcon />,
  },
  {
    segment: 'analytics',
    title: 'Analytics',
    icon: <AnalyticsIcon />,
  },
  {
    segment: 'news',
    title: 'News',
    icon: <NewspaperIcon />,
  },
  {
    segment: 'issues',
    title: 'Issues',
    icon: <ErrorIcon />,
  },
  {
    segment: 'options',
    title: 'Options',
    icon: <SettingsIcon />,
  },
  {
    segment: 'about',
    title: 'About',
    icon: <InfoIcon />,
  }
];

import Image from 'next/image';

const BRANDING = {
  title: 'AI Controlled Traffic Signals',
  logo: <Image src="/logo_dark.png" alt="Logo" width={230} height={58} />,
};


const AUTHENTICATION = {
  signIn,
  signOut,
};


export default async function RootLayout(props: { children: React.ReactNode }) {
  const session = AUTH_DISABLED ? FAKE_SESSION : await auth();

  console.log("Auth Disabled:", AUTH_DISABLED);
  return (
    <html lang="en" data-toolpad-color-scheme="light" suppressHydrationWarning>
      <head>
        <link
          rel="icon"
          href="/icon.png"
          type="image/png"
          sizes="64x64"
        />
      </head>
      <body>
        <SessionProvider session={session}>
          <AppRouterCacheProvider options={{ enableCssLayer: true }}>

            <NextAppProvider
              navigation={NAVIGATION}
              branding={BRANDING}
              session={session}
              authentication={AUTH_DISABLED ? undefined : AUTHENTICATION}
              theme={theme}
            >
              {props.children}
            </NextAppProvider>

          </AppRouterCacheProvider>
        </SessionProvider>
      </body>
    </html>
  );
}
