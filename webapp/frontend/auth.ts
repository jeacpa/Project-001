import NextAuth from 'next-auth';
import Google from 'next-auth/providers/google';
import Github from 'next-auth/providers/github';
import Credentials from 'next-auth/providers/credentials';
import Discord from 'next-auth/providers/discord';
import type { Provider } from 'next-auth/providers';


const providers: Provider[] = [
  Google({
    clientId: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
  }),

  Github({
    clientId: process.env.GITHUB_CLIENT_ID,
    clientSecret: process.env.GITHUB_CLIENT_SECRET,
  }),
  Credentials({
    credentials: {
      email: { label: 'Email Address', type: 'email' },
      password: { label: 'Password', type: 'password' },
    },
    authorize(c) {
      if (c.password !== 'password') {
        return null;
      }
      return {
        id: 'test',
        name: 'Test User',
        email: String(c.email),
      };
    },
  }),

  Discord({
    clientId: process.env.DISCORD_CLIENT_ID,
    clientSecret: process.env.DISCORD_CLIENT_SECRET,
  }),
];

if (!process.env.GOOGLE_CLIENT_ID) {
  console.warn('Missing environment variable "GOOGLE_CLIENT_ID"');
}
if (!process.env.GOOGLE_CLIENT_SECRET) {
  console.warn('Missing environment variable "GOOGLE_CLIENT_SECRET"');
}
if (!process.env.GITHUB_CLIENT_ID) {
  console.warn('Missing environment variable "GITHUB_CLIENT_ID"');
}
if (!process.env.GITHUB_CLIENT_SECRET) {
  console.warn('Missing environment variable "GITHUB_CLIENT_SECRET"');
}
if (!process.env.DISCORD_CLIENT_ID) {
  console.warn('Missing environment variable "DISCORD_CLIENT_ID"');
}
if (!process.env.DISCORD_CLIENT_SECRET) {
  console.warn('Missing environment variable "DISCORD_CLIENT_SECRET"');
}


export const providerMap = providers.map((provider) => {
  if (typeof provider === 'function') {
    const providerData = provider();
    return { id: providerData.id, name: providerData.name };
  }
  return { id: provider.id, name: provider.name };
});

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers,



  secret: process.env.AUTH_SECRET,
  pages: {
    signIn: '/auth/signin',
    error: '/errorPage'
  },
  callbacks: {
    // authorized({ auth: session, request: { nextUrl } }) {
    authorized() {
      // disable auth for now
      return true;
      // const isLoggedIn = !!session?.user;
      // const isPublicPage = nextUrl.pathname.startsWith('/public');

      // if (isPublicPage || isLoggedIn) {
      //   return true;
      // }

      // return false; // Redirect unauthenticated users to login page
    },
    // eslint-disable-next-line unused-imports/no-unused-vars
    async signIn() {
      // Return false if not in whitelist (will go to custom error page)
      return true;
    },
  },
});
