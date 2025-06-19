
export { auth as middleware } from './auth';

export const config = {
  // https://nextjs.org/docs/app/building-your-application/routing/middleware#matcher
  // NOTE: Need to have errorPage in here so that if something fails during login and a session is not
  //       created the default behaviour of redirect to login won't happen
  matcher: ['/((?!api|_next/static|_next/image|.*\.png$|errorPage).*)'],
};
