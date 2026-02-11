
// export { auth as middleware } from './auth';

import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { auth } from "./auth";

// Define the middleware function signature Next.js expects:
type MiddlewareFn = (req: NextRequest) => Response | void | Promise<Response | void>;

// Cast NextAuth's `auth` to that signature once:
const nextAuthMiddleware = auth as unknown as MiddlewareFn;

export default function middleware(req: NextRequest) {
  if (process.env.DISABLE_AUTH === "true") {
    return NextResponse.next();
  }

  return nextAuthMiddleware(req);
}

export const config = {
  // https://nextjs.org/docs/app/building-your-application/routing/middleware#matcher
  // NOTE: Need to have errorPage in here so that if something fails during login and a session is not
  //       created the default behaviour of redirect to login won't happen
  matcher: ['/((?!api|_next/static|_next/image|.*.png$|errorPage).*)'],
};
