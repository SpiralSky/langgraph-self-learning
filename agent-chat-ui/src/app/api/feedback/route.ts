import { NextRequest, NextResponse } from "next/server";

// Passthrough proxy for the separate feedback FastAPI (POST /feedback on the
// feedback service), mirroring the `api/[..._path]` proxy pattern. This is NOT
// a langgraph `Client` route — plain fetch + JSON passthrough with the same
// CORS headers as the SDK proxy.
//
// The target is the BASE URL of the feedback service (`/feedback` is appended);
// override with NEXT_PUBLIC_FEEDBACK_API_URL, default `http://localhost:8000`.
const FEEDBACK_API_URL =
  process.env.NEXT_PUBLIC_FEEDBACK_API_URL ?? "http://localhost:8000";

function getCorsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Expose-Headers": "content-location",
  };
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = await req.text();
    const res = await fetch(`${FEEDBACK_API_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    const responseHeaders: Record<string, string> = {};
    res.headers.forEach((value, key) => {
      responseHeaders[key] = value;
    });
    return new NextResponse(res.body, {
      status: res.status,
      statusText: res.statusText,
      headers: { ...responseHeaders, ...getCorsHeaders() },
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 502, headers: getCorsHeaders() },
    );
  }
}

export async function OPTIONS(): Promise<NextResponse> {
  return new NextResponse(null, { status: 204, headers: getCorsHeaders() });
}