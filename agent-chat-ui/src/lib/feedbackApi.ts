/**
 * Client-side helper for the feedback UI.
 *
 * POSTs to the same-origin Next.js passthrough route `/api/feedback` (i.e.
 * `${origin}/api/feedback`), which forwards the request server-side to the
 * separate feedback FastAPI — see `src/app/api/feedback/route.ts`.
 */

export type FeedbackRequestPayload = {
  question: string;
  responses: string[];
  comment: string;
  /** Optional graph state snapshot (excludes heavy message/ui channels). */
  graph_state?: Record<string, unknown> | null;
};

export type FeedbackAppliedPatch = {
  seq: number;
  action: "add" | "update" | "remove";
  title?: string | null;
  point_key?: string | null;
  content?: string | null;
  node_id?: string | null;
  node?: Record<string, unknown> | null;
  [key: string]: unknown;
};

/**
 * Submit feedback and return the `{"applied": [...]}` patch list.
 *
 * The feedback server computes structured behavior patches from the
 * suggestion via the LLM; the response lists the patches that were applied.
 */
export async function sendFeedback(
  payload: FeedbackRequestPayload,
): Promise<FeedbackAppliedPatch[]> {
  const res = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let detail = `POST /api/feedback -> ${res.status}`;
    try {
      const body = (await res.json()) as {
        detail?: string;
        error?: string;
      };
      detail = body?.detail ?? body?.error ?? detail;
    } catch {
      // keep the status-based message
    }
    throw new Error(detail);
  }

  const data = (await res.json()) as { applied?: FeedbackAppliedPatch[] };
  return data.applied ?? [];
}