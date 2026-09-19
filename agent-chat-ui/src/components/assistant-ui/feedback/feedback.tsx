"use client";

import { useLangGraphState } from "@assistant-ui/react-langgraph";
import { MessageSquareQuote, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type {
  FeedbackAppliedPatch,
  FeedbackRequestPayload,
} from "@/lib/feedbackApi";
import { sendFeedback } from "@/lib/feedbackApi";

/**
 * Feedback affordance attached to each assistant message.
 *
 * Trigger choice (recorded in shared.md): assistant-message affordance — the
 * task-05 inbox never shipped a "Share feedback..." input, so the message
 * chip is the single trigger. It opens a small form with the
 * `FeedbackRequest` fields (question / responses / comment) and POSTs to the
 * same-origin `/api/feedback` passthrough, which forwards to the separate
 * feedback FastAPI (`NEXT_PUBLIC_FEEDBACK_API_URL`). The returned
 * `{"applied": [...]}` patch list is surfaced via toast (count + titles).
 */

function patchLabel(patch: FeedbackAppliedPatch): string {
  return patch.title ?? (patch.node_id ? `node:${patch.node_id}` : "patch");
}

function AppliedSummary({ applied }: { applied: FeedbackAppliedPatch[] }) {
  if (applied.length === 0) {
    return <p>No behavior patches were changed.</p>;
  }
  return (
    <p>
      {applied.length} behavior patch{applied.length === 1 ? "" : "es"} applied:{" "}
      {applied.map((patch) => `${patch.action} ${patchLabel(patch)}`).join("; ")}
    </p>
  );
}

function FeedbackDialog({ onClose }: { onClose: () => void }) {
  // Optional graph-state snapshot: the current graph values minus the heavy
  // transport channels (messages / generative-UI). The feedback server
  // summarizes whatever it receives; we keep the payload lean.
  const values = useLangGraphState();
  const [question, setQuestion] = useState("");
  const [responses, setResponses] = useState<string[]>([""]);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const graphState =
    values &&
    Object.keys(values).some((key) => key !== "messages" && key !== "ui")
      ? Object.fromEntries(
          Object.entries(values).filter(
            ([key]) => key !== "messages" && key !== "ui",
          ),
        )
      : null;

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (submitting) return;

    const payload: FeedbackRequestPayload = {
      question,
      responses: responses.map((r) => r.trim()).filter(Boolean),
      comment,
      graph_state: graphState,
    };

    setSubmitting(true);
    try {
      const applied = await sendFeedback(payload);
      onClose();
      toast("Feedback submitted", {
        description: () => <AppliedSummary applied={applied} />,
        richColors: true,
        closeButton: true,
        duration: 5000,
      });
    } catch (e) {
      toast.error("Failed to submit feedback", {
        description: e instanceof Error ? e.message : String(e),
        richColors: true,
        closeButton: true,
        duration: 5000,
      });
    } finally {
      setSubmitting(false);
    }
  };

  const updateResponse = (index: number, value: string) =>
    setResponses((prev) =>
      prev.map((r, i) => (i === index ? value : r)),
    );
  const addResponse = () => setResponses((prev) => [...prev, ""]);
  const removeResponse = (index: number) =>
    setResponses((prev) =>
      prev.length > 1 ? prev.filter((_, i) => i !== index) : prev,
    );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Share feedback"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="flex max-h-[85vh] w-full max-w-lg flex-col gap-4 overflow-y-auto rounded-xl border bg-background p-5 shadow-lg">
        <div>
          <h2 className="text-base font-semibold tracking-tight">
            Share feedback
          </h2>
          <p className="text-muted-foreground text-sm">
            Suggest an improvement to the learning behavior. The feedback
            service computes behavior patches and applies them to the registry.
          </p>
        </div>

        <form
          onSubmit={(event) => void handleSubmit(event)}
          className="flex flex-col gap-4"
        >
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="feedback-question">
              Question<span className="text-rose-500">*</span>
            </Label>
            <Input
              id="feedback-question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What should the assistant do differently?"
              required
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <Label>Responses</Label>
            <div className="flex flex-col gap-2">
              {responses.map((response, index) => (
                <div key={index} className="flex items-center gap-2">
                  <Input
                    value={response}
                    onChange={(e) => updateResponse(index, e.target.value)}
                    placeholder={`Response ${index + 1}`}
                    aria-label={`Response ${index + 1}`}
                  />
                  {responses.length > 1 && (
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      aria-label="Remove response"
                      onClick={() => removeResponse(index)}
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  )}
                </div>
              ))}
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="self-start"
              onClick={addResponse}
            >
              <Plus className="size-4" />
              Add response
            </Button>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="feedback-comment">Comment</Label>
            <Textarea
              id="feedback-comment"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Optional suggestion / comment"
            />
          </div>

          <div className="mt-1 flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={submitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? "Submitting…" : "Submit feedback"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function FeedbackAffordance() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label="Share feedback about this response"
        className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-muted-foreground/70 transition-colors hover:bg-muted hover:text-muted-foreground"
      >
        <MessageSquareQuote className="size-3.5" />
        Feedback
      </button>
      {open && <FeedbackDialog onClose={() => setOpen(false)} />}
    </>
  );
}