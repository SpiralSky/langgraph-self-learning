import { LangGraphInterruptState } from "@assistant-ui/react-langgraph";
import { HITLRequest } from "./types";

/**
 * Structural check for the human-in-the-loop interrupt payload emitted by
 * langchain's `interrupt()` (action_requests + review_configs) — the same
 * schema the old `lib/agent-inbox-interrupt.ts` matched against.
 *
 * Works on `LangGraphInterruptState.value` (the runtime surfaces the raw
 * interrupt value; the rest of the interrupt object is protocol plumbing).
 */
export function isHitlInterruptValue(value: unknown): value is HITLRequest {
  if (!value || typeof value !== "object") {
    return false;
  }

  const hitlValue = value as Partial<HITLRequest>;
  const { action_requests: actionRequests, review_configs: reviewConfigs } =
    hitlValue;

  if (!Array.isArray(actionRequests) || actionRequests.length === 0) {
    return false;
  }
  if (!Array.isArray(reviewConfigs) || reviewConfigs.length === 0) {
    return false;
  }

  const hasValidActionRequests = actionRequests.every((request) => {
    return (
      request &&
      typeof request === "object" &&
      "name" in request &&
      typeof request.name === "string" &&
      "args" in request &&
      request.args !== null &&
      typeof request.args === "object"
    );
  });

  const hasValidConfigs = reviewConfigs.every((config) => {
    return (
      config &&
      typeof config === "object" &&
      "action_name" in config &&
      typeof config.action_name === "string" &&
      "allowed_decisions" in config &&
      Array.isArray(config.allowed_decisions)
    );
  });

  return hasValidActionRequests && hasValidConfigs;
}

export function isHitlInterrupt(
  interrupt: LangGraphInterruptState | undefined,
): boolean {
  return !!interrupt && isHitlInterruptValue(interrupt.value);
}