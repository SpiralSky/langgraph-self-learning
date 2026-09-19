"use client";

import { useState } from "react";
import {
  useLangGraphInterruptState,
  useLangGraphState,
} from "@assistant-ui/react-langgraph";
import { StateView } from "./components/state-view";
import { ThreadActionsView } from "./components/thread-actions-view";
import { isHitlInterruptValue } from "./interrupt-schema";

/**
 * Human-in-the-loop interrupt panel, rebuilt on assistant-ui's interrupt
 * primitives:
 * - `useLangGraphInterruptState` — the pending interrupt (streamed via
 *   `updates.__interrupt__` events or restored from thread state on load).
 * - `useLangGraphSendCommand` — resumes the interrupted run with the HITL
 *   decisions (`{ resume: { decisions } }`) or resolves it (`{ goto: END }`).
 * - `useLangGraphState` — live graph state for the state inspector.
 *
 * Renders null when no interrupt is pending. Old agent-inbox behaviors
 * preserved: approve/deny/edit input cards, per-action addressing for
 * multiple action requests, state/description side panel. The multi-tab
 * interrupt list is dropped — the runtime surfaces a single (first) pending
 * interrupt at a time.
 */
export function ThreadView() {
  const interrupt = useLangGraphInterruptState();
  const values = useLangGraphState();
  const [showState, setShowState] = useState(false);
  const [showDescription, setShowDescription] = useState(false);
  const showSidePanel = showState || showDescription;

  if (!interrupt) {
    return null;
  }

  const hitlValue = isHitlInterruptValue(interrupt.value)
    ? interrupt.value
    : undefined;
  const activeDescription = hitlValue?.action_requests?.[0]?.description ?? "";

  const handleShowSidePanel = (
    showStateFlag: boolean,
    showDescriptionFlag: boolean,
  ) => {
    if (showStateFlag && showDescriptionFlag) {
      console.error("Cannot show both state and description");
      return;
    }
    if (showStateFlag) {
      setShowDescription(false);
      setShowState(true);
    } else if (showDescriptionFlag) {
      setShowState(false);
      setShowDescription(true);
    } else {
      setShowState(false);
      setShowDescription(false);
    }
  };

  return (
    <div className="flex h-full w-full flex-col rounded-2xl bg-gray-50 p-8 lg:flex-row">
      {showSidePanel ? (
        <StateView
          handleShowSidePanel={handleShowSidePanel}
          description={activeDescription}
          values={values ?? {}}
          view={showState ? "state" : "description"}
        />
      ) : hitlValue ? (
        <div className="flex w-full flex-col gap-6">
          <ThreadActionsView
            interrupt={interrupt}
            handleShowSidePanel={handleShowSidePanel}
            showState={showState}
            showDescription={showDescription}
          />
        </div>
      ) : (
        <div className="flex w-full flex-col gap-6">
          <div className="overflow-hidden rounded-lg border border-gray-200">
            <div className="border-b border-gray-200 bg-gray-50 px-4 py-2">
              <h3 className="font-medium text-gray-900">Human Interrupt</h3>
            </div>
            <pre className="max-h-[500px] overflow-auto p-3 font-mono text-xs whitespace-pre-wrap text-gray-700">
              {JSON.stringify(interrupt.value ?? null, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}