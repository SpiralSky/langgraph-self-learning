"use client";

import { useLangGraphUIMessages } from "@assistant-ui/react-langgraph";
import { XIcon } from "lucide-react";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { DataPartView } from "@/components/assistant-ui/generative-ui";
import { WorkspaceBrowser } from "@/components/assistant-ui/WorkspaceBrowser";

export type ArtifactPanelContextValue = {
  open: boolean;
  setOpen: (open: boolean) => void;
};

const ArtifactPanelContext = createContext<ArtifactPanelContextValue>({
  open: false,
  setOpen: () => {},
});

export function useArtifactPanel() {
  return useContext(ArtifactPanelContext);
}

/**
 * Provides the artifact side-panel open state. Owned by MyAssistant so the
 * panel can sit beside the chat and be force-closed on thread switch.
 */
export function ArtifactPanelProvider({
  open,
  onOpenChange,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: React.ReactNode;
}) {
  const value = useMemo(
    () => ({ open, setOpen: onOpenChange }),
    [open, onOpenChange],
  );
  return (
    <ArtifactPanelContext.Provider value={value}>
      {children}
    </ArtifactPanelContext.Provider>
  );
}

/**
 * Right-side artifact panel (same flex shell as the thread sidebar).
 *
 * Renders the LangGraph generative-UI messages accumulated by the
 * react-langgraph runtime (`ui` state key / `custom` events / persisted
 * `load()` uiMessages) — the assistant-ui form of the old
 * ArtifactProvider + ArtifactPanel. Auto-opens when a new artifact lands
 * (mirrors the old auto-open-on-`execute_code` behavior, but driven by UI
 * message arrival since that is what carries artifacts now).
 *
 * ## Override point (documented)
 * The chat-inline rendering of data parts is suppressed to a compact chip via
 * `MessagePrimitive.Parts components.data` (see `Thread.tsx`); the full
 * artifact renders here through the same widget components via `DataPartView`
 * (registered per name with `makeAssistantDataUI` in `generative-ui.tsx`).
 * The workspace file browser (old ArtifactPanel content) is re-added by
 * task-07 on top of this panel.
 */
export function ArtifactPanel() {
  const { open, setOpen } = useArtifactPanel();
  const uiMessages = useLangGraphUIMessages();
  const [tab, setTab] = useState<"artifact" | "files">("artifact");

  const lastAutoOpenedSignature = useRef<string | null>(null);
  useEffect(() => {
    if (uiMessages.length === 0) return;
    const signature = JSON.stringify(
      uiMessages.map((ui) => [ui.id, ui.name, ui.props]),
    );
    if (lastAutoOpenedSignature.current === signature) return;
    lastAutoOpenedSignature.current = signature;
    setOpen(true);
  }, [setOpen, uiMessages]);

  if (!open) return null;

  return (
    <aside className="flex h-full w-[min(420px,80vw)] shrink-0 flex-col border-l bg-background text-foreground">
      <header className="flex items-center justify-between border-b px-3 py-2">
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Artifact
          </span>
          <div className="flex rounded-lg bg-muted p-0.5 text-xs">
            <button
              type="button"
              className={`rounded-md px-2 py-1 transition-colors ${
                tab === "artifact"
                  ? "bg-background font-medium shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
              onClick={() => setTab("artifact")}
            >
              Artifact
            </button>
            <button
              type="button"
              className={`rounded-md px-2 py-1 transition-colors ${
                tab === "files"
                  ? "bg-background font-medium shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
              onClick={() => setTab("files")}
            >
              Files
            </button>
          </div>
        </div>
        <button
          type="button"
          aria-label="Close artifact panel"
          className="rounded p-1 text-muted-foreground hover:bg-muted"
          onClick={() => setOpen(false)}
        >
          <XIcon className="size-5" />
        </button>
      </header>
      <div className="flex-1 overflow-y-auto p-4">
        {tab === "files" ? (
          <WorkspaceBrowser />
        ) : uiMessages.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No artifact yet. Ask the agent to generate code or analysis and it
            will appear here.
          </p>
        ) : (
          <div className="flex flex-col gap-4">
            {uiMessages.map((ui) => (
              <DataPartView
                key={ui.id}
                name={ui.name}
                data={ui.props}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}