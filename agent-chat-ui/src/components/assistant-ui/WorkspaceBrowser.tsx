"use client";

import { ExternalLink, FolderTree, RefreshCw, Download } from "lucide-react";
import { useCallback, useState } from "react";

import { CodeViewer } from "@/components/thread/CodeViewer";
import { FileTree } from "@/components/thread/FileTree";
import { OutputView } from "@/components/thread/OutputView";
import { useFileContent } from "@/hooks/useFileContent";
import { useWorkspaceFiles } from "@/hooks/useWorkspaceFiles";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

/**
 * Workspace file browser, ported from the old ArtifactPanel and mounted in the
 * new artifact side panel ("Files" tab).
 *
 * Known backend gap (shared.md): `/api/workspace/files` and
 * `/api/workspace/file` have NO backend implementation anywhere — these were
 * frontend-only calls against the LangGraph server. Per the agreed decision
 * `(a)` the hooks are kept and the panel renders a graceful "unavailable"
 * state when the fetch fails, so the feature activates automatically the day a
 * backend endpoint exists. No backend code is added in this plan.
 */
export function WorkspaceBrowser() {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const {
    items,
    loading: filesLoading,
    error: filesError,
    refetch,
  } = useWorkspaceFiles();
  const selectedHostPath =
    items.find((item) => item.path === selectedPath)?.host_path ?? null;
  const {
    content,
    loading: contentLoading,
    error: contentError,
    isBinary,
    isTooLarge,
  } = useFileContent(selectedPath ?? "");

  const handleDownloadFile = useCallback(async () => {
    if (!selectedPath) return;
    const res = await fetch(
      `${API_BASE}/api/workspace/file?path=${encodeURIComponent(selectedPath)}`,
    );
    if (!res.ok) return;
    const data: { name: string; content: string } = await res.json();
    const blob = new Blob([data.content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = data.name;
    a.click();
    URL.revokeObjectURL(url);
  }, [selectedPath]);

  const handleOpenInVSCode = useCallback(() => {
    if (selectedHostPath) {
      window.open(`vscode://file/${selectedHostPath}`, "_blank");
    }
  }, [selectedHostPath]);

  const unavailable = filesError !== null && !filesLoading;

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-md border border-border">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          <FolderTree className="size-4" />
          Workspace files
        </div>
        <button
          type="button"
          aria-label="Refresh file list"
          className="rounded p-1 text-muted-foreground hover:bg-muted"
          onClick={() => void refetch()}
        >
          <RefreshCw className="size-4" />
        </button>
      </div>
      <div className="flex min-h-0 flex-1">
        <div className="w-56 shrink-0 overflow-y-auto border-r border-border">
          <FileTree
            items={items}
            selectedPath={selectedPath}
            onSelect={setSelectedPath}
          />
        </div>
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-1.5 text-xs">
            <span className="truncate font-mono text-muted-foreground">
              {selectedPath ?? "—"}
            </span>
            <div className="flex shrink-0 items-center gap-1">
              <button
                type="button"
                aria-label="Download file"
                disabled={!selectedPath}
                className="rounded p-1 text-muted-foreground hover:bg-muted disabled:opacity-40"
                onClick={() => void handleDownloadFile()}
              >
                <Download className="size-3.5" />
              </button>
              <button
                type="button"
                aria-label="Open in VS Code"
                disabled={!selectedHostPath}
                className="rounded p-1 text-muted-foreground hover:bg-muted disabled:opacity-40"
                onClick={handleOpenInVSCode}
              >
                <ExternalLink className="size-3.5" />
              </button>
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-auto">
            {unavailable ? (
              <div className="p-4 text-sm text-muted-foreground">
                <p className="mb-1 font-medium text-foreground">
                  Workspace file browsing is not available
                </p>
                <p className="text-xs leading-relaxed">
                  The <code>GET /api/workspace/files</code> endpoint has no
                  backend implementation yet (it was a frontend-only call in the
                  old app). The browser stays wired, so it activates
                  automatically once a backend endpoint exists. (Last error:{" "}
                  {filesError})
                </p>
              </div>
            ) : filesLoading || contentLoading ? (
              <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                Loading…
              </div>
            ) : contentError ? (
              <div className="p-4 text-sm text-red-500">Error: {contentError}</div>
            ) : items.length === 0 ? (
              <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                No files in workspace
              </div>
            ) : selectedPath && content ? (
              <CodeViewer
                code={content}
                isBinary={isBinary}
                isTooLarge={isTooLarge}
              />
            ) : (
              <OutputView stdout="" stderr="" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}