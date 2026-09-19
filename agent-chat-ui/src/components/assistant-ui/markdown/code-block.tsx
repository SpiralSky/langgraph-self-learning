"use client";

import { useState } from "react";
import { CheckIcon, CopyIcon } from "lucide-react";

import { SyntaxHighlighter } from "./syntax-highlighter";
import { TooltipIconButton } from "./tooltip-icon-button";

/**
 * CodeBlock — a polished, shared code block renderer.
 *
 * Used both by the built-in markdown ```` ``` ```` fences and the `:::code`
 * widget so every code snippet in chat looks consistent: a dark bordered
 * container with an optional header bar (title/language + copy button) and a
 * syntax-highlighted body. Unknown languages fall back to a plain, readable
 * mono block.
 */
export interface CodeBlockProps {
  /** Raw source code. */
  code: string;
  /** Language for syntax highlighting (e.g. `python`, `tsx`, `bash`). */
  language?: string;
  /** Optional header label shown alongside the language badge. */
  title?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code,
  language,
  title,
}) => {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = () => {
    if (!code || isCopied) return;
    navigator.clipboard.writeText(code).then(() => {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 3000);
    });
  };

  const hasHeader = Boolean(title || language);

  return (
    <div className="my-4 overflow-hidden rounded-xl border border-zinc-700/70 bg-zinc-900 shadow-sm">
      {hasHeader && (
        <div className="flex items-center justify-between gap-4 border-b border-zinc-700/70 bg-zinc-800/70 px-4 py-2">
          <span className="flex min-w-0 items-center gap-2 text-xs">
            {language && (
              <span className="rounded-md bg-zinc-700/70 px-1.5 py-0.5 font-mono text-[11px] font-semibold tracking-wide text-zinc-200 uppercase">
                {language}
              </span>
            )}
            {title && (
              <span className="truncate font-medium text-zinc-300">
                {title}
              </span>
            )}
          </span>
          <TooltipIconButton
            tooltip="Copy"
            onClick={handleCopy}
          >
            {isCopied ? (
              <CheckIcon className="size-4" />
            ) : (
              <CopyIcon className="size-4" />
            )}
          </TooltipIconButton>
        </div>
      )}
      <div className="overflow-x-auto">
        <SyntaxHighlighter language={language}>{code}</SyntaxHighlighter>
      </div>
    </div>
  );
};
