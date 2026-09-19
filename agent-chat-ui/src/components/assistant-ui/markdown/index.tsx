"use client";

import "./markdown-styles.css";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { FC, isValidElement, memo, ReactElement } from "react";

import { cn } from "@/lib/utils";
import { parseWidgetFences } from "@/lib/remark-widget-fence";
import { sanitizeUrl } from "@/lib/markdown-security";
import { WidgetRenderer } from "./widget-renderer";
import { CodeBlock } from "./code-block";

import "katex/dist/katex.min.css";

/**
 * Markdown rendering pipeline for the assistant-ui message parts.
 *
 * Ported from the old `components/thread/markdown.tsx` by task-06 so the
 * widget feature no longer depends on the old thread tree (task-09 deletes
 * the stale copies). The pipeline is unchanged: the input is split into
 * plain markdown and `:::` widget segments via `parseWidgetFences`, each
 * rendered through the same react-markdown stack (GFM, math, KaTeX and the
 * shared component styles) or the `WidgetRenderer` respectively. Widget
 * bodies use the same `Markdown` renderer, so they support markdown just
 * like regular chat text.
 *
 * Raw HTML is never rendered (it is escaped as text) and all URLs are passed
 * through `sanitizeUrl`, so hostile schemes such as `javascript:` or `data:`
 * are dropped before they reach an `href` or image `src`.
 */

const katexOptions = {
  throwOnError: false,
  errorColor: "#cc0000",
  strict: "ignore",
} as const;

const defaultComponents: any = {
  h1: ({ className, ...props }: { className?: string }) => (
    <h1
      className={cn(
        "mb-8 scroll-m-20 text-4xl font-extrabold tracking-tight last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h2: ({ className, ...props }: { className?: string }) => (
    <h2
      className={cn(
        "mt-8 mb-4 scroll-m-20 text-3xl font-semibold tracking-tight first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h3: ({ className, ...props }: { className?: string }) => (
    <h3
      className={cn(
        "mt-6 mb-4 scroll-m-20 text-2xl font-semibold tracking-tight first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h4: ({ className, ...props }: { className?: string }) => (
    <h4
      className={cn(
        "mt-6 mb-4 scroll-m-20 text-xl font-semibold tracking-tight first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h5: ({ className, ...props }: { className?: string }) => (
    <h5
      className={cn(
        "my-4 text-lg font-semibold first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h6: ({ className, ...props }: { className?: string }) => (
    <h6
      className={cn("my-4 font-semibold first:mt-0 last:mb-0", className)}
      {...props}
    />
  ),
  p: ({ className, ...props }: { className?: string }) => (
    <p
      className={cn("mt-5 mb-5 leading-7 first:mt-0 last:mb-0", className)}
      {...props}
    />
  ),
  a: ({
    node: _node,
    className,
    href,
    target,
    rel,
    ...props
  }: {
    node?: unknown;
    className?: string;
    href?: string;
    target?: string;
    rel?: string;
  }) => {
    const safeHref = sanitizeUrl(href);
    return (
      <a
        className={cn(
          "text-primary font-medium underline underline-offset-4",
          className,
        )}
        href={safeHref || undefined}
        target={target}
        rel={target ? rel ?? "noopener noreferrer" : rel}
        {...props}
      />
    );
  },
  blockquote: ({ className, ...props }: { className?: string }) => (
    <blockquote
      className={cn("border-l-2 pl-6 italic", className)}
      {...props}
    />
  ),
  ul: ({ className, ...props }: { className?: string }) => (
    <ul
      className={cn("my-5 ml-6 list-disc [&>li]:mt-2", className)}
      {...props}
    />
  ),
  ol: ({ className, ...props }: { className?: string }) => (
    <ol
      className={cn("my-5 ml-6 list-decimal [&>li]:mt-2", className)}
      {...props}
    />
  ),
  hr: ({ className, ...props }: { className?: string }) => (
    <hr
      className={cn("my-5 border-b", className)}
      {...props}
    />
  ),
  table: ({ className, ...props }: { className?: string }) => (
    <table
      className={cn(
        "my-5 w-full border-separate border-spacing-0 overflow-y-auto",
        className,
      )}
      {...props}
    />
  ),
  th: ({ className, ...props }: { className?: string }) => (
    <th
      className={cn(
        "bg-muted px-4 py-2 text-left font-bold first:rounded-tl-lg last:rounded-tr-lg [&[align=center]]:text-center [&[align=right]]:text-right",
        className,
      )}
      {...props}
    />
  ),
  td: ({ className, ...props }: { className?: string }) => (
    <td
      className={cn(
        "border-b border-l px-4 py-2 text-left last:border-r [&[align=center]]:text-center [&[align=right]]:text-right",
        className,
      )}
      {...props}
    />
  ),
  tr: ({ className, ...props }: { className?: string }) => (
    <tr
      className={cn(
        "m-0 border-b p-0 first:border-t [&:last-child>td:first-child]:rounded-bl-lg [&:last-child>td:last-child]:rounded-br-lg",
        className,
      )}
      {...props}
    />
  ),
  sup: ({ className, ...props }: { className?: string }) => (
    <sup
      className={cn("[&>a]:text-xs [&>a]:no-underline", className)}
      {...props}
    />
  ),
  pre: ({ children }: { children?: React.ReactNode }) => {
    let code = "";
    let language: string | undefined;

    if (typeof children === "string") {
      code = children;
    } else if (isValidElement(children)) {
      const codeEl = children as ReactElement<{
        className?: string;
        children?: React.ReactNode;
      }>;
      language = /language-(\w+)/.exec(codeEl.props?.className ?? "")?.[1];
      const raw = codeEl.props?.children;
      code = Array.isArray(raw) ? raw.join("") : String(raw ?? "");
    }

    return (
      <CodeBlock
        code={code.replace(/\n$/, "")}
        language={language}
      />
    );
  },
  code: ({
    className,
    children,
    ...props
  }: {
    className?: string;
    children?: React.ReactNode;
  }) => (
    <code
      className={cn("bg-muted rounded-md px-1.5 py-0.5 font-medium", className)}
      {...props}
    >
      {children}
    </code>
  ),
  img: ({
    src,
    alt,
    title,
  }: {
    src?: string;
    alt?: string;
    title?: string;
  }) => {
    const safeSrc = sanitizeUrl(src);
    if (!safeSrc) {
      return null;
    }
    return (
      <img
        src={safeSrc}
        alt={alt ?? ""}
        title={title}
      />
    );
  },
  div: ({
    node,
    children,
    className,
    ...props
  }: {
    node?: { properties?: Record<string, unknown> };
    children?: React.ReactNode;
    className?: string;
  }) => (
    <div
      className={cn(className)}
      {...props}
    >
      {children}
    </div>
  ),
};

/**
 * Markdown — renders a markdown string exactly like the text of the old app:
 * plain markdown and `:::` widget segments each go through the same
 * react-markdown pipeline (GFM, math, KaTeX, shared component styles) or the
 * `WidgetRenderer`. Streaming-safe: it re-parses whatever text it is given on
 * every render, so partially streamed content (including a fence whose
 * closing `:::` has not arrived yet) renders progressively.
 */
const MarkdownImpl: FC<{ children: string }> = ({ children }) => {
  const segments = parseWidgetFences(children);
  return (
    <div className="markdown-content">
      {segments.map((segment, index) => {
        if (segment.type === "widget") {
          return (
            <WidgetRenderer
              key={index}
              name={segment.name}
              widgetProps={segment.props}
            >
              {segment.children}
            </WidgetRenderer>
          );
        }
        return (
          <ReactMarkdown
            key={index}
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[[rehypeKatex, katexOptions]]}
            components={defaultComponents}
            urlTransform={sanitizeUrl}
          >
            {segment.text}
          </ReactMarkdown>
        );
      })}
    </div>
  );
};

export const Markdown = memo(MarkdownImpl);

/** Alias kept for consumers of the old `components/thread/markdown-text.tsx` (agent-inbox). */
export const MarkdownText = Markdown;