import { SyntaxHighlighter } from "@/components/assistant-ui/markdown/syntax-highlighter";

interface CodeViewerProps {
  code: string;
  language?: string;
  isBinary?: boolean;
  isTooLarge?: boolean;
}

export function CodeViewer({
  code,
  language,
  isBinary,
  isTooLarge,
}: CodeViewerProps) {
  if (isBinary) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-500">
        Binary file
      </div>
    );
  }

  if (isTooLarge) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-500">
        File too large to display
      </div>
    );
  }

  return <SyntaxHighlighter language={language}>{code}</SyntaxHighlighter>;
}