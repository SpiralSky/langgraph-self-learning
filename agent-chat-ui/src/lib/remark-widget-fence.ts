/**
 * Tokenize a markdown string, extracting `:::` widget fences into discrete
 * segments so they can be rendered by the `WidgetRenderer` directly — outside
 * the markdown parser.
 *
 * Widget fences are line-based: an opening line `:::name attr="value" ...`
 * followed by raw body content and a closing line `:::`. Everything between the
 * opening and closing markers is treated as raw body content — no markdown
 * parsing is applied, so multi-line code snippets keep their exact blank lines
 * and indentation.
 *
 * Fences are intentionally NOT converted to raw HTML before markdown parsing:
 * react-markdown renders raw HTML as escaped text (no `rehype-raw`), and
 * CommonMark ends HTML blocks at the first blank line, which would split a
 * multi-line widget body into garbage markdown.
 *
 * ## Syntax
 *
 * ```markdown
 * :::text title="Info" color="blue"
 * This is the body of the text widget.
 * :::
 *
 * :::code title="My Snippet" language="tsx"
 * const x = 1;
 * :::
 * ```
 *
 * @param md - Raw markdown string.
 * @returns Ordered segments. Consecutive non-fence runs are kept as single
 *   ``"markdown"`` segments; each matched fence becomes a ``"widget"`` segment.
 *   An unclosed fence is left inside the enclosing markdown segment so the
 *   `:::` syntax stays visible to the author.
 */
export interface WidgetSegment {
  type: "widget";
  /** Widget name from the opening `:::` line (e.g. `code`). */
  name: string;
  /** Parsed fence attributes (attribute name → value), e.g. `{ language: "tsx" }`. */
  props: Record<string, string>;
  /** Raw body text with leading/trailing blank lines trimmed. */
  children: string;
}

export interface MarkdownSegment {
  type: "markdown";
  text: string;
}

export type WidgetFenceSegment = WidgetSegment | MarkdownSegment;

export function parseWidgetFences(md: string): WidgetFenceSegment[] {
  const segments: WidgetFenceSegment[] = [];
  const lines = md.split("\n");

  const openFence = /^:::(\w+)((?:[ \t]+\w+(?:="[^"]*")?)*)[ \t]*$/;
  const closeFence = /^:::[ \t]*$/;

  let markdownStart = 0;
  let i = 0;

  const flushMarkdown = (end: number): void => {
    if (end > markdownStart) {
      segments.push({
        type: "markdown",
        text: lines.slice(markdownStart, end).join("\n"),
      });
    }
  };

  while (i < lines.length) {
    const open = openFence.exec(lines[i]);
    if (!open) {
      i += 1;
      continue;
    }

    let j = i + 1;
    while (j < lines.length && !closeFence.test(lines[j])) {
      j += 1;
    }
    if (j >= lines.length) {
      // Unclosed fence — treat the opening line as plain markdown.
      i += 1;
      continue;
    }

    const [, name, attrs] = open;
    const props: Record<string, string> = {};
    if (attrs) {
      const attrRegex = /(\w+)(?:="([^"]*)")?/g;
      let m: RegExpExecArray | null;
      while ((m = attrRegex.exec(attrs)) !== null) {
        props[m[1]] = m[2] ?? "";
      }
    }

    flushMarkdown(i);
    segments.push({
      type: "widget",
      name,
      props,
      children: lines.slice(i + 1, j).join("\n").trim(),
    });

    i = j + 1;
    markdownStart = i;
  }

  flushMarkdown(lines.length);

  return segments;
}