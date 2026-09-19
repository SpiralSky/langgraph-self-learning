import { CodeBlock } from "@/components/assistant-ui/markdown/code-block";

/**
 * CodeBlockWidget — a syntax-highlighted code block with a title bar
 * and a copy-to-clipboard button.
 *
 * This widget mirrors the behaviour of the built-in markdown ` ``` ` code
 * fences but is registered as a first-class widget so it can be invoked
 * via `:::code`. The language is detected from the `language` attribute;
 * the title is shown in the header bar.
 *
 * ## Accepted props (from the ::: fence attributes)
 *
 * | Attribute  | Type   | Default    | Description                |
 * |------------|--------|------------|----------------------------|
 * | `language` | string | `""`       | Language for highlighting. |
 * | `title`    | string | `—`        | Optional header label.     |
 *
 * The fence body is the raw source code.
 *
 * ## Example
 *
 * ```markdown
 * :::code language="tsx" title="Component"
 * export function Hello() { return <div>Hi</div> }
 * :::
 * ```
 */
export interface CodeBlockWidgetProps {
  language?: string;
  title?: string;
}

export const CodeBlockWidget: React.FC<{
  props: CodeBlockWidgetProps;
  children?: string;
}> = ({ props, children }) => {
  const { language, title } = props;

  return (
    <CodeBlock
      code={(children ?? "").trimEnd()}
      language={language}
      title={title}
    />
  );
};
