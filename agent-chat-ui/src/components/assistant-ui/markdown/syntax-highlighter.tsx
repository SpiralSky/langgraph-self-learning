import { PrismAsyncLight as SyntaxHighlighterPrism } from "react-syntax-highlighter";
import tsx from "react-syntax-highlighter/dist/esm/languages/prism/tsx";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import jsx from "react-syntax-highlighter/dist/esm/languages/prism/jsx";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import markup from "react-syntax-highlighter/dist/esm/languages/prism/markup";
import css from "react-syntax-highlighter/dist/esm/languages/prism/css";
import c from "react-syntax-highlighter/dist/esm/languages/prism/c";
import cpp from "react-syntax-highlighter/dist/esm/languages/prism/cpp";
import csharp from "react-syntax-highlighter/dist/esm/languages/prism/csharp";
import go from "react-syntax-highlighter/dist/esm/languages/prism/go";
import java from "react-syntax-highlighter/dist/esm/languages/prism/java";
import kotlin from "react-syntax-highlighter/dist/esm/languages/prism/kotlin";
import rust from "react-syntax-highlighter/dist/esm/languages/prism/rust";
import swift from "react-syntax-highlighter/dist/esm/languages/prism/swift";
import ruby from "react-syntax-highlighter/dist/esm/languages/prism/ruby";
import php from "react-syntax-highlighter/dist/esm/languages/prism/php";
import dart from "react-syntax-highlighter/dist/esm/languages/prism/dart";
import scala from "react-syntax-highlighter/dist/esm/languages/prism/scala";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import graphql from "react-syntax-highlighter/dist/esm/languages/prism/graphql";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import diff from "react-syntax-highlighter/dist/esm/languages/prism/diff";
import toml from "react-syntax-highlighter/dist/esm/languages/prism/toml";
import docker from "react-syntax-highlighter/dist/esm/languages/prism/docker";
import git from "react-syntax-highlighter/dist/esm/languages/prism/git";
import makefile from "react-syntax-highlighter/dist/esm/languages/prism/makefile";
import ini from "react-syntax-highlighter/dist/esm/languages/prism/ini";
import powershell from "react-syntax-highlighter/dist/esm/languages/prism/powershell";
import shellSession from "react-syntax-highlighter/dist/esm/languages/prism/shell-session";
import objectivec from "react-syntax-highlighter/dist/esm/languages/prism/objectivec";
import lua from "react-syntax-highlighter/dist/esm/languages/prism/lua";
import { vscDarkPlus } from "react-syntax-highlighter/dist/cjs/styles/prism";
import { cn } from "@/lib/utils";
import { FC } from "react";

const REGISTERED: [string, unknown][] = [
  ["tsx", tsx],
  ["typescript", typescript],
  ["javascript", javascript],
  ["jsx", jsx],
  ["python", python],
  ["bash", bash],
  ["json", json],
  ["yaml", yaml],
  ["markup", markup],
  ["css", css],
  ["c", c],
  ["cpp", cpp],
  ["csharp", csharp],
  ["go", go],
  ["java", java],
  ["kotlin", kotlin],
  ["rust", rust],
  ["swift", swift],
  ["ruby", ruby],
  ["php", php],
  ["dart", dart],
  ["scala", scala],
  ["sql", sql],
  ["graphql", graphql],
  ["markdown", markdown],
  ["diff", diff],
  ["toml", toml],
  ["docker", docker],
  ["git", git],
  ["makefile", makefile],
  ["ini", ini],
  ["powershell", powershell],
  ["shell-session", shellSession],
  ["objectivec", objectivec],
  ["lua", lua],
];

for (const [name, module] of REGISTERED) {
  SyntaxHighlighterPrism.registerLanguage(
    name,
    module as Parameters<typeof SyntaxHighlighterPrism.registerLanguage>[1],
  );
}

const SUPPORTED_LANGUAGES = new Set(REGISTERED.map(([name]) => name));

const ALIASES: Record<string, string> = {
  py: "python",
  python2: "python",
  python3: "python",
  js: "javascript",
  node: "javascript",
  nodejs: "javascript",
  ts: "typescript",
  sh: "bash",
  shell: "bash",
  zsh: "bash",
  console: "shell-session",
  shellsession: "shell-session",
  yml: "yaml",
  html: "markup",
  htm: "markup",
  xml: "markup",
  svg: "markup",
  xhtml: "markup",
  mdx: "markdown",
  md: "markdown",
  jsonc: "json",
  json5: "json",
  diff: "diff",
  patch: "diff",
  "c++": "cpp",
  cplusplus: "cpp",
  cs: "csharp",
  objc: "objectivec",
  ps1: "powershell",
  dockerfile: "docker",
  gql: "graphql",
  plaintext: "text",
  txt: "text",
  text: "text",
};

function resolveLanguage(language?: string): string | null {
  if (!language) return null;
  const key = language.toLowerCase().trim();
  const stripped = key.replace(/[\d.]+$/, "");
  const canonical = ALIASES[key] ?? ALIASES[stripped] ?? stripped;
  return SUPPORTED_LANGUAGES.has(canonical) ? canonical : null;
}

interface SyntaxHighlighterProps {
  children: string;
  language?: string;
  className?: string;
}

const CODE_FONT_FAMILY =
  'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';

export const SyntaxHighlighter: FC<SyntaxHighlighterProps> = ({
  children,
  language,
  className,
}) => {
  const resolved = resolveLanguage(language);

  if (!resolved) {
    return (
      <pre
        className={cn(
          "max-w-4xl px-4 py-3 text-sm leading-relaxed text-zinc-200",
          className,
        )}
        style={{
          fontFamily: CODE_FONT_FAMILY,
          whiteSpace: "pre",
          wordBreak: "normal",
        }}
      >
        <code>{children}</code>
      </pre>
    );
  }

  return (
    <SyntaxHighlighterPrism
      language={resolved}
      style={vscDarkPlus}
      customStyle={{
        margin: 0,
        width: "100%",
        background: "transparent",
        padding: "1.5rem 1rem",
        fontFamily: CODE_FONT_FAMILY,
        fontSize: "0.875rem",
        lineHeight: "1.7",
      }}
      className={className}
    >
      {children}
    </SyntaxHighlighterPrism>
  );
};
