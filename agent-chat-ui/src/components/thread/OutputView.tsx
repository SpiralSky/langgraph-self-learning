interface OutputViewProps {
  stdout?: string;
  stderr?: string;
}

export function OutputView({ stdout, stderr }: OutputViewProps) {
  return (
    <div className="flex flex-col gap-2 p-3 font-mono text-xs leading-relaxed">
      {stdout ? (
        <pre className="whitespace-pre-wrap text-zinc-200">{stdout}</pre>
      ) : null}
      {stderr ? (
        <pre className="whitespace-pre-wrap text-red-400">{stderr}</pre>
      ) : null}
      {!stdout && !stderr ? (
        <span className="text-zinc-500">No output</span>
      ) : null}
    </div>
  );
}