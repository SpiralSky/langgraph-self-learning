import { useMemo } from "react";
import { Tree } from "react-arborist";
import { File, Folder, FileText } from "lucide-react";

interface RawItem {
  path: string;
  type: string;
}

interface FileTreeProps {
  items: RawItem[];
  selectedPath: string | null;
  onSelect: (path: string) => void;
}

function buildTree(items: RawItem[]) {
  const root: { id: string; name: string; children: any[] } = {
    id: ".",
    name: "workspace",
    children: [],
  };

  for (const item of items) {
    const parts = item.path.split("/").filter(Boolean);
    let current = root;
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isFile = i === parts.length - 1 && item.type === "file";
      const childId = parts.slice(0, i + 1).join("/");
      const existing = current.children.find((c: any) => c.id === childId);
      if (existing) {
        current = existing;
        continue;
      }
      const node: any = {
        id: childId,
        name: part,
        type: isFile ? "file" : "directory",
        children: isFile ? undefined : [],
      };
      current.children.push(node);
      if (!isFile) current = node;
    }
  }

  return root.children;
}

function TreeIcon({ type, name }: { type: string; name: string }) {
  if (type === "directory") {
    return <Folder className="h-4 w-4 shrink-0 text-amber-500" />;
  }
  const ext = name.split(".").pop()?.toLowerCase();
  if (ext === "ts" || ext === "tsx" || ext === "js" || ext === "jsx") {
    return <FileText className="h-4 w-4 shrink-0 text-blue-400" />;
  }
  return <File className="h-4 w-4 shrink-0 text-zinc-400" />;
}

function ArboristRow({
  node,
  style,
  onSelect,
  selectedPath,
}: {
  node: any;
  style: any;
  onSelect: (path: string) => void;
  selectedPath: string | null;
}) {
  const isSelected = node.id === selectedPath;
  return (
    <div
      style={style}
      className={`flex items-center gap-1 rounded px-1 py-0.5 text-sm cursor-pointer select-none ${
        isSelected
          ? "bg-zinc-700 text-white"
          : "text-zinc-300 hover:bg-zinc-800 hover:text-white"
      }`}
      onClick={() => onSelect(node.id)}
    >
      <TreeIcon type={node.type} name={node.name} />
      <span className="truncate">{node.name}</span>
    </div>
  );
}

export function FileTree({ items, selectedPath, onSelect }: FileTreeProps) {
  const treeData = useMemo(() => buildTree(items), [items]);

  if (treeData.length === 0) {
    return <div className="p-3 text-sm text-zinc-500">No files found</div>;
  }

  return (
    <Tree
      data={treeData}
      openByDefault={true}
      selection={selectedPath ?? undefined}
      onSelect={(nodes: any[]) => {
        if (nodes.length > 0) onSelect(nodes[0].id);
      }}
    >
      {(props: any) => (
        <ArboristRow
          node={props.node}
          style={props.style}
          onSelect={onSelect}
          selectedPath={selectedPath}
        />
      )}
    </Tree>
  );
}