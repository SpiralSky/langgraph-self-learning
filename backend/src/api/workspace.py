from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

from graphs.learning_graph.config import config

WORKSPACE_ROOT = Path(config.workspace_host_path).resolve()
MAX_FILE_SIZE = 100 * 1024

TEXT_EXTENSIONS = {
    ".txt", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
    ".md", ".html", ".css", ".csv", ".xml", ".sh", ".bash", ".sql",
    ".rst", ".toml", ".ini", ".cfg", ".log", ".env", ".gitignore",
    ".dockerignore", ".svg", ".txt",
}
BINARY_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv",
    ".wav", ".ogg", ".flac", ".aac", ".m4a", ".webm",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    ".exe", ".bin", ".dll", ".so", ".dylib", ".pyc", ".pyo",
}

router = APIRouter(prefix="/api/workspace")


def _resolve_path(subpath: str) -> Path:
    target = (WORKSPACE_ROOT / subpath).resolve()
    try:
        target.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise HTTPException(status_code=404, detail="Path not found")
    return target


def _is_binary(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return False
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return True
        text_chars = sum(1 for b in chunk if 32 <= b < 127 or b in (9, 10, 13))
        return text_chars / max(len(chunk), 1) < 0.85
    except (IOError, OSError):
        return True


HOST_PATH = Path(config.workspace_host_path).resolve()


@router.get("/files")
async def list_files(path: str = Query(".", description="Subpath within workspace")):
    target = _resolve_path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    items = []
    for item in sorted(target.iterdir()):
        try:
            rel_path = item.relative_to(WORKSPACE_ROOT)
        except ValueError:
            continue
        items.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "path": str(rel_path),
            "host_path": str(HOST_PATH / rel_path),
        })
    return items


@router.get("/file")
async def read_file(path: str = Query(..., description="Subpath within workspace")):
    target = _resolve_path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    if _is_binary(target):
        raise HTTPException(status_code=403, detail="Binary files are not supported")
    size = target.stat().st_size
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large ({size} bytes, max {MAX_FILE_SIZE})")
    try:
        content = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=403, detail="Cannot read file as text")
    return {
        "name": target.name,
        "content": content,
        "size": size,
        "host_path": str(HOST_PATH / target.relative_to(WORKSPACE_ROOT)),
    }
