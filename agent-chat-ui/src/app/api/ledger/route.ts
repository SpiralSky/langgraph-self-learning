import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const LEDGER_PATH = process.env.LEDGER_PATH ?? "/shared/ledger";
const PENDING_DIR = path.join(LEDGER_PATH, "pending");
const APPROVALS_DIR = path.join(LEDGER_PATH, "approvals");
const REJECTED_DIR = path.join(LEDGER_PATH, "rejected");

async function listPending(): Promise<Array<Record<string, unknown>>> {
  await fs.mkdir(PENDING_DIR, { recursive: true });
  const entries = await fs.readdir(PENDING_DIR);
  const results: Array<Record<string, unknown>> = [];
  for (const entry of entries) {
    if (!entry.endsWith(".json")) continue;
    try {
      const raw = await fs.readFile(path.join(PENDING_DIR, entry), "utf8");
      results.push({ filename: entry, ...JSON.parse(raw) });
    } catch {
      continue;
    }
  }
  return results;
}

export async function GET(): Promise<NextResponse> {
  return NextResponse.json({ pending: await listPending() });
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  const body = (await req.json()) as {
    action?: string;
    filename?: string;
  };
  const { action, filename } = body;
  if (
    !filename ||
    (action !== "approve" && action !== "reject") ||
    path.basename(filename) !== filename ||
    !filename.endsWith(".json")
  ) {
    return NextResponse.json({ error: "invalid request" }, { status: 400 });
  }
  const src = path.join(PENDING_DIR, filename);
  try {
    const raw = await fs.readFile(src, "utf8");
    const record = {
      ...JSON.parse(raw),
      status: action === "approve" ? "approved" : "rejected",
      decided_by: "human",
      decided_at: new Date().toISOString(),
    };
    const destDir = action === "approve" ? APPROVALS_DIR : REJECTED_DIR;
    await fs.mkdir(destDir, { recursive: true });
    await fs.writeFile(
      path.join(destDir, filename),
      JSON.stringify(record, null, 2),
    );
    await fs.unlink(src);
    return NextResponse.json({ ok: true, filename });
  } catch {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
}