import { PendingApprovals } from "@/components/approvals/pending-approvals";

export default function ApprovalsPage() {
  return (
    <main className="flex min-h-screen flex-col items-center gap-4 bg-background p-8">
      <h1 className="text-2xl font-semibold tracking-tight">Approvals</h1>
      <PendingApprovals />
    </main>
  );
}