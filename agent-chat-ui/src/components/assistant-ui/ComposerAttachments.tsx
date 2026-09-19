"use client";

import {
  AttachmentPrimitive,
  ComposerPrimitive,
  useAuiEvent,
  useAuiState,
} from "@assistant-ui/react";
import { FileIcon, XIcon } from "lucide-react";
import { useEffect, useState } from "react";
import type { Attachment } from "@assistant-ui/react";
import { toast } from "sonner";

/**
 * Preview source for a composer attachment: a local object URL while the
 * pending file is in the composer, falling back to the completed part's
 * `image` data URL (e.g. attachments restored from a sent message).
 *
 * Mirrors the assistant-ui reference `use-attachment-src` hook: the object
 * URL's lifetime straddles commit, so allocation, revocation and state all
 * live in the effect (the sync setState branches are the reference impl's own
 * pattern).
 */
function useAttachmentSrc(attachment: Attachment) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!attachment.file) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setObjectUrl(null);
      return;
    }
    const url = URL.createObjectURL(attachment.file);
    setObjectUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [attachment.file]);

  if (objectUrl) return objectUrl;
  const image = attachment.content?.find((part) => part.type === "image");
  return image && image.type === "image" ? image.image : null;
}

function AttachmentItem({ attachment }: { attachment: Attachment }) {
  const src = useAttachmentSrc(attachment);
  const isImage =
    attachment.type === "image" ||
    attachment.content?.some((part) => part.type === "image");
  const isPdf =
    attachment.contentType === "application/pdf" ||
    attachment.content?.some(
      (part) => part.type === "file" && part.mimeType === "application/pdf",
    );

  return (
    <AttachmentPrimitive.Root className="group flex items-center gap-2 rounded-lg border border-input bg-muted/60 py-1 pl-1.5 pr-1 text-xs">
      {isImage && src ? (
        <img
          src={src}
          alt={attachment.name}
          className="h-8 w-8 shrink-0 rounded object-cover"
        />
      ) : (
        <FileIcon
          className={`size-4 shrink-0 ${isPdf ? "text-red-500" : "text-muted-foreground"}`}
        />
      )}
      <span className="max-w-40 truncate">{attachment.name}</span>
      <AttachmentPrimitive.Remove
        aria-label={`Remove ${attachment.name}`}
        className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <XIcon className="size-3.5" />
      </AttachmentPrimitive.Remove>
    </AttachmentPrimitive.Root>
  );
}

/**
 * Renders the composer's attached files (images with thumbnails, PDFs with a
 * file chip) plus the remove button. Place inside `ComposerPrimitive.Root` —
 * the primitive provides the per-index attachment context. Renders nothing
 * while the composer has no attachments.
 */
export function ComposerAttachments() {
  const count = useAuiState((state) => state.composer.attachments.length);
  if (count === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 pb-2">
      <ComposerPrimitive.Attachments>
        {({ attachment }) => <AttachmentItem attachment={attachment} />}
      </ComposerPrimitive.Attachments>
    </div>
  );
}

/**
 * Surfaces attachment failures (invalid file type, duplicate, adapter error)
 * as toasts, mirroring the old use-file-upload toasts. Mount inside the
 * `AssistantRuntimeProvider`.
 */
export function AttachmentAddErrorToast() {
  useAuiEvent("composer.attachmentAddError", ({ reason, message }) => {
    if (reason === "not-accepted") {
      toast.error(
        "Unsupported file type. Please upload a JPEG, PNG, GIF, WEBP image or a PDF.",
      );
    } else if (reason === "adapter-error") {
      toast.error(message);
    }
  });
  return null;
}