"use client";

import type {
  Attachment,
  AttachmentAdapter,
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";

/**
 * File types accepted by the composer (mirrors the old use-file-upload hook):
 * JPEG/PNG/GIF/WEBP images sent as LangChain `image_url` content blocks, and
 * PDFs sent as base64 `file` content blocks.
 */
export const SUPPORTED_ATTACHMENT_TYPES = [
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
  "application/pdf",
] as const;

const SUPPORTED = new Set<string>(SUPPORTED_ATTACHMENT_TYPES);

const readAsDataUrl = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

const signatureOf = (file: File) =>
  `${file.name}::${file.type}::${file.size}`;

/**
 * Attachment adapter matching the old `use-file-upload` behavior.
 *
 * - Images: converted to a base64 data URL and sent as an `image` message
 *   part (the react-langgraph wire converter turns it into a LangChain
 *   `image_url` content block).
 * - PDFs: converted to base64 and sent as a `file` message part (LangChain
 *   `file` content block, `source_type: "base64"`).
 * - Unsupported types are rejected in `add()` (surfaced to the UI through the
 *   `composer.attachmentAddError` event as `adapter-error`).
 * - Per-message duplicates are blocked (each file can be attached once per
 *   message), matching the old toast behavior.
 *
 * Wired into `useLangGraphRuntime` as `adapters.attachments` (see
 * `MyAssistant.tsx`). The adapter instance is shared across threads, so the
 * duplicate signature set is scoped to files still pending in a composer.
 */
export class MultimodalAttachmentAdapter implements AttachmentAdapter {
  accept = SUPPORTED_ATTACHMENT_TYPES.join(",");

  private pendingSignatures = new Set<string>();

  add({ file }: { file: File }): Promise<PendingAttachment> {
    if (!SUPPORTED.has(file.type)) {
      return Promise.reject(
        new Error(
          `Unsupported file type: ${file.type}. Please upload a JPEG, PNG, GIF, WEBP image or a PDF.`,
        ),
      );
    }
    const signature = signatureOf(file);
    if (this.pendingSignatures.has(signature)) {
      return Promise.reject(
        new Error(
          `Duplicate file(s) detected: ${file.name}. Each file can only be uploaded once per message.`,
        ),
      );
    }
    this.pendingSignatures.add(signature);
    return Promise.resolve({
      id: crypto.randomUUID(),
      type: file.type === "application/pdf" ? "document" : "image",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "requires-action", reason: "composer-send" },
    });
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const { file } = attachment;
    this.pendingSignatures.delete(signatureOf(file));
    const dataUrl = await readAsDataUrl(file);
    const base64 = dataUrl.split(",")[1] ?? "";
    const isPdf = file.type === "application/pdf";
    return {
      ...attachment,
      status: { type: "complete" },
      content: isPdf
        ? [
            {
              type: "file",
              filename: file.name,
              data: base64,
              mimeType: file.type,
            },
          ]
        : [
            {
              type: "image",
              image: dataUrl,
              filename: file.name,
            },
          ],
    };
  }

  async remove(attachment: Attachment): Promise<void> {
    if (attachment.file) {
      this.pendingSignatures.delete(signatureOf(attachment.file));
    }
  }
}

/** Shared singleton — the composer adapter is stateless across threads apart
 *  from the pending-signature dedupe set, so one instance is safe. */
export const multimodalAttachmentAdapter = new MultimodalAttachmentAdapter();