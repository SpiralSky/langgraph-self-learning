from langchain_core.messages import BaseMessage


def message_to_text(message: BaseMessage) -> str:
    """
    Extract the plain-text content of a message.

    Handles both ``str`` content and openai-style block lists, mirroring the
    extraction shared by ``retrieve_memory``, ``save_memory``, and
    ``response_builder``.

    Block-list handling iterates each block: dicts with ``"type": "text"``
    contribute their ``"text"`` value; objects with a ``.text`` attribute
    contribute that attribute. Everything is joined with spaces.

    :param message: Conversation message to serialize.
    :type message: BaseMessage
    :return: The message text as a single string.
    :rtype: str
    """
    match message.content:
        case str(text):
            return text

        case list(content_blocks):
            text_parts = []
            for block in content_blocks:
                match block:
                    case {"type": "text", "text": str(text)}:
                        text_parts.append(text)
                    case _ if hasattr(block, "text"):
                        text_parts.append(block.text)
            return " ".join(text_parts)

        case _:
            return str(message.content)