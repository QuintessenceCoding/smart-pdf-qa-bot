def split_text(text, max_words=300, overlap=50):
    """
    Splits text into smaller chunks by paragraphs and word count with overlap.

    Args:
        text (str): The full text to split.
        max_words (int): Max number of words per chunk.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    paragraphs = text.split('\n\n')  # split by paragraph

    chunks = []
    current_chunk_words = []

    for para in paragraphs:
        words = para.split()
        if not words:
            continue

        # If adding this paragraph exceeds max_words, create a chunk and start a new one
        if len(current_chunk_words) + len(words) > max_words:
            chunks.append(' '.join(current_chunk_words))
            # Start next chunk with overlap words
            current_chunk_words = current_chunk_words[-overlap:] if overlap < len(current_chunk_words) else current_chunk_words
        current_chunk_words.extend(words)

    # Add the last chunk
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))

    return chunks
