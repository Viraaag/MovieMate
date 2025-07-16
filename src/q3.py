def solution(paragraphs, aligns, width):
    result_lines = []
# Helper: align a line string according to align type
    def align_line(line_str, align):
        spaces_needed = width - len(line_str)
    if align == "LEFT":
        return line_str +' ' * spaces_needed
    else: # RIGHT
        return ' ' * spaces_needed + line_str
    # Process each paragraph
    for para, align in zip(paragraphs, aligns):
        current_line_words = []
        current_len = 0
    for word in para:
    # If adding word (and space if needed) exceeds
        added_len = len(word) if not
        current_line_words else len(word) + 1
        if current_len + added_len > width:
    # Finalize current line
            line_str = ''.join(current_line_words)
            result_lines.append(align_line(line_str,
            align))
    # Start new line
        current_line_words = [word]
        current_len = len(word)
    else:
        if current_line_words:
        current_len += 1 # for space
        current_line_words.append(word)
        current_len += len(word)
    # Any remaining words
    if current_line_words:
        line_str = ''.join(current_line_words)
        result_lines.append(align_line(line_str, align))
    # Add border
    border = '*' * (width + 2)
    final_result = [border]
    for line in result_lines:
    final_result.append(f"*{line}*")
    final_result.append(border)
    return final_result