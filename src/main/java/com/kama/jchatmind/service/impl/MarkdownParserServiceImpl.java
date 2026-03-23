package com.kama.jchatmind.service.impl;

import com.kama.jchatmind.service.MarkdownParserService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Service
@Slf4j
public class MarkdownParserServiceImpl implements MarkdownParserService {

    private static final Pattern HEADING_PATTERN = Pattern.compile("^\\s{0,3}(#{1,6})\\s+(.+?)\\s*$");
    private static final Pattern TRAILING_HASH_PATTERN = Pattern.compile("\\s+#+\\s*$");

    private static class HeadingMarker {
        private final int lineIndex;
        private final int level;
        private final String title;

        private HeadingMarker(int lineIndex, int level, String title) {
            this.lineIndex = lineIndex;
            this.level = level;
            this.title = title;
        }
    }

    @Override
    public List<MarkdownSection> parseMarkdown(InputStream inputStream) {
        try {
            String markdown = new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
            List<MarkdownSection> sections = extractSections(markdown);
            log.info("Markdown parsed successfully, sections={}", sections.size());
            return sections;
        } catch (Exception e) {
            log.error("Failed to parse markdown", e);
            throw new RuntimeException("Failed to parse markdown: " + e.getMessage(), e);
        }
    }

    private List<MarkdownSection> extractSections(String markdown) {
        String[] lines = markdown.split("\\r?\\n", -1);
        List<HeadingMarker> headings = collectHeadings(lines);
        List<MarkdownSection> sections = new ArrayList<>();

        for (int i = 0; i < headings.size(); i++) {
            HeadingMarker current = headings.get(i);
            int endLine = lines.length - 1;

            for (int j = i + 1; j < headings.size(); j++) {
                HeadingMarker next = headings.get(j);
                if (next.level <= current.level) {
                    endLine = next.lineIndex - 1;
                    break;
                }
            }

            String content = joinLines(lines, current.lineIndex, endLine).trim();
            sections.add(new MarkdownSection(current.title, content));
        }

        return sections;
    }

    private List<HeadingMarker> collectHeadings(String[] lines) {
        List<HeadingMarker> headings = new ArrayList<>();
        boolean inCodeFence = false;

        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            String trimmed = line.trim();

            if (trimmed.startsWith("```") || trimmed.startsWith("~~~")) {
                inCodeFence = !inCodeFence;
                continue;
            }
            if (inCodeFence) {
                continue;
            }

            Matcher matcher = HEADING_PATTERN.matcher(line);
            if (!matcher.matches()) {
                continue;
            }

            int level = matcher.group(1).length();
            String rawTitle = matcher.group(2);
            String title = TRAILING_HASH_PATTERN.matcher(rawTitle).replaceAll("").trim();
            if (!title.isBlank()) {
                headings.add(new HeadingMarker(i, level, title));
            }
        }

        return headings;
    }

    private String joinLines(String[] lines, int fromInclusive, int toInclusive) {
        if (fromInclusive < 0 || toInclusive < fromInclusive || fromInclusive >= lines.length) {
            return "";
        }
        int end = Math.min(toInclusive, lines.length - 1);
        StringBuilder sb = new StringBuilder();
        for (int i = fromInclusive; i <= end; i++) {
            if (sb.length() > 0) {
                sb.append("\n");
            }
            sb.append(lines[i]);
        }
        return sb.toString();
    }
}
