package com.kama.jchatmind.agent.tools;

import com.kama.jchatmind.mapper.ChunkBgeM3Mapper;
import com.kama.jchatmind.model.entity.ChunkBgeM3;
import com.kama.jchatmind.service.RagService;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class KnowledgeTools implements Tool {

    private static final int TOP_K = 3;
    private static final int DENSE_CANDIDATE_LIMIT = 24;
    private static final int SPARSE_CANDIDATE_LIMIT = 24;
    private static final int RRF_K = 60;
    private static final double CONFIDENCE_THRESHOLD = 0.60;
    private static final Pattern TOKEN_PATTERN = Pattern.compile("[\\p{L}\\p{N}_\\-.]+");

    private final RagService ragService;
    private final ChunkBgeM3Mapper chunkBgeM3Mapper;

    public KnowledgeTools(RagService ragService, ChunkBgeM3Mapper chunkBgeM3Mapper) {
        this.ragService = ragService;
        this.chunkBgeM3Mapper = chunkBgeM3Mapper;
    }

    @Override
    public String getName() {
        return "KnowledgeTool";
    }

    @Override
    public String getDescription() {
        return "Retrieve multiple relevant snippets from all knowledge bases with provenance metadata.";
    }

    @Override
    public ToolType getType() {
        return ToolType.FIXED;
    }

    @org.springframework.ai.tool.annotation.Tool(
            name = "KnowledgeTool",
            description = "Retrieve multiple relevant snippets from all knowledge bases. kbsId is ignored; use query."
    )
    public String knowledgeQuery(String kbsId, String query) {
        if (query == null || query.isBlank()) {
            return "No query provided.";
        }

        float[] embedding = ragService.embed(query);
        String vectorLiteral = toPgVector(embedding);
        List<String> keywords = extractKeywords(query);
        QueryMode mode = detectQueryMode(query, keywords);
        double denseWeight = mode == QueryMode.PRECISE ? 0.35 : 0.65;
        double sparseWeight = mode == QueryMode.PRECISE ? 0.65 : 0.35;

        List<ChunkBgeM3> denseChunks = chunkBgeM3Mapper.similaritySearchAllKnowledgeBases(vectorLiteral, DENSE_CANDIDATE_LIMIT);
        List<ChunkBgeM3> sparseChunks = keywords.isEmpty()
                ? List.of()
                : chunkBgeM3Mapper.keywordSearchAllKnowledgeBases(keywords, SPARSE_CANDIDATE_LIMIT);

        List<ScoredChunk> chunks = rankHybridCandidates(denseChunks, sparseChunks, keywords, denseWeight, sparseWeight, TOP_K);

        if (chunks == null || chunks.isEmpty()) {
            return "No relevant knowledge found across all knowledge bases.";
        }

        List<ScoredChunk> accepted = chunks.stream()
                .filter(scored -> scored != null && scored.confidence >= CONFIDENCE_THRESHOLD)
                .toList();

        if (accepted.isEmpty()) {
            accepted = chunks.stream().limit(1).toList();
        }

        List<ChunkBgeM3> valid = new ArrayList<>();
        for (ScoredChunk scored : accepted) {
            ChunkBgeM3 chunk = scored.chunk;
            if (chunk != null && chunk.getContent() != null && !chunk.getContent().trim().isEmpty()) {
                valid.add(chunk);
            }
        }

        if (valid.isEmpty()) {
            return "No relevant knowledge found across all knowledge bases.";
        }

        StringBuilder sb = new StringBuilder();
        for (ChunkBgeM3 chunk : valid) {
            String snippet = buildFocusedSnippet(chunk.getContent(), keywords);
            if (snippet.isBlank()) {
                continue;
            }
            if (sb.length() > 0) {
                sb.append("\n\n");
            }
            sb.append(snippet);
        }

        if (sb.length() == 0) {
            return "No relevant knowledge found across all knowledge bases.";
        }
        return sb.toString().trim();
    }

    private String toPgVector(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            sb.append(v[i]);
            if (i < v.length - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    private List<String> extractKeywords(String query) {
        Matcher matcher = TOKEN_PATTERN.matcher(query.toLowerCase(Locale.ROOT));
        List<String> tokens = new ArrayList<>();
        while (matcher.find()) {
            String token = matcher.group().trim();
            if (token.length() >= 2) {
                tokens.add(token);
            }
        }
        return tokens;
    }

    private QueryMode detectQueryMode(String query, List<String> keywords) {
        String normalized = query == null ? "" : query.trim();
        int len = normalized.length();
        boolean shortQuery = len <= 18 || keywords.size() <= 4;
        boolean codeLike = false;
        for (String kw : keywords) {
            if (kw.contains("-") || kw.contains("_") || kw.contains(".") || hasMixedAlphaNumeric(kw) || isLikelyAcronym(kw)) {
                codeLike = true;
                break;
            }
        }
        if (shortQuery && codeLike) {
            return QueryMode.PRECISE;
        }
        if (shortQuery && (normalized.startsWith("\u4ec0\u4e48\u662f") || normalized.toLowerCase(Locale.ROOT).startsWith("what is"))) {
            return QueryMode.PRECISE;
        }
        return QueryMode.SEMANTIC;
    }

    private boolean hasMixedAlphaNumeric(String token) {
        boolean hasAlpha = false;
        boolean hasDigit = false;
        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            if (Character.isLetter(c)) {
                hasAlpha = true;
            }
            if (Character.isDigit(c)) {
                hasDigit = true;
            }
        }
        return hasAlpha && hasDigit;
    }

    private boolean isLikelyAcronym(String token) {
        if (token.length() < 2 || token.length() > 10) {
            return false;
        }
        int alphaCount = 0;
        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            if (Character.isLetter(c)) {
                alphaCount++;
            } else if (!Character.isDigit(c)) {
                return false;
            }
        }
        return alphaCount >= 2;
    }

    private List<ScoredChunk> rankHybridCandidates(
            List<ChunkBgeM3> dense,
            List<ChunkBgeM3> sparse,
            List<String> keywords,
            double denseWeight,
            double sparseWeight,
            int topK) {
        Map<String, ChunkBgeM3> byId = new HashMap<>();
        Map<String, Double> scoreById = new HashMap<>();
        Map<String, Integer> denseRankById = new HashMap<>();
        Map<String, Integer> sparseRankById = new HashMap<>();

        accumulateRrf(dense, denseWeight, byId, scoreById, denseRankById);
        accumulateRrf(sparse, sparseWeight, byId, scoreById, sparseRankById);

        List<ScoredChunk> ranked = scoreById.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .map(entry -> {
                    String id = entry.getKey();
                    ChunkBgeM3 chunk = byId.get(id);
                    double confidence = estimateConfidence(
                            chunk,
                            keywords,
                            denseRankById.get(id),
                            sparseRankById.get(id),
                            denseWeight,
                            sparseWeight
                    );
                    return new ScoredChunk(chunk, confidence);
                })
                .sorted((a, b) -> Double.compare(b.confidence, a.confidence))
                .limit(topK)
                .toList();

        return ranked;
    }

    private void accumulateRrf(
            List<ChunkBgeM3> chunks,
            double weight,
            Map<String, ChunkBgeM3> byId,
            Map<String, Double> scoreById,
            Map<String, Integer> rankById) {
        if (chunks == null || chunks.isEmpty()) {
            return;
        }
        int rank = 1;
        for (ChunkBgeM3 chunk : chunks) {
            if (chunk == null || chunk.getId() == null || chunk.getId().isBlank()) {
                rank++;
                continue;
            }
            String id = chunk.getId();
            byId.putIfAbsent(id, chunk);
            rankById.putIfAbsent(id, rank);
            double score = weight / (RRF_K + rank);
            scoreById.put(id, scoreById.getOrDefault(id, 0.0) + score);
            rank++;
        }
    }

    private double estimateConfidence(
            ChunkBgeM3 chunk,
            List<String> keywords,
            Integer denseRank,
            Integer sparseRank,
            double denseWeight,
            double sparseWeight) {
        if (chunk == null || chunk.getContent() == null) {
            return 0.0;
        }
        String text = chunk.getContent().toLowerCase(Locale.ROOT);

        double denseScore = rankToScore(denseRank);
        double sparseScore = rankToScore(sparseRank);
        double retrievalScore = denseWeight * denseScore + sparseWeight * sparseScore;

        int matched = 0;
        for (String kw : keywords) {
            if (text.contains(kw.toLowerCase(Locale.ROOT))) {
                matched++;
            }
        }
        double keywordCoverage = keywords.isEmpty() ? 0.0 : (double) matched / keywords.size();

        int headingLevel = detectHeadingLevel(chunk.getContent());
        double granularityBonus = headingLevel >= 2 ? 0.18 : 0.0;

        double confidence = 0.55 * retrievalScore + 0.45 * keywordCoverage + granularityBonus;
        return Math.max(0.0, Math.min(1.0, confidence));
    }

    private double rankToScore(Integer rank) {
        if (rank == null) {
            return 0.0;
        }
        return 1.0 / (1.0 + Math.log10(rank + 1.0));
    }

    private int detectHeadingLevel(String content) {
        if (content == null || content.isBlank()) {
            return 99;
        }
        String[] lines = content.split("\\r?\\n", 2);
        if (lines.length == 0) {
            return 99;
        }
        String first = lines[0].trim();
        int level = 0;
        while (level < first.length() && first.charAt(level) == '#') {
            level++;
        }
        if (level >= 1 && level <= 6) {
            return level;
        }
        return 99;
    }

    private String buildFocusedSnippet(String content, List<String> keywords) {
        if (content == null || content.isBlank()) {
            return "";
        }
        String text = content.trim();
        if (keywords == null || keywords.isEmpty()) {
            return clip(text, 900);
        }
        String lower = text.toLowerCase(Locale.ROOT);
        int bestIdx = -1;
        for (String kw : keywords) {
            int idx = lower.indexOf(kw.toLowerCase(Locale.ROOT));
            if (idx >= 0 && (bestIdx < 0 || idx < bestIdx)) {
                bestIdx = idx;
            }
        }
        if (bestIdx < 0) {
            return clip(text, 900);
        }
        int start = Math.max(0, bestIdx - 220);
        int end = Math.min(text.length(), bestIdx + 520);
        String excerpt = text.substring(start, end).trim();
        if (start > 0) {
            excerpt = "..." + excerpt;
        }
        if (end < text.length()) {
            excerpt = excerpt + "...";
        }
        return excerpt;
    }

    private String clip(String text, int maxLen) {
        if (text.length() <= maxLen) {
            return text;
        }
        return text.substring(0, maxLen) + "...";
    }

    private enum QueryMode {
        PRECISE,
        SEMANTIC
    }

    private static class ScoredChunk {
        private final ChunkBgeM3 chunk;
        private final double confidence;

        private ScoredChunk(ChunkBgeM3 chunk, double confidence) {
            this.chunk = chunk;
            this.confidence = confidence;
        }
    }
}
