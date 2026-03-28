package com.kama.jchatmind.service.impl;

import com.kama.jchatmind.mapper.ChunkBgeM3Mapper;
import com.kama.jchatmind.model.entity.ChunkBgeM3;
import com.kama.jchatmind.service.RagService;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
@Slf4j
public class RagServiceImpl implements RagService {
    private static final double MIN_CONFIDENCE = 0.60d;

    private final WebClient webClient;
    private final ChunkBgeM3Mapper chunkBgeM3Mapper;

    public RagServiceImpl(WebClient.Builder builder, ChunkBgeM3Mapper chunkBgeM3Mapper) {
        this.webClient = builder.baseUrl("http://localhost:11434").build();
        this.chunkBgeM3Mapper = chunkBgeM3Mapper;
    }

    @Data
    private static class EmbeddingResponse {
        private float[] embedding;
    }

    private float[] doEmbed(String text) {
        EmbeddingResponse resp = webClient.post()
                .uri("/api/embeddings")
                .bodyValue(Map.of(
                        "model", "bge-m3",
                        "prompt", text
                ))
                .retrieve()
                .bodyToMono(EmbeddingResponse.class)
                .block();

        Assert.notNull(resp, "Embedding response cannot be null");
        return resp.getEmbedding();
    }

    @Override
    public float[] embed(String text) {
        return doEmbed(text);
    }

    @Override
    public List<String> similaritySearch(String kbId, String title) {
        float[] queryVector = doEmbed(title);
        String queryEmbedding = toPgVector(queryVector);
        try {
            List<ChunkBgeM3> chunks = chunkBgeM3Mapper.similaritySearch(kbId, queryEmbedding, 3);
            return filterByConfidence(chunks, queryVector);
        } catch (Exception e) {
            log.warn("RAG similaritySearch failed: kbId={}, error={}", kbId, e.getMessage());
            return List.of();
        }
    }

    @Override
    public List<String> similaritySearchAllKnowledgeBases(String title) {
        float[] queryVector = doEmbed(title);
        String queryEmbedding = toPgVector(queryVector);
        try {
            List<ChunkBgeM3> chunks = chunkBgeM3Mapper.similaritySearchAllKnowledgeBases(queryEmbedding, 5);
            return filterByConfidence(chunks, queryVector);
        } catch (Exception e) {
            log.warn("RAG similaritySearchAllKnowledgeBases failed: error={}", e.getMessage());
            return List.of();
        }
    }

    private List<String> filterByConfidence(List<ChunkBgeM3> chunks, float[] queryVector) {
        if (chunks == null || chunks.isEmpty() || queryVector == null || queryVector.length == 0) {
            return List.of();
        }
        List<String> result = new ArrayList<>();
        for (ChunkBgeM3 chunk : chunks) {
            if (chunk == null || chunk.getEmbedding() == null || chunk.getContent() == null) {
                continue;
            }
            double confidence = cosineSimilarity(queryVector, chunk.getEmbedding());
            if (confidence >= MIN_CONFIDENCE) {
                result.add(chunk.getContent());
            }
        }
        return result;
    }

    private double cosineSimilarity(float[] a, float[] b) {
        if (a == null || b == null || a.length == 0 || a.length != b.length) {
            return -1.0d;
        }
        double dot = 0.0d;
        double normA = 0.0d;
        double normB = 0.0d;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0.0d || normB == 0.0d) {
            return -1.0d;
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private String toPgVector(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            sb.append(v[i]);
            if (i < v.length - 1) sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }
}
