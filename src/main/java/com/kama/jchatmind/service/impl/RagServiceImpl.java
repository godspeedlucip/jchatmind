package com.kama.jchatmind.service.impl;

import com.kama.jchatmind.mapper.ChunkBgeM3Mapper;
import com.kama.jchatmind.model.entity.ChunkBgeM3;
import com.kama.jchatmind.service.RagService;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;
import java.util.Map;

@Service
@Slf4j
public class RagServiceImpl implements RagService {

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
        String queryEmbedding = toPgVector(doEmbed(title));
        try {
            List<ChunkBgeM3> chunks = chunkBgeM3Mapper.similaritySearch(kbId, queryEmbedding, 3);
            return chunks.stream().map(ChunkBgeM3::getContent).toList();
        } catch (Exception e) {
            log.warn("RAG similaritySearch failed: kbId={}, error={}", kbId, e.getMessage());
            return List.of();
        }
    }

    @Override
    public List<String> similaritySearchAllKnowledgeBases(String title) {
        String queryEmbedding = toPgVector(doEmbed(title));
        try {
            List<ChunkBgeM3> chunks = chunkBgeM3Mapper.similaritySearchAllKnowledgeBases(queryEmbedding, 5);
            return chunks.stream().map(ChunkBgeM3::getContent).toList();
        } catch (Exception e) {
            log.warn("RAG similaritySearchAllKnowledgeBases failed: error={}", e.getMessage());
            return List.of();
        }
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
