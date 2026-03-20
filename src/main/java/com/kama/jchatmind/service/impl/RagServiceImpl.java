package com.kama.jchatmind.service.impl;

import com.kama.jchatmind.mapper.ChunkBgeM3Mapper;
import com.kama.jchatmind.model.entity.ChunkBgeM3;
import com.kama.jchatmind.service.RagService;
import lombok.Data;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;
import java.util.Map;

@Service
public class RagServiceImpl implements RagService {

    // 封装本地的模型调用
    private final WebClient webClient;
    private final ChunkBgeM3Mapper chunkBgeM3Mapper;

    // 构造函数, 自动注入一个 WebClient.Builder 和 ChunkBgeM3Mapper
    public RagServiceImpl(WebClient.Builder builder, ChunkBgeM3Mapper chunkBgeM3Mapper) {
        this.webClient = builder.baseUrl("http://localhost:11434").build();
        this.chunkBgeM3Mapper = chunkBgeM3Mapper;
    }

    // 是用来接收 HTTP 接口返回结果的
    @Data
    private static class EmbeddingResponse {
        private float[] embedding;
    }

    private float[] doEmbed(String text) {
        EmbeddingResponse resp = webClient.post()
                .uri("/api/embeddings") // 就是在baseurl的基础上再拼接一个uri
                .bodyValue(Map.of(
                        "model", "bge-m3",
                        "prompt", text
                ))
                .retrieve() //表示开始处理服务端返回的响应体。
                .bodyToMono(EmbeddingResponse.class) //表示“未来会返回一个 EmbeddingResponse”。
                .block(); //把响应式调用强制转成同步调用
        Assert.notNull(resp, "Embedding response cannot be null");
        return resp.getEmbedding();
    }

    @Override
    // 是把内部的 doEmbed 暴露出去。因为doEmbed有接口的具体实现，直接暴露不太合适
    public float[] embed(String text) {
        return doEmbed(text);
    }

    @Override
    // 查询title对应的最相似的向量，并返回对应的content
    public List<String> similaritySearch(String kbId, String title) {
        String queryEmbedding = toPgVector(doEmbed(title));  //把 Java 的 float[] 转成 PostgreSQL 向量字符串格式。
        List<ChunkBgeM3> chunks = chunkBgeM3Mapper.similaritySearch(kbId, queryEmbedding, 3); //用postgres的方法去查询最相近的3个记录
        return chunks.stream().map(ChunkBgeM3::getContent).toList(); // 将content转换为一个List返回
    }

    // 将float[] = {0.1f, 0.2f, 0.3f}转换为[0.1,0.2,0.3]
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
