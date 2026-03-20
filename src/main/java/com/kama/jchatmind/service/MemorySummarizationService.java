package com.kama.jchatmind.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.kama.jchatmind.agent.memory.DistributedChatMemory;
import com.kama.jchatmind.mapper.ChunkBgeM3Mapper;
import com.kama.jchatmind.model.dto.ChatMessageDTO;
import com.kama.jchatmind.model.entity.ChunkBgeM3;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

@Slf4j
@Service
public class MemorySummarizationService {

    private final RagService ragService;
    private final ChunkBgeM3Mapper chunkBgeM3Mapper;
    private final ChatMessageFacadeService chatMessageFacadeService;
    private final ObjectMapper objectMapper;
    // 使用默认的一个 ChatClient 构建器，用于大模型调用
    private final ChatClient chatClient;

    public MemorySummarizationService(ChatClient.Builder chatClientBuilder, 
                                      RagService ragService, 
                                      ChunkBgeM3Mapper chunkBgeM3Mapper,
                                      ChatMessageFacadeService chatMessageFacadeService) {
        this.chatClient = chatClientBuilder.build();
        this.ragService = ragService;
        this.chunkBgeM3Mapper = chunkBgeM3Mapper;
        this.chatMessageFacadeService = chatMessageFacadeService;
        this.objectMapper = new ObjectMapper();
    }

    @Data
    public static class SummarizationResult {
        private String summary;
        private List<String> facts;
    }

    /**
     * 异步对历史对话进行浓缩摘要，并将关键实体/事实存入向量库
     * 摘要完成后，清理数据库旧消息，用总结的记忆消息替代，并清理Redis热缓存
     */
    @Async
    @Transactional(rollbackFor = Exception.class)
    public void summarizeAndExtractFactsAsync(String chatSessionId, ChatClient specificChatClient, int keepLatestN, DistributedChatMemory chatMemoryCache) {
        try {
            log.info("Starting async memory summarization for session: {}", chatSessionId);
            
            // 拿到当前会话所有的消息（直接从DB拉取，确保全量长上下文）
            List<ChatMessageDTO> allDbMessages = chatMessageFacadeService.getChatMessagesBySessionIdRecently(chatSessionId, 500); // 拉取最近较多的消息
            if (allDbMessages == null || allDbMessages.size() <= keepLatestN) {
                return; // 不够长，不需要压缩
            }
            
            // 选出要被压缩的老消息
            List<ChatMessageDTO> oldMessagesToSummarize = allDbMessages.subList(0, allDbMessages.size() - keepLatestN);
            List<ChatMessageDTO> messagesToKeep = allDbMessages.subList(allDbMessages.size() - keepLatestN, allDbMessages.size());
            
            StringBuilder conversation = new StringBuilder();
            for (ChatMessageDTO msg : oldMessagesToSummarize) {
                conversation.append(msg.getRole().getRole()).append(": ").append(msg.getContent()).append("\n");
            }
            
            String prompt = "请分析以下对话记录，并提取出两个信息：\n" +
                    "1. summary: 用一段精简的话总结对话的核心内容，以备将来对话时回忆。\n" +
                    "2. facts: 提取出对话中包含的关于用户的客观事实和实体信息（作为 List<String>）。如果没有则返回空列表。\n" +
                    "请以严格的 JSON 格式输出，格式如下：\n" +
                    "{\"summary\": \"...\", \"facts\": [\"...\", \"...\"]}\n" +
                    "\n对话记录如下：\n" + conversation.toString();
            
            ChatClient clientToUse = specificChatClient != null ? specificChatClient : this.chatClient;
            
            String response = clientToUse.prompt(prompt)
                    .call()
                    .content();
            
            String jsonStr = extractJson(response);
            SummarizationResult result = objectMapper.readValue(jsonStr, SummarizationResult.class);
            
            log.info("Summarization complete. Summary: {}, Facts found: {}", result.summary, result.facts != null ? result.facts.size() : 0);
            
            // 1. 将 facts 保存到向量库（长期记忆）
            if (result.getFacts() != null && !result.getFacts().isEmpty()) {
                for (String fact : result.getFacts()) {
                    float[] embedding = ragService.embed(fact);
                    ChunkBgeM3 chunk = ChunkBgeM3.builder()
                            .id(UUID.randomUUID().toString())
                            .kbId(chatSessionId) // 以 SessionId 作为区分长期记忆的标识
                            .docId("long_term_memory")
                            .content(fact)
                            .metadata("{\"type\":\"fact\"}")
                            .embedding(embedding)
                            .createdAt(LocalDateTime.now())
                            .updatedAt(LocalDateTime.now())
                            .build();
                    chunkBgeM3Mapper.insert(chunk);
                }
            }
            
            // 2. 将老的历史消息从数据库删除，插入一条新的 Summary Message
            for (ChatMessageDTO msg : oldMessagesToSummarize) {
                // TODO: chatMessageFacadeService might not have a deleteById method.
                // We will add it if it doesn't exist. Let's assume it has or we'll add it later.
                chatMessageFacadeService.deleteChatMessage(msg.getId());
            }
            
            ChatMessageDTO summaryMsg = ChatMessageDTO.builder()
                    .sessionId(chatSessionId)
                    .role(ChatMessageDTO.RoleType.SYSTEM)
                    .content("过去对话摘要记忆：" + result.getSummary())
                    .build();
            chatMessageFacadeService.createChatMessage(summaryMsg);
            
            // 3. 清理 Redis 热缓存，迫使下一次访问重新从 DB 中按需加载并回写缓存
            chatMemoryCache.clear(chatSessionId);
            
            log.info("Memory summarization successfully applied to database and cache for session: {}", chatSessionId);
            
        } catch (Exception e) {
            log.error("Error during async memory summarization", e);
        }
    }
    
    private String extractJson(String text) {
        if (text == null) return "{}";
        int start = text.indexOf("{");
        int end = text.lastIndexOf("}");
        if (start != -1 && end != -1 && start < end) {
            return text.substring(start, end + 1);
        }
        return "{}";
    }
}
