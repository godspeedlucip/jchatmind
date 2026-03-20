package com.kama.jchatmind.agent.memory;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.jsontype.impl.LaissezFaireSubTypeValidator;
import com.kama.jchatmind.model.dto.ChatMessageDTO;
import com.kama.jchatmind.service.ChatMessageFacadeService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.*;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * 分布式会话存放：将会话历史持久化转移至 Redis 做热缓存 + PostgreSQL 做冷备
 */
public class DistributedChatMemory implements ChatMemory {

    private static final Logger log = LoggerFactory.getLogger(DistributedChatMemory.class);

    private final RedisTemplate<String, Object> redisTemplate;
    private final ChatMessageFacadeService chatMessageFacadeService;
    private final int maxMessages;
    private final ObjectMapper objectMapper;

    private static final String KEY_PREFIX = "chat:memory:";
    private static final long CACHE_TTL_HOURS = 24; // 热缓存暂设为 24 小时过期

    public DistributedChatMemory(RedisTemplate<String, Object> redisTemplate, 
                                 ChatMessageFacadeService chatMessageFacadeService, 
                                 int maxMessages) {
        this.redisTemplate = redisTemplate;
        this.chatMessageFacadeService = chatMessageFacadeService;
        this.maxMessages = maxMessages;
        
        // 用于 Redis 中序列化和反序列化 MessageDTO
        this.objectMapper = new ObjectMapper();
        this.objectMapper.activateDefaultTyping(LaissezFaireSubTypeValidator.instance, ObjectMapper.DefaultTyping.NON_FINAL, JsonTypeInfo.As.PROPERTY);
    }

    private String getCacheKey(String conversationId) {
        return KEY_PREFIX + conversationId;
    }

    @Override
    public void add(String conversationId, List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return;
        }

        String key = getCacheKey(conversationId);
        
        // 1. 将现有的历史记录覆盖到 Redis 中
        // 注意：因为 JChatMind 是把本轮新的记录全量放回来，并且加上了新增消息，我们目前采取对该 key 进行覆写或 Trim
        redisTemplate.delete(key);
        
        // 仅保留最新的 maxMessages 条记录
        int startIndex = Math.max(0, messages.size() - maxMessages);
        List<Message> messagesToSave = messages.subList(startIndex, messages.size());
        
        // 转化为可序列化的 DTO 存入缓存
        List<ChatMessageDTO> dtoList = convertToDTOs(messagesToSave);
        
        try {
            for (ChatMessageDTO dto : dtoList) {
                String json = objectMapper.writeValueAsString(dto);
                redisTemplate.opsForList().rightPush(key, json);
            }
            redisTemplate.expire(key, CACHE_TTL_HOURS, TimeUnit.HOURS);
        } catch (JsonProcessingException e) {
            log.error("Failed to serialize message to Redis JSON", e);
        }
    }

    @Override
    public List<Message> get(String conversationId) {
        String key = getCacheKey(conversationId);
        Long size = redisTemplate.opsForList().size(key);
        
        if (size != null && size > 0) {
            // Redis 命中 (热缓存)
            int fetchCount = maxMessages;
            List<Object> cachedData = redisTemplate.opsForList().range(key, -fetchCount, -1);
            if (cachedData != null && !cachedData.isEmpty()) {
                log.info("Hit Redis cache for conversationId: {}", conversationId);
                List<ChatMessageDTO> dtoList = new ArrayList<>();
                try {
                    for (Object obj : cachedData) {
                        ChatMessageDTO dto = objectMapper.readValue((String) obj, ChatMessageDTO.class);
                        dtoList.add(dto);
                    }
                    return convertToMessages(dtoList);
                } catch (Exception e) {
                    log.error("Failed to deserialize message from Redis cache, falling back to PostgreSQL", e);
                    // 反序列化失败，清除缓存，fallback 到 DB
                    redisTemplate.delete(key);
                }
            }
        }
        
        // Redis 未命中 (冷备)，从 PostgreSQL 读取最近的消息
        log.info("Miss Redis cache, loading from PostgreSQL for conversationId: {}", conversationId);
        int fetchCount = maxMessages;
        List<ChatMessageDTO> dbMessages = chatMessageFacadeService.getChatMessagesBySessionIdRecently(conversationId, fetchCount);
        
        if (dbMessages == null || dbMessages.isEmpty()) {
            return new ArrayList<>();
        }
        
        List<Message> loadedMessages = convertToMessages(dbMessages);
        
        // 加载后回写预热 Redis
        add(conversationId, loadedMessages);
        
        return loadedMessages;
    }

    @Override
    public void clear(String conversationId) {
        redisTemplate.delete(getCacheKey(conversationId));
    }
    
    // --- 转换方法 ---
    private List<ChatMessageDTO> convertToDTOs(List<Message> messages) {
        List<ChatMessageDTO> dtos = new ArrayList<>();
        ChatMessageDTO.ChatMessageDTOBuilder builder = ChatMessageDTO.builder();
        
        for (Message msg : messages) {
            if (msg instanceof SystemMessage) {
                SystemMessage sysMsg = (SystemMessage) msg;
                dtos.add(builder.role(ChatMessageDTO.RoleType.SYSTEM).content(sysMsg.getText()).build());
            } else if (msg instanceof UserMessage) {
                UserMessage userMsg = (UserMessage) msg;
                dtos.add(builder.role(ChatMessageDTO.RoleType.USER).content(userMsg.getText()).build());
            } else if (msg instanceof AssistantMessage) {
                AssistantMessage astMsg = (AssistantMessage) msg;
                dtos.add(builder.role(ChatMessageDTO.RoleType.ASSISTANT).content(astMsg.getText())
                        .metadata(ChatMessageDTO.MetaData.builder().toolCalls(astMsg.getToolCalls()).build())
                        .build());
            } else if (msg instanceof ToolResponseMessage) {
                ToolResponseMessage toolMsg = (ToolResponseMessage) msg;
                for (ToolResponseMessage.ToolResponse tr : toolMsg.getResponses()) {
                    dtos.add(builder.role(ChatMessageDTO.RoleType.TOOL).content(tr.responseData())
                            .metadata(ChatMessageDTO.MetaData.builder().toolResponse(tr).build())
                            .build());
                }
            }
        }
        return dtos;
    }

    private List<Message> convertToMessages(List<ChatMessageDTO> chatMessages) {
        List<Message> memory = new ArrayList<>();
        for (ChatMessageDTO chatMessageDTO : chatMessages) {
            switch (chatMessageDTO.getRole()) {
                case SYSTEM:
                    if (!StringUtils.hasLength(chatMessageDTO.getContent())) continue;
                    memory.add(0, new SystemMessage(chatMessageDTO.getContent())); // 倒序的话注意 这里可能不需要 add(0) 依赖查询结果顺序
                    break;
                case USER:
                    if (!StringUtils.hasLength(chatMessageDTO.getContent())) continue;
                    memory.add(new UserMessage(chatMessageDTO.getContent()));
                    break;
                case ASSISTANT:
                    memory.add(AssistantMessage.builder()
                            .content(chatMessageDTO.getContent())
                            .toolCalls(chatMessageDTO.getMetadata() == null ? null : chatMessageDTO.getMetadata().getToolCalls())
                            .build());
                    break;
                case TOOL:
                    memory.add(ToolResponseMessage.builder()
                            .responses(List.of(chatMessageDTO.getMetadata().getToolResponse()))
                            .build());
                    break;
                default:
                    log.error("不支持的 Message 类型: {}, content = {}", chatMessageDTO.getRole().getRole(), chatMessageDTO.getContent());
                    break;
            }
        }
        return memory;
    }
}
