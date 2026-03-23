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
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Distributed chat memory with Redis hot cache + DB cold storage.
 */
public class DistributedChatMemory implements ChatMemory {

    private static final Logger log = LoggerFactory.getLogger(DistributedChatMemory.class);

    private final RedisTemplate<String, Object> redisTemplate;
    private final ChatMessageFacadeService chatMessageFacadeService;
    private final int maxMessages;
    private final ObjectMapper objectMapper;

    private static final String KEY_PREFIX = "chat:memory:";
    private static final long CACHE_TTL_HOURS = 24;

    public DistributedChatMemory(RedisTemplate<String, Object> redisTemplate,
                                 ChatMessageFacadeService chatMessageFacadeService,
                                 int maxMessages) {
        this.redisTemplate = redisTemplate;
        this.chatMessageFacadeService = chatMessageFacadeService;
        this.maxMessages = maxMessages;

        this.objectMapper = new ObjectMapper();
        this.objectMapper.activateDefaultTyping(
                LaissezFaireSubTypeValidator.instance,
                ObjectMapper.DefaultTyping.NON_FINAL,
                JsonTypeInfo.As.PROPERTY
        );
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
        redisTemplate.delete(key);

        int startIndex = Math.max(0, messages.size() - maxMessages);
        List<Message> messagesToSave = messages.subList(startIndex, messages.size());
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
            List<Object> cachedData = redisTemplate.opsForList().range(key, -maxMessages, -1);
            if (cachedData != null && !cachedData.isEmpty()) {
                try {
                    List<ChatMessageDTO> dtoList = new ArrayList<>();
                    for (Object obj : cachedData) {
                        dtoList.add(objectMapper.readValue((String) obj, ChatMessageDTO.class));
                    }
                    return convertToMessages(dtoList);
                } catch (Exception e) {
                    log.error("Failed to deserialize Redis cache, fallback to DB", e);
                    redisTemplate.delete(key);
                }
            }
        }

        List<ChatMessageDTO> dbMessages = chatMessageFacadeService.getChatMessagesBySessionIdRecently(conversationId, maxMessages);
        if (dbMessages == null || dbMessages.isEmpty()) {
            return new ArrayList<>();
        }

        List<Message> loadedMessages = convertToMessages(dbMessages);
        add(conversationId, loadedMessages);
        return loadedMessages;
    }

    @Override
    public void clear(String conversationId) {
        redisTemplate.delete(getCacheKey(conversationId));
    }

    private List<ChatMessageDTO> convertToDTOs(List<Message> messages) {
        List<ChatMessageDTO> dtos = new ArrayList<>();
        for (Message msg : messages) {
            if (msg instanceof SystemMessage sysMsg) {
                dtos.add(ChatMessageDTO.builder()
                        .role(ChatMessageDTO.RoleType.SYSTEM)
                        .content(sysMsg.getText())
                        .build());
            } else if (msg instanceof UserMessage userMsg) {
                dtos.add(ChatMessageDTO.builder()
                        .role(ChatMessageDTO.RoleType.USER)
                        .content(userMsg.getText())
                        .build());
            } else if (msg instanceof AssistantMessage astMsg) {
                dtos.add(ChatMessageDTO.builder()
                        .role(ChatMessageDTO.RoleType.ASSISTANT)
                        .content(astMsg.getText())
                        .metadata(ChatMessageDTO.MetaData.builder()
                                .toolCalls(astMsg.getToolCalls())
                                .build())
                        .build());
            } else if (msg instanceof ToolResponseMessage toolMsg) {
                for (ToolResponseMessage.ToolResponse tr : toolMsg.getResponses()) {
                    dtos.add(ChatMessageDTO.builder()
                            .role(ChatMessageDTO.RoleType.TOOL)
                            .content(tr.responseData())
                            .metadata(ChatMessageDTO.MetaData.builder()
                                    .toolResponse(tr)
                                    .build())
                            .build());
                }
            }
        }
        return dtos;
    }

    private List<Message> convertToMessages(List<ChatMessageDTO> chatMessages) {
        List<Message> memory = new ArrayList<>();
        boolean canAppendToolResponse = false;

        for (ChatMessageDTO dto : chatMessages) {
            if (dto.getRole() == null) {
                continue;
            }

            switch (dto.getRole()) {
                case SYSTEM -> {
                    if (!StringUtils.hasLength(dto.getContent())) {
                        continue;
                    }
                    memory.add(new SystemMessage(dto.getContent()));
                    canAppendToolResponse = false;
                }
                case USER -> {
                    if (!StringUtils.hasLength(dto.getContent())) {
                        continue;
                    }
                    memory.add(new UserMessage(dto.getContent()));
                    canAppendToolResponse = false;
                }
                case ASSISTANT -> {
                    List<AssistantMessage.ToolCall> toolCalls =
                            dto.getMetadata() == null ? null : dto.getMetadata().getToolCalls();
                    boolean hasText = StringUtils.hasLength(dto.getContent());
                    boolean hasToolCalls = toolCalls != null && !toolCalls.isEmpty();
                    if (!hasText && !hasToolCalls) {
                        continue;
                    }
                    memory.add(AssistantMessage.builder()
                            .content(dto.getContent())
                            .toolCalls(toolCalls)
                            .build());
                    canAppendToolResponse = hasToolCalls;
                }
                case TOOL -> {
                    if (!canAppendToolResponse
                            || dto.getMetadata() == null
                            || dto.getMetadata().getToolResponse() == null) {
                        log.warn("Skip orphan tool message in memory reconstruction, id={}", dto.getId());
                        continue;
                    }
                    memory.add(ToolResponseMessage.builder()
                            .responses(List.of(dto.getMetadata().getToolResponse()))
                            .build());
                }
            }
        }

        return memory;
    }
}
