package com.kama.jchatmind.agent;

import com.kama.jchatmind.agent.graph.AgentGraphState;
import com.kama.jchatmind.agent.graph.GraphEngine;
import com.kama.jchatmind.converter.ChatMessageConverter;
import com.kama.jchatmind.message.SseMessage;
import com.kama.jchatmind.model.dto.ChatMessageDTO;
import com.kama.jchatmind.model.response.CreateChatMessageResponse;
import com.kama.jchatmind.model.vo.ChatMessageVO;
import com.kama.jchatmind.service.ChatMessageFacadeService;
import com.kama.jchatmind.service.MemorySummarizationService;
import com.kama.jchatmind.service.RagService;
import com.kama.jchatmind.service.SseService;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class JChatMind {
    // 智能体 ID
    private String agentId;
    private String name;
    private String description;
    private String systemPrompt;

    // 状态
    private AgentState agentState;

    // 模型的聊天会话 ID
    private String chatSessionId;

    // 模型的聊天记录窗口
    private ChatMemory chatMemory;

    // SSE 服务, 用于发送消息给前端
    private SseService sseService;

    private ChatMessageConverter chatMessageConverter;

    // 持久化聊天消息到数据库
    private ChatMessageFacadeService chatMessageFacadeService;
    
    private MemorySummarizationService memorySummarizationService;
    private org.springframework.ai.chat.client.ChatClient chatClient;
    private RagService ragService;

    // AI 返回的，已经持久化，但是需要 sse 发给前端的消息
    private final List<ChatMessageDTO> pendingChatMessages = new ArrayList<>();

    // 多智能体图执行引擎
    @Setter
    private GraphEngine graphEngine;

    public JChatMind() {
    }

    public JChatMind(String agentId,
                     String name,
                     String description,
                     String systemPrompt,
                     ChatMemory chatMemory,
                     String chatSessionId,
                     SseService sseService,
                     ChatMessageFacadeService chatMessageFacadeService,
                     ChatMessageConverter chatMessageConverter,
                     GraphEngine graphEngine,
                     MemorySummarizationService memorySummarizationService,
                     org.springframework.ai.chat.client.ChatClient chatClient,
                     RagService ragService
    ) {
        this.agentId = agentId;
        this.name = name;
        this.description = description;
        this.systemPrompt = systemPrompt;

        this.chatSessionId = chatSessionId;
        this.sseService = sseService;

        this.chatMessageFacadeService = chatMessageFacadeService;
        this.chatMessageConverter = chatMessageConverter;
        this.memorySummarizationService = memorySummarizationService;
        this.chatClient = chatClient;
        this.ragService = ragService;

        this.agentState = AgentState.IDLE;
        
        this.graphEngine = graphEngine;

        this.chatMemory = chatMemory;
    }


    /**
     * 图引擎每产生一个有效的心智消息或者响应，都会回调此方法。
     * 进行落库并发送给前端
     */
    public void handleGeneratedMessage(Message message) {
        saveMessage(message);
        refreshPendingMessages();
    }

    private void saveMessage(Message message) {
        ChatMessageDTO.ChatMessageDTOBuilder builder = ChatMessageDTO.builder();
        if (message instanceof AssistantMessage) {
            AssistantMessage assistantMessage = (AssistantMessage) message;
            String text = assistantMessage.getText();
            boolean hasToolCalls = assistantMessage.getToolCalls() != null
                    && !assistantMessage.getToolCalls().isEmpty();
            if ((text == null || text.trim().isEmpty()) && !hasToolCalls) {
                log.debug("Skip persisting empty assistant message for session {}", this.chatSessionId);
                return;
            }
            // Hide worker planning text when this message is actually a tool-call carrier.
            String persistedContent = hasToolCalls ? "" : text;
            ChatMessageDTO chatMessageDTO = builder.role(ChatMessageDTO.RoleType.ASSISTANT)
                    .content(persistedContent)
                    .sessionId(this.chatSessionId)
                    .metadata(ChatMessageDTO.MetaData.builder()
                            .toolCalls(assistantMessage.getToolCalls())
                            .build())
                    .build();
            CreateChatMessageResponse chatMessage = chatMessageFacadeService.createChatMessage(chatMessageDTO);
            chatMessageDTO.setId(chatMessage.getChatMessageId());
            pendingChatMessages.add(chatMessageDTO);
            
        } else if (message instanceof ToolResponseMessage) {
            ToolResponseMessage toolResponseMessage = (ToolResponseMessage) message;
            for (ToolResponseMessage.ToolResponse toolResponse : toolResponseMessage.getResponses()) {
                ChatMessageDTO chatMessageDTO = builder.role(ChatMessageDTO.RoleType.TOOL)
                        .content(toolResponse.responseData())
                        .sessionId(this.chatSessionId)
                        .metadata(ChatMessageDTO.MetaData.builder()
                                .toolResponse(toolResponse)
                                .build())
                        .build();
                CreateChatMessageResponse chatMessage = chatMessageFacadeService.createChatMessage(chatMessageDTO);
                chatMessageDTO.setId(chatMessage.getChatMessageId());
                pendingChatMessages.add(chatMessageDTO);
            }
        }
    }

    private void refreshPendingMessages() {
        for (ChatMessageDTO message : pendingChatMessages) {
            ChatMessageVO vo = chatMessageConverter.toVO(message);
            String chatMessageId = StringUtils.hasLength(message.getId())
                    ? message.getId()
                    : vo.getId();
            sendGeneratedContentMessage(vo, chatMessageId);
        }
        pendingChatMessages.clear();
    }

    private void sendGeneratedContentMessage(ChatMessageVO vo, String chatMessageId) {
        SseMessage sseMessage = SseMessage.builder()
                .type(SseMessage.Type.AI_GENERATED_CONTENT)
                .payload(SseMessage.Payload.builder()
                        .message(vo)
                        .build())
                .metadata(SseMessage.Metadata.builder()
                        .chatMessageId(chatMessageId)
                        .build())
                .build();
        sseService.send(this.chatSessionId, sseMessage);
    }

    public String startAssistantStream() {
        ChatMessageDTO streamMessageDTO = ChatMessageDTO.builder()
                .role(ChatMessageDTO.RoleType.ASSISTANT)
                .content("")
                .sessionId(this.chatSessionId)
                .metadata(ChatMessageDTO.MetaData.builder().build())
                .build();
        CreateChatMessageResponse response = chatMessageFacadeService.createChatMessage(streamMessageDTO);
        streamMessageDTO.setId(response.getChatMessageId());

        ChatMessageVO vo = chatMessageConverter.toVO(streamMessageDTO);
        SseMessage startEvent = SseMessage.builder()
                .type(SseMessage.Type.AI_GENERATED_CONTENT_START)
                .payload(SseMessage.Payload.builder()
                        .message(vo)
                        .build())
                .metadata(SseMessage.Metadata.builder()
                        .chatMessageId(vo.getId())
                        .build())
                .build();
        sseService.send(this.chatSessionId, startEvent);
        return vo.getId();
    }

    public void appendAssistantStream(String messageId, String deltaContent) {
        if (!StringUtils.hasLength(messageId) || !StringUtils.hasLength(deltaContent)) {
            return;
        }

        chatMessageFacadeService.appendChatMessage(messageId, deltaContent);

        SseMessage deltaEvent = SseMessage.builder()
                .type(SseMessage.Type.AI_GENERATED_CONTENT_DELTA)
                .payload(SseMessage.Payload.builder()
                        .deltaContent(deltaContent)
                        .build())
                .metadata(SseMessage.Metadata.builder()
                        .chatMessageId(messageId)
                        .build())
                .build();
        sseService.send(this.chatSessionId, deltaEvent);
    }

    public void finishAssistantStream(String messageId) {
        if (!StringUtils.hasLength(messageId)) {
            return;
        }
        SseMessage endEvent = SseMessage.builder()
                .type(SseMessage.Type.AI_GENERATED_CONTENT_END)
                .payload(SseMessage.Payload.builder()
                        .done(Boolean.TRUE)
                        .build())
                .metadata(SseMessage.Metadata.builder()
                        .chatMessageId(messageId)
                        .build())
                .build();
        sseService.send(this.chatSessionId, endEvent);
    }

    public void run() {
        if (agentState != AgentState.IDLE) {
            throw new IllegalStateException("Agent is not idle");
        }

        try {
            agentState = AgentState.EXECUTING;

            // 装载图引擎的状态上下文
            AgentGraphState state = new AgentGraphState();
            // get 方法只接受 conversationId，DistributedChatMemory 会受内部 maxMessages 取值限制
            List<Message> memoryMessages = new ArrayList<>(this.chatMemory.get(this.chatSessionId));
            
            // 构建长期记忆增强的系统提示词
            String enhancedSystemPrompt = this.systemPrompt;
            if (this.ragService != null && !memoryMessages.isEmpty() && memoryMessages.get(memoryMessages.size() - 1) instanceof UserMessage) {
                UserMessage lastUserMsg = (UserMessage) memoryMessages.get(memoryMessages.size() - 1);
                String userLatestInput = lastUserMsg.getText();
                try {
                    List<String> facts = this.ragService.similaritySearch(this.chatSessionId, userLatestInput);
                    if (facts != null && !facts.isEmpty()) {
                        enhancedSystemPrompt += "\n\n【关于用户的过去记忆】：\n" + String.join("\n", facts);
                    }
                } catch (Exception e) {
                    log.error("Failed to retrieve long term vector memory", e);
                }
            }
            
            // 每次运行前，动态追加当前 Agent 的 System prompt（不放入持久化层）
            if (StringUtils.hasLength(enhancedSystemPrompt)) {
                // 简单处理：永远放第一条
                memoryMessages.add(0, new SystemMessage(enhancedSystemPrompt));
            }
            state.setMessages(memoryMessages);
            // Mark this turn start index so graph nodes can ignore previous-turn tool states.
            state.getAttributes().put("turn_start_message_count", memoryMessages.size());

            // 启动 Supervisor-Worker 图流转
            if (graphEngine != null) {
                graphEngine.run(state);
            } else {
                log.warn("未装配 GraphEngine，执行失败");
            }

            // 更新对话记忆窗口在 Redis 中的状态
            List<Message> finalMessages = state.getMessages();
            if (StringUtils.hasLength(this.systemPrompt) && !finalMessages.isEmpty() && finalMessages.get(0) instanceof SystemMessage) {
                // 将前面塞入的临时 SystemMessage 剔除，不写入热缓存，避免污染用户真实的历史记忆
                finalMessages = finalMessages.subList(1, finalMessages.size());
            }
            
            // 更新 Redis 热缓存
            this.chatMemory.add(this.chatSessionId, finalMessages);
            
            // 触发内存摘要机制 (判断 Token/消息长度逼近阈值)
            // 这里简单以消息条数作为阈值，实际可结合 Tokenizer
            int threshold = 30; // 假设满30条消息触发压缩
            int keepLatestN = 10; // 压缩后仅保留最近10条原消息，其余转为摘要
            if (finalMessages.size() >= threshold) {
                if (this.memorySummarizationService != null && this.chatMemory instanceof com.kama.jchatmind.agent.memory.DistributedChatMemory) {
                    this.memorySummarizationService.summarizeAndExtractFactsAsync(
                        this.chatSessionId,
                        this.chatClient,
                        keepLatestN,
                        (com.kama.jchatmind.agent.memory.DistributedChatMemory) this.chatMemory
                    );
                }
            }

            agentState = AgentState.FINISHED;
            log.info("Agent 任务编排结束");

        } catch (Exception e) {
            agentState = AgentState.ERROR;
            log.error("Error executing Agent Graph", e);
            throw new RuntimeException("Error executing Agent Graph", e);
        } finally {
            // Always notify frontend that this round has finished,
            // so input can be re-enabled even when graph execution fails.
            SseMessage doneMessage = SseMessage.builder()
                    .type(SseMessage.Type.AI_DONE)
                    .payload(SseMessage.Payload.builder()
                            .done(Boolean.TRUE)
                            .build())
                    .build();
            sseService.send(this.chatSessionId, doneMessage);
        }
    }

    @Override
    public String toString() {
        return "JChatMind {" +
                "name = " + name + ",\n" +
                "description = " + description + ",\n" +
                "agentId = " + agentId + ",\n" +
                "systemPrompt = " + systemPrompt + "}";
    }
}
