package com.kama.jchatmind.agent;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.kama.jchatmind.agent.graph.GraphEngine;
import com.kama.jchatmind.agent.graph.RagWorkerNode;
import com.kama.jchatmind.agent.graph.SqlWorkerNode;
import com.kama.jchatmind.agent.graph.SupervisorNode;
import com.kama.jchatmind.agent.graph.TokenStreamPublisher;
import com.kama.jchatmind.agent.graph.ToolWorkerNode;
import com.kama.jchatmind.agent.tools.Tool;
import com.kama.jchatmind.config.ChatClientRegistry;
import com.kama.jchatmind.converter.AgentConverter;
import com.kama.jchatmind.converter.ChatMessageConverter;
import com.kama.jchatmind.converter.KnowledgeBaseConverter;
import com.kama.jchatmind.mapper.AgentMapper;
import com.kama.jchatmind.mapper.KnowledgeBaseMapper;
import com.kama.jchatmind.model.dto.AgentDTO;
import com.kama.jchatmind.model.dto.ChatMessageDTO;
import com.kama.jchatmind.model.dto.KnowledgeBaseDTO;
import com.kama.jchatmind.model.entity.Agent;
import com.kama.jchatmind.model.entity.KnowledgeBase;
import com.kama.jchatmind.service.ChatMessageFacadeService;
import com.kama.jchatmind.service.RagService;
import com.kama.jchatmind.service.SseService;
import com.kama.jchatmind.service.ToolFacadeService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;
import org.springframework.aop.support.AopUtils;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.util.*;
import java.lang.reflect.Method;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
public class JChatMindFactory {

    private static final Logger log = LoggerFactory.getLogger(JChatMindFactory.class);
    private final ChatClientRegistry chatClientRegistry;
    private final SseService sseService;
    private final AgentMapper agentMapper;
    private final AgentConverter agentConverter;
    private final KnowledgeBaseMapper knowledgeBaseMapper;
    private final KnowledgeBaseConverter knowledgeBaseConverter;
    private final ToolFacadeService toolFacadeService;
    private final ChatMessageFacadeService chatMessageFacadeService;
    private final ChatMessageConverter chatMessageConverter;
    private final RedisTemplate<String, Object> redisTemplate;
    private final com.kama.jchatmind.service.MemorySummarizationService memorySummarizationService;
    private final RagService ragService;

    // 娣囶喖顦叉禍鍡曞紬闁插秶娈戦獮璺哄絺缁狙冨焼 BUG閿涙艾鍨归梽銈勭啊閸樼喐婀伴惃鍕灇閸涙ê褰夐柌?private AgentDTO agentConfig;

    public JChatMindFactory(
            ChatClientRegistry chatClientRegistry,
            SseService sseService,
            AgentMapper agentMapper,
            AgentConverter agentConverter,
            KnowledgeBaseMapper knowledgeBaseMapper,
            KnowledgeBaseConverter knowledgeBaseConverter,
            ToolFacadeService toolFacadeService,
            ChatMessageFacadeService chatMessageFacadeService,
            ChatMessageConverter chatMessageConverter,
            RedisTemplate<String, Object> redisTemplate,
            com.kama.jchatmind.service.MemorySummarizationService memorySummarizationService,
            RagService ragService
    ) {
        this.chatClientRegistry = chatClientRegistry;
        this.sseService = sseService;
        this.agentMapper = agentMapper; 
        this.agentConverter = agentConverter; 
        this.knowledgeBaseMapper = knowledgeBaseMapper;
        this.knowledgeBaseConverter = knowledgeBaseConverter;
        this.toolFacadeService = toolFacadeService;
        this.chatMessageFacadeService = chatMessageFacadeService;
        this.chatMessageConverter = chatMessageConverter;
        this.redisTemplate = redisTemplate;
        this.memorySummarizationService = memorySummarizationService;
        this.ragService = ragService;
    }

    private Agent loadAgent(String agentId) {
        return agentMapper.selectById(agentId);
    }

    // 缁夊娅庢禍鍡楀斧婵娈?loadMemory() 閺傝纭堕敍灞兼唉閻?DistributedChatMemory 閸愬懘鍎撮幏澶婂絿


    private AgentDTO toAgentConfig(Agent agent) {
        try {
            return agentConverter.toDTO(agent);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("鐟欙絾鐎?Agent 闁板秶鐤嗘径杈Е", e);
        }
    }

    private List<KnowledgeBaseDTO> resolveRuntimeKnowledgeBases(AgentDTO agentConfig) {
        List<String> allowedKbIds = agentConfig.getAllowedKbs();
        if (allowedKbIds == null || allowedKbIds.isEmpty()) {
            return Collections.emptyList();
        }
        List<KnowledgeBase> knowledgeBases = knowledgeBaseMapper.selectByIdBatch(allowedKbIds);
        if (knowledgeBases.isEmpty()) {
            return Collections.emptyList();
        }
        List<KnowledgeBaseDTO> kbDTOs = new ArrayList<>();
        try {
            for (KnowledgeBase knowledgeBase : knowledgeBases) {
                kbDTOs.add(knowledgeBaseConverter.toDTO(knowledgeBase));
            }
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
        return kbDTOs;
    }

    private List<Tool> resolveRuntimeTools(AgentDTO agentConfig) {
        List<Tool> runtimeTools = new ArrayList<>();

        // Do not expose internal loop-control tool to the LLM.
        runtimeTools.addAll(
                toolFacadeService.getFixedTools().stream()
                        .filter(tool -> !"terminate".equalsIgnoreCase(tool.getName()))
                        .toList()
        );

        List<String> allowedToolNames = agentConfig.getAllowedTools();
        if (allowedToolNames == null || allowedToolNames.isEmpty()) {
            // Empty means no restriction: enable all optional tools by default.
            runtimeTools.addAll(toolFacadeService.getOptionalTools());
            return runtimeTools;
        }
        Map<String, Tool> optionalToolMap = toolFacadeService.getOptionalTools()
                .stream()
                .collect(Collectors.toMap(Tool::getName, Function.identity()));
                
        for (String toolName : allowedToolNames) {
            Tool tool = optionalToolMap.get(toolName);
            if (tool != null) {
                runtimeTools.add(tool);
            }
        }
        return runtimeTools;
    }

    private List<Tool> disableKnowledgeToolIfNoKb(
            List<Tool> runtimeTools,
            List<KnowledgeBaseDTO> knowledgeBases
    ) {
        if (knowledgeBases != null && !knowledgeBases.isEmpty()) {
            return runtimeTools;
        }
        return runtimeTools.stream()
                .filter(tool -> !toolSupportsDeclaredName(tool, "KnowledgeTool"))
                .toList();
    }

    private List<ToolCallback> buildToolCallbacks(List<Tool> runtimeTools) {
        List<ToolCallback> callbacks = new ArrayList<>();
        for (Tool tool : runtimeTools) {
            Object target = resolveToolTarget(tool);
            ToolCallback[] toolCallbacks = MethodToolCallbackProvider.builder()
                    .toolObjects(target)
                    .build()
                    .getToolCallbacks();
            callbacks.addAll(Arrays.asList(toolCallbacks));
        }
        return callbacks;
    }

    private Set<String> resolveAvailableToolNames(List<Tool> runtimeTools) {
        Set<String> names = new LinkedHashSet<>();
        for (Tool tool : runtimeTools) {
            if (tool.getName() != null && !tool.getName().isBlank()) {
                names.add(tool.getName());
            }

            Class<?> targetClass = resolveToolClass(tool);
            for (Method method : targetClass.getMethods()) {
                org.springframework.ai.tool.annotation.Tool annotation =
                        method.getAnnotation(org.springframework.ai.tool.annotation.Tool.class);
                if (annotation == null) {
                    continue;
                }
                String declaredName = annotation.name();
                if (declaredName != null && !declaredName.isBlank()) {
                    names.add(declaredName);
                } else {
                    names.add(method.getName());
                }
            }
        }
        return names;
    }

    private Class<?> resolveToolClass(Tool tool) {
        try {
            return AopUtils.isAopProxy(tool)
                    ? AopUtils.getTargetClass(tool)
                    : tool.getClass();
        } catch (Exception e) {
            return tool.getClass();
        }
    }

    private Object resolveToolTarget(Tool tool) {
        return tool;
    }

    private boolean toolSupportsDeclaredName(Tool tool, String expectedName) {
        if (tool == null || expectedName == null || expectedName.isBlank()) {
            return false;
        }
        if (expectedName.equalsIgnoreCase(tool.getName())) {
            return true;
        }
        Class<?> targetClass = resolveToolClass(tool);
        for (Method method : targetClass.getMethods()) {
            org.springframework.ai.tool.annotation.Tool annotation =
                    method.getAnnotation(org.springframework.ai.tool.annotation.Tool.class);
            if (annotation == null) {
                continue;
            }
            String declaredName = annotation.name();
            if (declaredName != null && !declaredName.isBlank()) {
                if (expectedName.equalsIgnoreCase(declaredName)) {
                    return true;
                }
            } else if (expectedName.equalsIgnoreCase(method.getName())) {
                return true;
            }
        }
        return false;
    }

        public JChatMind create(String agentId, String chatSessionId) {
        Agent agent = loadAgent(agentId);
        AgentDTO agentConfig = toAgentConfig(agent);

        int maxMessages = agentConfig.getChatOptions().getMessageLength();
        ChatMemory chatMemory = new com.kama.jchatmind.agent.memory.DistributedChatMemory(
                redisTemplate, chatMessageFacadeService, maxMessages);

        List<KnowledgeBaseDTO> knowledgeBases = resolveRuntimeKnowledgeBases(agentConfig);
        List<Tool> runtimeTools = resolveRuntimeTools(agentConfig);
        runtimeTools = disableKnowledgeToolIfNoKb(runtimeTools, knowledgeBases);
        Set<String> availableToolNames = resolveAvailableToolNames(runtimeTools);

        List<Tool> ragTools = runtimeTools.stream()
                .filter(tool -> toolSupportsDeclaredName(tool, "KnowledgeTool"))
                .toList();
        List<Tool> sqlTools = runtimeTools.stream()
                .filter(tool -> toolSupportsDeclaredName(tool, "databaseQuery"))
                .toList();
        List<Tool> toolTools = runtimeTools.stream()
                .filter(tool -> !toolSupportsDeclaredName(tool, "KnowledgeTool"))
                .filter(tool -> !toolSupportsDeclaredName(tool, "databaseQuery"))
                .toList();

        List<ToolCallback> ragToolCallbacks = buildToolCallbacks(ragTools);
        List<ToolCallback> sqlToolCallbacks = buildToolCallbacks(sqlTools);
        List<ToolCallback> toolWorkerCallbacks = buildToolCallbacks(toolTools);

        Set<String> ragToolNames = resolveAvailableToolNames(ragTools);
        Set<String> sqlToolNames = resolveAvailableToolNames(sqlTools);
        Set<String> toolWorkerNames = resolveAvailableToolNames(toolTools);

        ChatClient chatClient = chatClientRegistry.get(agent.getModel());
        if (Objects.isNull(chatClient)) {
            throw new IllegalStateException("Cannot find ChatClient for model: " + agent.getModel());
        }

        RagService ragServiceForAgent = (knowledgeBases == null || knowledgeBases.isEmpty())
                ? null
                : ragService;

        JChatMind jChatMind = new JChatMind(
                agent.getId(),
                agent.getName(),
                agent.getDescription(),
                agent.getSystemPrompt(),
                chatMemory,
                chatSessionId,
                sseService,
                chatMessageFacadeService,
                chatMessageConverter,
                null,
                memorySummarizationService,
                chatClient,
                ragServiceForAgent
        );

        org.springframework.ai.chat.prompt.ChatOptions chatOptions =
                org.springframework.ai.model.tool.DefaultToolCallingChatOptions.builder()
                        .internalToolExecutionEnabled(false)
                        .build();

        Set<String> availableWorkers = new LinkedHashSet<>();
        availableWorkers.add("RAG_WORKER");
        availableWorkers.add("SQL_WORKER");
        availableWorkers.add("TOOL_WORKER");

        SupervisorNode supervisorNode = new SupervisorNode(chatClient, chatOptions, availableToolNames, availableWorkers);
        TokenStreamPublisher tokenStreamPublisher = new TokenStreamPublisher() {
            @Override
            public String startAssistantStream() {
                return jChatMind.startAssistantStream();
            }

            @Override
            public void appendAssistantStream(String messageId, String deltaContent) {
                jChatMind.appendAssistantStream(messageId, deltaContent);
            }

            @Override
            public void finishAssistantStream(String messageId) {
                jChatMind.finishAssistantStream(messageId);
            }
        };
        RagWorkerNode ragWorkerNode = new RagWorkerNode(chatClient, chatOptions, ragToolCallbacks, ragToolNames, tokenStreamPublisher);
        SqlWorkerNode sqlWorkerNode = new SqlWorkerNode(chatClient, chatOptions, sqlToolCallbacks, sqlToolNames, tokenStreamPublisher);
        ToolWorkerNode toolWorkerNode = new ToolWorkerNode(chatClient, chatOptions, toolWorkerCallbacks, toolWorkerNames, tokenStreamPublisher);

        GraphEngine engine = new GraphEngine("SUPERVISOR", jChatMind::handleGeneratedMessage);
        engine.addNode(supervisorNode);
        engine.addNode(ragWorkerNode);
        engine.addNode(sqlWorkerNode);
        engine.addNode(toolWorkerNode);

        jChatMind.setGraphEngine(engine);

        return jChatMind;
    }
}


