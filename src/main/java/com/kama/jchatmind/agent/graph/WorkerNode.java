package com.kama.jchatmind.agent.graph;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.tool.ToolCallback;

import java.util.List;
import java.util.stream.Collectors;

@Slf4j
public class WorkerNode implements AgentNode {

    private final ChatClient chatClient;
    private final ChatOptions chatOptions;
    private final List<ToolCallback> tools;
    private final ToolCallingManager toolCallingManager;

    public WorkerNode(ChatClient chatClient, ChatOptions chatOptions, List<ToolCallback> tools) {
        this.chatClient = chatClient;
        this.chatOptions = chatOptions;
        this.tools = tools;
        this.toolCallingManager = ToolCallingManager.builder().build();
    }

    @Override
    public String getName() {
        return "WORKER";
    }

    @Override
    public AgentGraphState process(AgentGraphState state) {
        log.info("[GraphEngine] 节点执行: {}", getName());

        String workerPrompt = "你是一个专门操作工具的执行机器人 (Worker)。你的唯一目标是尽可能准确地使用注册好的外部工具来响应需求，千万不要自己瞎编。";

        Prompt prompt = Prompt.builder()
                .chatOptions(this.chatOptions)
                .messages(state.getMessages())
                .build();

        ChatResponse response = chatClient.prompt(prompt)
                .system(workerPrompt)
                .toolCallbacks(tools.toArray(new ToolCallback[0]))
                .call()
                .chatClientResponse()
                .chatResponse();

        AssistantMessage output = response.getResult().getOutput();
        List<AssistantMessage.ToolCall> toolCalls = output.getToolCalls();

        // 无论如何，要保存 Worker 大模型的直接返回信息（通常包含工具调用的申明）
        state.getMessages().add(output);
        state.getAttributes().put("latest_tool_call_msg", output);

        if (toolCalls != null && !toolCalls.isEmpty()) {
            log.info("[GraphEngine] WORKER 决定执行工具...");
            ToolExecutionResult executionResult = toolCallingManager.executeToolCalls(prompt, response);

            // 提取工具调用的结果（通常是会话历史的最后一条）
            List<Message> history = executionResult.conversationHistory();
            ToolResponseMessage toolResponseMsg = (ToolResponseMessage) history.get(history.size() - 1);

            state.getMessages().add(toolResponseMsg);
            state.getAttributes().put("latest_tool_response_msg", toolResponseMsg);
            
            String logMsg = toolResponseMsg.getResponses().stream()
                .map(r -> r.name() + " -> " + r.responseData())
                .collect(Collectors.joining(", "));
            log.info("[GraphEngine] 工具执行完毕，结果: {}", logMsg);

        } else {
             log.warn("[GraphEngine] WORKER 没有产生任何工具调用!");
        }

        // 把执行后的上下文再次送入给 Supervisor，让大脑判断还需要不需要继续做什么，或者是否得到了最终答案
        state.setNextNode("SUPERVISOR");
        return state;
    }
}
