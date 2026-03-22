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

        String workerPrompt = """
                You are a worker that executes tasks with tools when needed.

                Rules:
                1) If a tool is needed, call the appropriate tool.
                2) If no tool is needed, answer the user directly and clearly.
                3) Never output meta-planning text.
                """;

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
        String outputText = output.getText();
        boolean hasMeaningfulText = outputText != null && !outputText.trim().isEmpty();

        // If no tool call is produced, treat worker output as final answer.
        if (toolCalls == null || toolCalls.isEmpty()) {
            if (hasMeaningfulText) {
                state.getMessages().add(output);
                state.getAttributes().put("latest_message", output);
                state.setNextNode("FINISH");
            } else {
                // Graceful fallback: never finish a round with empty visible text.
                AssistantMessage fallback = new AssistantMessage("我没有生成有效回复，请换个问法再试一次。");
                state.getMessages().add(fallback);
                state.getAttributes().put("latest_message", fallback);
                state.setNextNode("FINISH");
            }
            return state;
        }

        // Persist tool-call message and execute tools.
        state.getMessages().add(output);
        state.getAttributes().put("latest_tool_call_msg", output);

        log.info("[GraphEngine] WORKER 决定执行工具...");
        ToolExecutionResult executionResult = toolCallingManager.executeToolCalls(prompt, response);

        List<Message> history = executionResult.conversationHistory();
        ToolResponseMessage toolResponseMsg = (ToolResponseMessage) history.get(history.size() - 1);

        state.getMessages().add(toolResponseMsg);
        state.getAttributes().put("latest_tool_response_msg", toolResponseMsg);

        String logMsg = toolResponseMsg.getResponses().stream()
                .map(r -> r.name() + " -> " + r.responseData())
                .collect(Collectors.joining(", "));
        log.info("[GraphEngine] 工具执行完毕，结果: {}", logMsg);

        // Build a direct final answer from tool results to avoid extra routing loops.
        String finalText = toolResponseMsg.getResponses().stream()
                .map(ToolResponseMessage.ToolResponse::responseData)
                .collect(Collectors.joining("\n"));
        AssistantMessage finalMsg = new AssistantMessage(finalText);
        state.getMessages().add(finalMsg);
        state.getAttributes().put("latest_message", finalMsg);
        state.setNextNode("FINISH");
        return state;
    }
}
