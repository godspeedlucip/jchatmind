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
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Slf4j
public class WorkerNode implements AgentNode {

    private static final Pattern INVOKE_NAME_PATTERN = Pattern.compile("name\\s*=\\s*\"([^\"]+)\"", Pattern.CASE_INSENSITIVE);

    private final ChatClient chatClient;
    private final ChatOptions chatOptions;
    private final List<ToolCallback> tools;
    private final ToolCallingManager toolCallingManager;
    private final Set<String> availableToolNames;

    public WorkerNode(ChatClient chatClient,
                      ChatOptions chatOptions,
                      List<ToolCallback> tools,
                      Set<String> availableToolNames) {
        this.chatClient = chatClient;
        this.chatOptions = chatOptions;
        this.tools = tools;
        this.availableToolNames = availableToolNames;
        this.toolCallingManager = ToolCallingManager.builder().build();
    }

    @Override
    public String getName() {
        return "WORKER";
    }

    @Override
    public AgentGraphState process(AgentGraphState state) {
        log.info("[GraphEngine] Node: {}", getName());

        state.getAttributes().put("tool_failed", false);
        state.getAttributes().remove("last_tool_error");
        state.getAttributes().remove("worker_candidate_answer");
        state.getAttributes().remove("last_executed_tool_names");

        String toolNames = availableToolNames == null || availableToolNames.isEmpty()
                ? "(none)"
                : String.join(", ", availableToolNames);

        String workerPrompt = String.format(
                "You are a worker node.%n%n"
                        + "Available tool names (exact): %s%n%n"
                        + "Rules:%n"
                        + "1) Call tools when needed.%n"
                        + "2) If no tool is needed, provide plain text only.%n"
                        + "3) Never output DSML/XML/function-call markup in text.%n"
                        + "4) Never invent tool names.%n"
                        + "5) Always prioritize the latest user request over older turns.%n"
                        + "6) For databaseQuery, do not repeat previous failed SQL; generate SQL that matches the latest user request.%n",
                toolNames
        );

        Prompt prompt = Prompt.builder()
                .chatOptions(chatOptions)
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
        String outputText = output.getText() == null ? "" : output.getText().trim();

        if (toolCalls == null || toolCalls.isEmpty()) {
            String workerText;
            if (looksLikeToolMarkup(outputText)) {
                String maybeToolName = extractToolName(outputText);
                if (!maybeToolName.isBlank() && (availableToolNames == null || !availableToolNames.contains(maybeToolName))) {
                    workerText = "Detected unregistered tool call '" + maybeToolName + "'. Available tools: " + toolNames + ".";
                } else {
                    workerText = "Detected invalid tool-call markup. Available tools: " + toolNames + ".";
                }
                state.getAttributes().put("tool_failed", true);
                state.getAttributes().put("last_tool_error", workerText);
            } else if (!outputText.isBlank()) {
                workerText = outputText;
            } else {
                workerText = "Task processed, but no displayable output was produced.";
            }

            state.getAttributes().put("worker_candidate_answer", workerText);
            state.setNextNode("SUPERVISOR");
            return state;
        }

        try {
            log.info("[GraphEngine] WORKER executing tool calls...");
            ToolExecutionResult executionResult = toolCallingManager.executeToolCalls(prompt, response);

            List<Message> history = executionResult.conversationHistory();
            state.getMessages().add(output);
            state.getAttributes().put("latest_tool_call_msg", output);

            ToolResponseMessage toolResponseMsg = (ToolResponseMessage) history.get(history.size() - 1);

            state.getMessages().add(toolResponseMsg);
            state.getAttributes().put("latest_tool_response_msg", toolResponseMsg);

            String logMsg = toolResponseMsg.getResponses().stream()
                    .map(r -> r.name() + " -> " + r.responseData())
                    .collect(Collectors.joining(", "));
            log.info("[GraphEngine] Tool results: {}", logMsg);
        } catch (Exception ex) {
            log.error("[GraphEngine] Tool execution failed", ex);
            String error = "Tool execution failed: " + ex.getMessage();
            state.getAttributes().put("tool_failed", true);
            state.getAttributes().put("last_tool_error", error);
            state.getAttributes().put("worker_candidate_answer", error);
            state.getAttributes().remove("latest_tool_call_msg");
            state.getAttributes().remove("latest_tool_response_msg");
        }

        state.setNextNode("SUPERVISOR");
        return state;
    }

    private boolean looksLikeToolMarkup(String text) {
        if (text == null || text.isBlank()) {
            return false;
        }
        String lower = text.toLowerCase(Locale.ROOT);
        return lower.contains("dsml")
                || lower.contains("function_calls")
                || lower.contains("tool_calls")
                || lower.contains("<invoke")
                || lower.contains("<function")
                || lower.contains("</function");
    }

    private String extractToolName(String text) {
        if (text == null || text.isBlank()) {
            return "";
        }
        Matcher matcher = INVOKE_NAME_PATTERN.matcher(text);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        return "";
    }
}
