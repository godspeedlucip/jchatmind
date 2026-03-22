package com.kama.jchatmind.agent.graph;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;

@Slf4j
public class SupervisorNode implements AgentNode {

    private final ChatClient chatClient;
    private final ChatOptions chatOptions;

    public SupervisorNode(ChatClient chatClient, ChatOptions chatOptions) {
        this.chatClient = chatClient;
        this.chatOptions = chatOptions;
    }

    @Override
    public String getName() {
        return "SUPERVISOR";
    }

    @Override
    public AgentGraphState process(AgentGraphState state) {
        log.info("[GraphEngine] 节点执行: {}", getName());

        String supervisorPrompt = """
                You are the supervisor of a multi-agent system.

                Your child worker is:
                - WORKER: can call tools (time, weather, files, DB, external APIs, email, etc.).

                Routing policy (strict):
                1) If external facts/tools/actions are needed, output exactly: WORKER
                2) If you can answer directly from current context, output the final answer and append [FINISH] at the end.

                Important constraints:
                - Do not output role introduction, meta-planning, or chain-of-thought.
                - If uncertain about whether tools are needed, choose WORKER.
                """;

        Prompt prompt = Prompt.builder()
                .chatOptions(chatOptions)
                .messages(state.getMessages())
                .build();

        ChatResponse response = chatClient.prompt(prompt)
                .system(supervisorPrompt)
                .call()
                .chatClientResponse()
                .chatResponse();

        AssistantMessage output = response.getResult().getOutput();
        String text = output.getText() != null ? output.getText().trim() : "";
        String upper = text.toUpperCase();
        String lower = text.toLowerCase();

        boolean routeWorker = upper.equals("WORKER") || upper.contains("WORKER");
        boolean hasFinishMarker = text.contains("[FINISH]");
        boolean looksLikeMetaIntro =
                lower.contains("supervisor")
                        || text.contains("调度主管")
                        || text.contains("高层任务调度")
                        || text.contains("路由")
                        || text.contains("WORKER:");

        // Only finalize when explicitly marked as final answer.
        if (hasFinishMarker && !routeWorker && !looksLikeMetaIntro) {
            String cleanText = text.replace("[FINISH]", "").trim();
            if (cleanText.isEmpty()) {
                state.setNextNode("WORKER");
                return state;
            }
            AssistantMessage finalMsg = new AssistantMessage(cleanText);

            state.getMessages().add(finalMsg);
            state.getAttributes().put("latest_message", finalMsg);
            state.setNextNode("FINISH");
            return state;
        }

        // Fallback to WORKER for all ambiguous outputs.
        state.setNextNode("WORKER");
        return state;
    }
}
