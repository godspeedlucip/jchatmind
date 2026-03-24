package com.kama.jchatmind.agent.graph;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.messages.Message;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

@Slf4j
public class GraphEngine {

    private final Map<String, AgentNode> nodes = new HashMap<>();
    private final String entryPoint;
    private final Consumer<Message> onMessageGenerated;

    public GraphEngine(String entryPoint, Consumer<Message> onMessageGenerated) {
        this.entryPoint = entryPoint;
        this.onMessageGenerated = onMessageGenerated;
    }

    public void addNode(AgentNode node) {
        nodes.put(node.getName(), node);
    }

    public void run(AgentGraphState state) {
        state.setNextNode(entryPoint);
        int steps = 0;
        int maxSteps = 40;

        log.info("[GraphEngine] ================= Agent graph start =================");

        while (!"FINISH".equals(state.getNextNode()) && steps < maxSteps) {
            String currentNodeName = state.getNextNode();
            AgentNode node = nodes.get(currentNodeName);
            if (node == null) {
                log.error("[GraphEngine] Unknown node: [{}]", currentNodeName);
                break;
            }

            state.getAttributes().put("step_count", steps + 1);
            state = node.process(state);

            handleGeneratedMessages(state);

            log.info("[GraphEngine] Route: [{}] -> [{}]", currentNodeName, state.getNextNode());
            steps++;
        }

        if (steps >= maxSteps) {
            log.warn("[GraphEngine] Reached max steps ({})", maxSteps);
        }

        log.info("[GraphEngine] ================= Agent graph finished =================");
    }

    private void handleGeneratedMessages(AgentGraphState state) {
        Map<String, Object> attrs = state.getAttributes();

        emitIfPresent(attrs, "latest_message");
        emitIfPresent(attrs, "latest_tool_call_msg");
        emitIfPresent(attrs, "latest_tool_response_msg");
    }

    private void emitIfPresent(Map<String, Object> attrs, String key) {
        Object value = attrs.get(key);
        if (value instanceof Message) {
            onMessageGenerated.accept((Message) value);
        }
        attrs.remove(key);
    }
}
