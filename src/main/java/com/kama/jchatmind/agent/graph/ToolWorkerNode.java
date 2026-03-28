package com.kama.jchatmind.agent.graph;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.tool.ToolCallback;

import java.util.List;
import java.util.Set;

public class ToolWorkerNode extends AbstractWorkerNode {

    public ToolWorkerNode(ChatClient chatClient,
                          ChatOptions chatOptions,
                          List<ToolCallback> tools,
                          Set<String> availableToolNames,
                          TokenStreamPublisher tokenStreamPublisher) {
        super(chatClient, chatOptions, tools, availableToolNames, tokenStreamPublisher);
    }

    @Override
    public String getName() {
        return "TOOL_WORKER";
    }
}
