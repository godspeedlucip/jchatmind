package com.kama.jchatmind.agent.graph;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.tool.ToolCallback;

import java.util.List;
import java.util.Set;

public class SqlWorkerNode extends AbstractWorkerNode {

    public SqlWorkerNode(ChatClient chatClient,
                         ChatOptions chatOptions,
                         List<ToolCallback> tools,
                         Set<String> availableToolNames) {
        super(chatClient, chatOptions, tools, availableToolNames);
    }

    @Override
    public String getName() {
        return "SQL_WORKER";
    }
}

