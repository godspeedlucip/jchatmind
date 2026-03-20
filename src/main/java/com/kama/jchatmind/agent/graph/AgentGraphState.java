package com.kama.jchatmind.agent.graph;

import lombok.Data;
import org.springframework.ai.chat.messages.Message;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 黑板模式下的全局状态树，在各个多智能体节点中流转
 */
@Data
public class AgentGraphState {

    // 当前上下文的所有消息流
    private List<Message> messages;

    // 下一次循环应该进入哪个节点处理
    private String nextNode;

    // 用于在不同 Agent 之间传递临时数据，例如工具返回值
    private Map<String, Object> attributes = new HashMap<>();
}
