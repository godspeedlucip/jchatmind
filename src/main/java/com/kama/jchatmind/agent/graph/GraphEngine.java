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

    /**
     * @param entryPoint     初始起始节点名称
     * @param onMessageGenerated 每当有新消息生成时（要发送给前端或者落库保存），进行的回调
     */
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
        int maxSteps = 15; // 设置边界，防止大模型陷入死循环无限来回调用

        log.info("[GraphEngine] ================= 开始执行 Agent 图引擎 =================");

        while (!"FINISH".equals(state.getNextNode()) && steps < maxSteps) {
            String currentNodeName = state.getNextNode();
            AgentNode node = nodes.get(currentNodeName);
            if (node == null) {
                log.error("[GraphEngine] 致命错误: 找不到被请求的路由节点 [{}]", currentNodeName);
                break;
            }

            // 执行前清空上一步产生的临时属性
            state.getAttributes().clear();

            // 进入节点执行
            state = node.process(state);

            // 处理节点执行过程中产生需要向外抛出的 Message
            handleGeneratedMessages(state);

            log.info("[GraphEngine] 路由流转: [{}] -> [{}]", currentNodeName, state.getNextNode());
            steps++;
        }

        if (steps >= maxSteps) {
            log.warn("[GraphEngine] 图引擎流转次数达到上限 ({})，强行终止。", maxSteps);
        }

        log.info("[GraphEngine] ================= 流转结束 =================");
    }

    private void handleGeneratedMessages(AgentGraphState state) {
        Map<String, Object> attrs = state.getAttributes();
        
        // 我们利用回调，将需要持久化和通过 SSE 下解的消息交给 JChatMind 外壳统一处理
        if (attrs.containsKey("latest_message")) {
            onMessageGenerated.accept((Message) attrs.get("latest_message"));
        }
        if (attrs.containsKey("latest_tool_call_msg")) {
            onMessageGenerated.accept((Message) attrs.get("latest_tool_call_msg"));
        }
        if (attrs.containsKey("latest_tool_response_msg")) {
            onMessageGenerated.accept((Message) attrs.get("latest_tool_response_msg"));
        }
    }
}
