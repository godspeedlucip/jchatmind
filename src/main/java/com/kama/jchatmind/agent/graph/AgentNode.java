package com.kama.jchatmind.agent.graph;

/**
 * 智能体图执行引擎中的独立节点（例如：Supervisor 节点，或者 Worker 节点）
 */
public interface AgentNode {

    /**
     * @return 节点的唯一名称，用于在 State 中进行路由
     */
    String getName();

    /**
     * 接收全局状态，进行本节点的专属处理，并根据结果决定路由方向
     * @param state 传入的全局状态
     * @return 处理完毕后的全局状态
     */
    AgentGraphState process(AgentGraphState state);
}
