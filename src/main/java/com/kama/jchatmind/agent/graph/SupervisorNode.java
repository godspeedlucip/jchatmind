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
                你是一个高层任务调度主管 (Supervisor)。
                你的直属下级为：
                1. WORKER: 一个可以直接执行获取时间、邮件、知识库以及各种外部系统调用的全能机器人。
                
                请仔细分析用户的指令内容和当前的上下文：
                1. 如果你发现用户在询问具体的天气、时间、文件信息、数据库信息或者你想帮他发邮件，或者你感觉需要最新的外部事实资料，请**坚决且只输出英文单词："WORKER"**，不要加任何其他字符，这会把它路由给下级工具人执行。
                2. 如果你认为根据上下文已经得到了最终答案，或者仅仅是日常问候（如：你好，谢谢），不需要查任何外部工具，请综合上下文给用户最终的回答，并在你的回答末尾加上标记 "[FINISH]"。
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
        String text = output.getText();
        
        if (text != null && text.contains("WORKER")) {
            // 调度给 Worker，不需要保存这段内心独白给用户看
            state.setNextNode("WORKER");
        } else {
            // 生成最终答案
            String cleanText = text != null ? text.replace("[FINISH]", "").trim() : "";
            AssistantMessage finalMsg = new AssistantMessage(cleanText);
            
            // 写入全局记忆
            state.getMessages().add(finalMsg);
            // 通过属性抛出给外层引擎落盘
            state.getAttributes().put("latest_message", finalMsg);
            
            state.setNextNode("FINISH");
        }
        return state;
    }
}
