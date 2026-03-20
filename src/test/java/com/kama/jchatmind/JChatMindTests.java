package com.kama.jchatmind;

import com.kama.jchatmind.agent.examples.JChatMindV1;
import com.kama.jchatmind.agent.examples.JChatMindV2;
import com.kama.jchatmind.agent.tools.test.CityTool;
import com.kama.jchatmind.agent.tools.test.DateTool;
import com.kama.jchatmind.agent.tools.test.WeatherTool;
import com.kama.jchatmind.service.RagService;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.deepseek.DeepSeekChatModel;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.Arrays;

@SpringBootTest
public class JChatMindTests {
    @Autowired
    @Qualifier("glm-4.6")
    ChatClient chatClient;
    @Qualifier("deepseek-chat")
    @Autowired
    private ChatClient chat;

    @Test
    public void testAgentV1(){
        JChatMindV1 jChatMindV1 = new JChatMindV1(
                "agent v1",
                "test agent",
                "请你扮演一个旅游顾问",
                chatClient,
                20,
                "default"

        );
        String response = jChatMindV1.chat("推荐我一些适合夏天去的旅游景点，并且给出理由。");
        System.out.println(response);

        response = jChatMindV1.chat("我刚才让你干什么？");
        System.out.println(response);
    }


    @Autowired
    private CityTool cityTool;

    @Autowired
    private DateTool dateTool;

    @Autowired
    private WeatherTool weatherTool;

    @Test
    public void testAgentV2() {
        // 准备工具回调
        ToolCallback[] toolCallbacks = MethodToolCallbackProvider.builder()
                .toolObjects(cityTool, dateTool, weatherTool)
                .build()
                .getToolCallbacks();

        // 创建 V2 实例
        JChatMindV2 agent = new JChatMindV2(
                "test-agent-v2",
                "测试 Agent V2",
                "你是一个智能助手，可以帮助用户查询天气、日期和城市信息。",
                chatClient,
                20,
                "test-session-v2",
                Arrays.asList(toolCallbacks)
        );

        // 测试需要调用工具的对话
        String userInput = "今天的天气怎么样？";
        String response = agent.chat(userInput);

        System.out.println("用户输入：" + userInput);
        System.out.println("AI 回复：" + response);
        System.out.println("对话历史长度：" + agent.getConversationHistory().size());
    }

    @Autowired
    RagService ragService;

    @Test
    public void  test_vector(){
        String text = "hello word";
        System.out.println(ragService.embed(text));
    }

}
