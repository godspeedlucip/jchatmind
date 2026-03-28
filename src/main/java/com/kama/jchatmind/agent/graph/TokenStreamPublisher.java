package com.kama.jchatmind.agent.graph;

public interface TokenStreamPublisher {

    String startAssistantStream();

    void appendAssistantStream(String messageId, String deltaContent);

    void finishAssistantStream(String messageId);
}

