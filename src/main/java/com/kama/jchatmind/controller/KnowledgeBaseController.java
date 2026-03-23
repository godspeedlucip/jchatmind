package com.kama.jchatmind.controller;

import com.kama.jchatmind.model.common.ApiResponse;
import com.kama.jchatmind.model.request.CreateKnowledgeBaseRequest;
import com.kama.jchatmind.model.request.UpdateKnowledgeBaseRequest;
import com.kama.jchatmind.model.response.CreateKnowledgeBaseResponse;
import com.kama.jchatmind.model.response.GetKnowledgeBasesResponse;
import com.kama.jchatmind.service.KnowledgeBaseFacadeService;
import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
@AllArgsConstructor
public class KnowledgeBaseController {

    private final KnowledgeBaseFacadeService knowledgeBaseFacadeService;

    // List all knowledge bases
    @GetMapping("/knowledge-bases")
    public ApiResponse<GetKnowledgeBasesResponse> getKnowledgeBases() {
        return ApiResponse.success(knowledgeBaseFacadeService.getKnowledgeBases());
    }

    // Create a knowledge base
    @PostMapping("/knowledge-bases")
    public ApiResponse<CreateKnowledgeBaseResponse> createKnowledgeBase(@RequestBody CreateKnowledgeBaseRequest request) {
        return ApiResponse.success(knowledgeBaseFacadeService.createKnowledgeBase(request));
    }

    // Delete a knowledge base
    @DeleteMapping("/knowledge-bases/{knowledgeBaseId}")
    public ApiResponse<Void> deleteKnowledgeBase(@PathVariable("knowledgeBaseId") String knowledgeBaseId) {
        knowledgeBaseFacadeService.deleteKnowledgeBase(knowledgeBaseId);
        return ApiResponse.success();
    }

    // Update a knowledge base
    @PatchMapping("/knowledge-bases/{knowledgeBaseId}")
    public ApiResponse<Void> updateKnowledgeBase(
            @PathVariable("knowledgeBaseId") String knowledgeBaseId,
            @RequestBody UpdateKnowledgeBaseRequest request) {
        knowledgeBaseFacadeService.updateKnowledgeBase(knowledgeBaseId, request);
        return ApiResponse.success();
    }
}
