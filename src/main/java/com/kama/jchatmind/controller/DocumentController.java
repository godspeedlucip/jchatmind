package com.kama.jchatmind.controller;

import com.kama.jchatmind.model.common.ApiResponse;
import com.kama.jchatmind.model.request.CreateDocumentRequest;
import com.kama.jchatmind.model.request.UpdateDocumentRequest;
import com.kama.jchatmind.model.response.CreateDocumentResponse;
import com.kama.jchatmind.model.response.GetDocumentsResponse;
import com.kama.jchatmind.service.DocumentFacadeService;
import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
@AllArgsConstructor
public class DocumentController {

    private final DocumentFacadeService documentFacadeService;

    // List all documents
    @GetMapping("/documents")
    public ApiResponse<GetDocumentsResponse> getDocuments() {
        return ApiResponse.success(documentFacadeService.getDocuments());
    }

    // List documents by knowledge base id
    @GetMapping("/documents/kb/{kbId}")
    public ApiResponse<GetDocumentsResponse> getDocumentsByKbId(@PathVariable("kbId") String kbId) {
        return ApiResponse.success(documentFacadeService.getDocumentsByKbId(kbId));
    }

    // Create a document record only
    @PostMapping("/documents")
    public ApiResponse<CreateDocumentResponse> createDocument(@RequestBody CreateDocumentRequest request) {
        return ApiResponse.success(documentFacadeService.createDocument(request));
    }

    // Upload a document and create record
    @PostMapping("/documents/upload")
    public ApiResponse<CreateDocumentResponse> uploadDocument(
            @RequestParam("kbId") String kbId,
            @RequestParam("file") MultipartFile file) {
        return ApiResponse.success(documentFacadeService.uploadDocument(kbId, file));
    }

    // Delete a document
    @DeleteMapping("/documents/{documentId}")
    public ApiResponse<Void> deleteDocument(@PathVariable("documentId") String documentId) {
        documentFacadeService.deleteDocument(documentId);
        return ApiResponse.success();
    }

    // Update a document
    @PatchMapping("/documents/{documentId}")
    public ApiResponse<Void> updateDocument(
            @PathVariable("documentId") String documentId,
            @RequestBody UpdateDocumentRequest request) {
        documentFacadeService.updateDocument(documentId, request);
        return ApiResponse.success();
    }
}
