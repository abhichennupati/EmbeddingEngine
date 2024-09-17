//
//  EmbeddingEngineTests.swift
//  EmbeddingEngineTests
//
//  Created by Abhiram Chennupati on 9/2/24.
//

import XCTest
import Tokenizers
@testable import EmbeddingManager // Replace with your actual module name

final class EmbeddingEngineTests: XCTestCase {
    
    // Helper method to load a sample vocab
    private func loadSampleVocab() -> [String: Int] {
        return [
            "[CLS]": 101,
            "[SEP]": 102,
            "example": 2003,
            "text": 2002
        ]
    }
    
    func testInitialization() async throws {
        let documents = ["This is a test document."]
        
        // Mock vocab loading and tokenizer setup
        let vocab = loadSampleVocab()
        let tokenizer = BertTokenizer(vocab: vocab, merges: nil, tokenizeChineseChars: true, bosToken: "[CLS]", eosToken: "[SEP]")
        
        // Initialize the EmbeddingEngine
        let embeddingEngine = try EmbeddingEngine(documents: documents)
        
        // Test that the EmbeddingEngine is initialized correctly
        XCTAssertEqual(embeddingEngine.documents.count, 1)
        XCTAssertEqual(embeddingEngine.INPUT_LENGTH, 128)
    }
    
    func testTokenization() async throws {
        let documents = ["Example text for tokenization."]
        let embeddingEngine = try EmbeddingEngine(documents: documents)
        
        let (tokenIds, wordIds) = embeddingEngine.tokenizeText(text: "Example text")
        
        XCTAssertEqual(tokenIds.count, 2)  // example, text
        XCTAssertEqual(wordIds.count, 2)   // example, text
    }
    
    func testGetEmbedding() async throws {
        let documents = ["Example text for testing embedding."]
        let embeddingEngine = try EmbeddingEngine(documents: documents)
        
        let (tokenIds, _) = embeddingEngine.tokenizeText(text: "Example text")
        let embeddings = try embeddingEngine.getEmbedding(tokens: tokenIds)
        
        XCTAssertEqual(embeddings.count, 2)  // Should be same size as input text length
        XCTAssertEqual(embeddings[0].count, 768)  // The embedding dimension should be 768
    }
    
    func testAddDocument() async throws {
        let documents = ["Example text for testing addDocument."]
        let embeddingEngine = try EmbeddingEngine(documents: documents)
        
        XCTAssertNoThrow(try embeddingEngine.addDocument(text: "Another document"))
        XCTAssertGreaterThan(embeddingEngine.index.count, 0)  // The FAISS index should have at least one entry
    }
    
    func testGetGlobalKeywords() async throws {
        let documents = ["This is a sample document for testing global keywords."]
        let embeddingEngine = try EmbeddingEngine(documents: documents)
        
        let globalKeywords = try embeddingEngine.getGlobalKeywords()
        
        XCTAssertGreaterThan(globalKeywords.count, 0)  // Global keywords should be generated
    }
    
    func testGetLocalKeywords() async throws {
        let documents = ["This is another sample document for local keyword extraction."]
        let embeddingEngine = try EmbeddingEngine(documents: documents)
        
        let localKeywords = try embeddingEngine.getLocalKeywords(text: "Sample text for testing local keywords.")
        
        XCTAssertGreaterThan(localKeywords.count, 0)  // Local keywords should be generated
    }
}
