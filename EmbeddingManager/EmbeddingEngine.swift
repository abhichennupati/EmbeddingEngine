//
//  EmbeddingEngine.swift
//  EmbeddingManager
//
//  Created by Abhiram Chennupati on 8/30/24.
//

import Foundation
import CoreML

import SwiftFaiss
import Tokenizers

enum EmbeddingError: Error {
    case inputTooLong(actualLength: Int, maxLength: Int)
}

enum GeneralError: Error {
    case tokenizerError(tokens: [String])
}

class EmbeddingEngine {
    let bertModel: DistilBERT
    var index: BaseIndex
    let INPUT_LENGTH = 128
    var tokenizer: BertTokenizer
    var wordToInt: [String: Int]
    var intToWord: [Int: String]
    var documents: [String] // Should probably change this in the future, for now documents are just represented as a list of String objects
    var embeddingToWord: [[Float]: String]
    
    init(documents: [String]) throws {
        self.documents = documents
        self.intToWord = [:]
        self.wordToInt = [:]
        self.embeddingToWord = [:]
        //self.index = try IVFFlatIndex(quantizer: FlatIndex(d: 768, metricType: .l2), d: 768, nlist: 1) // the initial nlist value might need to be tweaked depending on performance - a good heuristic for this is sqrt(# of vectors), we will also need to dynamically adjust this if too many vectors are added. might also need to tune metrictype

        self.index = try FlatIndex(d: 768, metricType: .l2) // may have to change to a different type of index later.
        self.bertModel = try DistilBERT(configuration: .init())
        
        let vocab = {
                    let url = URL(fileURLWithPath: "/Users/achennupati/Downloads/bert-vocab.txt")
                    let vocabTxt = try! String(contentsOf: url)
                    let tokens = vocabTxt.split(separator: "\n").map { String($0) }
                    var vocab: [String: Int] = [:]
                    for (i, token) in tokens.enumerated() {
                        vocab[token] = i
                    }
                    return vocab
                }()
        
        
        
        
//        let vocabURL = URL(fileURLWithPath: "/Users/achennupati/Downloads/bert-vocab.txt")
//        let vocabData = try Data(contentsOf: vocabURL)
//        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        self.tokenizer = BertTokenizer(vocab: vocab, merges: nil, tokenizeChineseChars: true, bosToken: "[CLS]", eosToken: "[SEP]")
        
        for document in self.documents {
            try addDocument(text: document)
        }
    }
    
    
    /// Takes in an input sequence of tokens, and returns an array of embeddings, representing each token
    func getEmbedding(tokens: [Int]) throws -> [[Float]] {
        if tokens.count > INPUT_LENGTH {
            throw EmbeddingError.inputTooLong(actualLength: tokens.count, maxLength: INPUT_LENGTH)
        }
        
        // Building a mask, to handle inputs that are less than the EMBEDDING_LENGTH param. MLMultiArray initializes to 0, so we don't have to set those to zero
        let shape = [1, INPUT_LENGTH] as [NSNumber]
        let input = try MLMultiArray(shape: shape, dataType: .int32)
        let mask = try MLMultiArray(shape: shape, dataType: .int32)
        for x in 0..<tokens.count {
            let key = [0, x] as [NSNumber]
            input[key] = NSNumber(value: tokens[x])
            mask[key] = NSNumber(1)
        }
        
        let bertInput = DistilBERTInput(input_ids: input, attention_mask: mask)
        let output = try bertModel.prediction(input: bertInput)
        
        // Convert MLMultiArray to [[Float]] directly within the function
        
        let hiddenSize = output.last_hidden_state.shape[2].intValue // 768
        
        let embeddingArray: [[Float]] = (0..<tokens.count).map { i in
            (0..<hiddenSize).map { j in
                output.last_hidden_state[[0, i, j] as [NSNumber]].floatValue
            }
        }
        return embeddingArray
    }
    
    /// Takes in a string of text, and returns a tuple, the first index being a list of tokens, and the second index being a list of wordToInt mappings, that are stored in the two dictionaries.
//    func tokenizeText(text: String) -> ([Int], [Int]) {
//        let tokens = tokenizer.tokenize(text: text)
//        
//        var ids: [Int] = []
//        var idWord: [Int] = []
//        
//        for token in tokens {
//            if let id = tokenizer.convertTokenToId(token) {
//                ids.append(id)
//                if let wordID = wordToInt[token] {
//                    idWord.append(wordID)
//                } else {
//                    let newID = wordToInt.keys.count
//                    wordToInt[token] = newID
//                    intToWord[newID] = token
//                    idWord.append(newID)
//                }
//            }
//        }
//        return (ids, idWord)
//    }

    func tokenizeText(text: String) -> ([Int], [String]) {
        let tokens = tokenizer.tokenize(text: text)

        var ids: [Int] = []
        var words: [String] = []

        for token in tokens {
            if let id = tokenizer.convertTokenToId(token) {
                ids.append(id)
                words.append(token)
            }
        }
        return (ids, words)
    }

    
    /// takes in a document (represented as a string) and generates embeddings for that text, putting each embedding in the FAISS Index with some
    func addDocument(text: String) throws {
        let tokenized = tokenizeText(text: text)
        let tokens = tokenized.0
        let words = tokenized.1
        
        // Process tokens in chunks of INPUT_LENGTH
        for i in stride(from: 0, to: tokens.count, by: INPUT_LENGTH) {
            // Get the current chunk of tokens
            let endIndex = min(i + INPUT_LENGTH, tokens.count)
            let tokenChunk = Array(tokens[i..<endIndex])
            
            let embeddings = try getEmbedding(tokens: tokenChunk)
            // try index.add(embedding, ids: wordIdChunk)
            try index.add(embeddings)
            for i in 0..<embeddings.count {
                embeddingToWord[embeddings[i]] = words[i]
            }
        }
    }
    
    private func retrieveEmbeddingsFromIndex() throws -> [[Float]] {
        var embeddings: [[Float]] = []
        
        for i in 0..<index.count {
            let embedding = try index.reconstruct(i)
            embeddings.append(embedding)
        }
        
        return embeddings
    }

    private func clusterStoredEmbeddings(numberOfClusters: Int) throws -> [[Float]] {
        // get embeddings from FAISS index
        let embeddings = try retrieveEmbeddingsFromIndex()
        
        let d = index.d
        let centroids = try kMeansClustering(embeddings, d: d, k: numberOfClusters)
        
        return centroids
    }

    // TODO check if this is actually right? and also move the functionality within the for loop to another local keyword extraction function maybe
    func getGlobalKeywords() throws -> [String] {
        let k = min(5, index.count)
        let clusters = try clusterStoredEmbeddings(numberOfClusters: k)
        
        var keywords: [String] = []
        
        for cluster in clusters {
            let bestMatch = try index.search([cluster], k: 1)
            print(bestMatch)
            let embedding = try index.reconstruct(bestMatch.labels[0][0])
            if let word = embeddingToWord[embedding] {
                keywords.append(word)
            }
        
        }
        return keywords
    }
    
    // TODO check if this works, this is straight from ChatGPT
    func getLocalKeywords(text: String) throws -> [String] {
        // Step 1: Tokenize the input text and get the embeddings
        let tokenized = tokenizeText(text: text)
        let tokens = tokenized.0
        
        
        var embeddings: [[Float]] = []
        
        // Step 2: Generate embeddings for the entire document
        for i in stride(from: 0, to: tokens.count, by: INPUT_LENGTH) {
            let endIndex = min(i + INPUT_LENGTH, tokens.count)
            let tokenChunk = Array(tokens[i..<endIndex])
            let embeddingChunk = try getEmbedding(tokens: tokenChunk)
            embeddings.append(contentsOf: embeddingChunk)
        }
        
        // Step 3: Perform k-means clustering on the embeddings within the document
        let numberOfClusters = min(5, embeddings.count) // Or use a different value or heuristic
        let clusters = try kMeansClustering(embeddings, d: 768, k: numberOfClusters)
        
        var keywords: [String] = []
        
        // Step 4: For each cluster, find the closest word in the FAISS index
        for cluster in clusters {
            let bestMatch = try index.search([cluster], k: 1)
            if let word = intToWord[bestMatch.labels[0][0]] {
                keywords.append(word)
            }
        }
        
        return keywords
    }
}

