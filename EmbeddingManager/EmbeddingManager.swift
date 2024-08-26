//
//  EmbeddingManager.swift
//  EmbeddingManager
//
//  Created by Abhiram Chennupati on 8/21/24.
//

import Foundation
import CoreML

// Issues with the current implementation:
// * Isn't really optimal when things are deleted - embeddings can't really be moved backwards, so it will require recalculating everything after the deleted text.
// * How often should we be updating this index?

enum EmbeddingError: Error {
    case inputTooLong(actualLength: Int, maxLength: Int)
}


class EmbeddingManager {
    let INPUT_SIZE = 128
    let bertModel: DistilBERT
    
    init() throws {
        bertModel = try DistilBERT(configuration: .init())
    }
    
    /// Maps indexes of the text to their dirty table values. 0 means that they aren't used, -1 means that it's been changed and the embedding hasn't been updated, 1 means that the embedding is up to date.
    var DirtyTable = [Int: Int]()
    
    /// A reverse index, that maps the embedding values to their respective locations in the text.
    var EmbeddingToIndex = [[Float]: Int]()
    
    /// A mapping of a location in the text to its respective embedding.
    var IndexToEmbedding = [Int: [Float]]()
    
    
    
    // TODO integrate this with faiss for vector search!
    /// Takes in an array of tokens, generates an embedding, and adds it to the EmbeddingManager.
    func addEmbedding(tokens: [Int], startLocation: Int) throws {
        let embedding = try getEmbedding(tokens: tokens)
        EmbeddingToIndex[embedding] = startLocation
        IndexToEmbedding[startLocation] = embedding
        DirtyTable[startLocation] = 1
    }
    
    /// Takes an array of tokens (up to the model maximum input size), and returns an embedding representing that input token array.
    /// This masks shorter token sequences automaticaly, and concatenates the individual token embeddings by taking a mean.
    func getEmbedding(tokens: [Int]) throws -> [Float] {
        let length = tokens.count
        if length > INPUT_SIZE {
            throw EmbeddingError.inputTooLong(actualLength: length, maxLength: INPUT_SIZE)
        }
        // MLMultiArray is the required input type, and these also handle inputs that aren't the full size of the embedding.
        let shape = [0, INPUT_SIZE] as [NSNumber]
        let input = try MLMultiArray(shape: shape, dataType: .int32)
        let mask = try MLMultiArray(shape: shape, dataType: .int32)
        
        for x in 0..<length {
            let key = [1, x] as [NSNumber]
            input[key] = NSNumber(value: tokens[x])
            mask[key] = NSNumber(1)
        }
        
        let bertInput = DistilBERTInput(input_ids: input, attention_mask: mask)
        let output = try bertModel.prediction(input: bertInput)
        return aggregateEmbeddings(embeddings: output.last_hidden_state, length: length)
    }
    
    /// Takes in a MLMultiarray containing a list of embedding vectors, and returns the mean, as an array containing Floats.
    private func aggregateEmbeddings(embeddings: MLMultiArray, length: Int) -> [Float] {
        let hiddenSize = embeddings.shape[2].intValue
        var result = [Float](repeating: 0.0, count: hiddenSize)
        
        for i in 0..<hiddenSize {
            var sum: Float = 0.0
            for j in 0..<length {
                sum += embeddings[[0, j, i] as [NSNumber]].floatValue
            }
            result[i] = sum / Float(length)
        }
        return result
    }
    
}



