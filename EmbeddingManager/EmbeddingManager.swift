//
//  EmbeddingManager.swift
//  EmbeddingManager
//
//  Created by Abhiram Chennupati on 8/21/24.
//

import Foundation


// Issues with the current implementation:
// * Isn't really optimal when things are deleted - embeddings can't really be moved backwards, so it will require recalculating everything after the deleted text.
// * How often should we be updating this index?

class EmbeddingManager {
    
    
    /// Maps indexes of the text to their dirty table values. 0 means that they aren't used, -1 means that it's been changed and the embedding hasn't been updated, 1 means that it has been updated.
    var DirtyTable = [Int: Int]()
    
    /// A reverse index, that maps the embedding values to their respective locations in the text.
    var EmbeddingToIndex = [Float: Int]()
    
    /// A mapping of a location in the text to its respective embedding.
    var IndexToEmbedding = [Int: Float]()
    
    
    func generateEmbedding(text: String, index: Int) {
        
    }
    
    
    
}
