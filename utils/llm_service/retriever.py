import os
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import json
import faiss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

import torch
from typing import List, Tuple, Dict

load_dotenv()

class Retriever:
    def __init__(self):
        self.RETRIEVER_SCORE_THRESHOLD = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", 0.6))
        self.RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 10))
        self.RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))
        self.device = os.getenv("DEFAULT_CUDA_DEVICE", "cuda:0")

        self.retriever_tokenizer = AutoTokenizer.from_pretrained(os.getenv("LOCAL_EMBEDDING_MODEL_PATH"), trust_remote_code=True, use_safetensors=True)
        self.retriever_model = AutoModel.from_pretrained(os.getenv("LOCAL_EMBEDDING_MODEL_PATH"), trust_remote_code=True, use_safetensors=True).to(self.device)
        self.retriever_model.eval()
        
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(os.getenv("LOCAL_RERANKER_MODEL_PATH"))
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(os.getenv("LOCAL_RERANKER_MODEL_PATH")).to(self.device)
        self.reranker_model.eval()
        
        self.index = None
        with open(os.getenv("KB_PATH"), 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        
        self.documents = [k['page_content'] for k in self.knowledge_base]
        
        self.index_path = os.getenv("FAISS_STORE_PATH")
        try:
            self.load_index(self.index_path)
        except Exception as e:
            print(f"加载索引失败: {e}。开始构建新索引...")
            self.build_index()
            self.save_index(self.index_path)

    def _get_embeddings_batch(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size), desc="Generating Embeddings"):
            batch_sentences = sentences[i:i + batch_size]
            
            with torch.no_grad():
                encoded_input = self.retriever_tokenizer(
                    batch_sentences, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=512
                )
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

                model_output = self.retriever_model(**encoded_input)

                sentence_embeddings = model_output[0][:, 0]
                normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                all_embeddings.append(normalized_embeddings.cpu().numpy())
        
        if not all_embeddings:
            hidden_size = self.retriever_model.config.hidden_size
            return np.empty((0, hidden_size), dtype=np.float32)

        return np.vstack(all_embeddings)
    
    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        with torch.no_grad():
            encoded_input = self.retriever_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            model_output = self.retriever_model(**encoded_input)
            
            sentence_embeddings = model_output[0][:, 0]
            normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            return normalized_embeddings.cpu().numpy()

    def build_index(self):
        doc_embeddings = self._get_embeddings_batch(self.documents)
        
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) 
        
        self.index.add(doc_embeddings.astype('float32'))
    
    def retrieve(self, query: str) -> List[Tuple[int, float]]:
        query_embedding = self._get_embeddings([query])
        
        scores, indices = self.index.search(query_embedding.astype('float32'), self.RETRIEVE_TOP_K)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and score >= self.RETRIEVER_SCORE_THRESHOLD:
                results.append((int(idx), float(score)))
        
        return results
    
    def rerank(self, query: str, retrieved_docs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if not retrieved_docs:
            return []
        
        pairs = []
        doc_indices = []
        for idx, retriever_score in retrieved_docs:
            doc_text = self.documents[idx]
            pairs.append([query, doc_text])
            doc_indices.append((idx, retriever_score))
        
        with torch.no_grad():
            features = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)
            
            scores = self.reranker_model(**features).logits
            reranker_scores = torch.sigmoid(scores).squeeze(-1).cpu().numpy()
        
        reranked_results = []
        for (idx, retriever_score), reranker_score in zip(doc_indices, reranker_scores):
            reranked_results.append((idx, float(retriever_score), float(reranker_score)))
        
        reranked_results.sort(key=lambda x: x[2], reverse=True)
        
        return reranked_results[:self.RERANK_TOP_K]
    
    def search(self, query: str) -> List[Dict]:
        retrieved_results = self.retrieve(query)
        if not retrieved_results:
            return []
        
        reranked_results = self.rerank(query, retrieved_results)
        final_results = []
        for idx, retriever_score, reranker_score in reranked_results:
            final_results.append({
                'document_id': idx,
                'document': self.knowledge_base[idx],
                'retriever_score': retriever_score,
                'reranker_score': reranker_score,
                'combined_score': (retriever_score + reranker_score) / 2 
            })
        
        return final_results

    def save_index(self, filepath: str):
        if self.index is not None:
            faiss.write_index(self.index, filepath)

    def load_index(self, filepath: str):
        self.index = faiss.read_index(filepath)
