
import openai
import pandas as pd
import numpy as np
import pickle
from transformers import GPT2TokenizerFast
from typing import List, Dict, Tuple
import logging
logger = logging.getLogger('chatGPT_QA_logs')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class chatGPT_QA:
    
    def __init__(self, COMPLETIONS_MODEL,
                 MAX_SECTION_LEN = 4000, temperature=0.0, max_tokens=300, qry_passage_match = True, 
                 DOC_EMBEDDINGS_MODEL ="text-search-curie-doc-001", QUERY_EMBEDDINGS_MODEL="text-search-curie-query-001", 
                 doc_stride = 100, header = ''):

        self.COMPLETIONS_MODEL = COMPLETIONS_MODEL
        self.MAX_SECTION_LEN = MAX_SECTION_LEN
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.qry_passage_match = qry_passage_match
        self.DOC_EMBEDDINGS_MODEL = DOC_EMBEDDINGS_MODEL
        self.QUERY_EMBEDDINGS_MODEL = QUERY_EMBEDDINGS_MODEL
        self.doc_stride = doc_stride
        self.header = header

    def get_embedding(self, text: str, model: str) -> List[float]:
        result = openai.Embedding.create(
          model=model,
          input=text)
        return result["data"][0]["embedding"]
    
    def vector_similarity(self, x: List[float], y: List[float]) -> float:
        """
        We could use cosine similarity or dot product to calculate the similarity between vectors.
        In practice, we have found it makes little difference. 
        """
        return np.dot(np.array(x), np.array(y))

    def get_query_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text, self.QUERY_EMBEDDINGS_MODEL)

    def compute_doc_embeddings(self, df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        
        return {
            idx: self.get_embedding(r.content.replace("\n", " "), self.DOC_EMBEDDINGS_MODEL) for idx, r in df.iterrows()
        }
    
    def order_document_sections_by_query_similarity(self, query: str, contexts: Dict[Tuple[str, str], np.array]) -> List[Tuple[float, Tuple[str, str]]]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 

        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_query_embedding(query)

        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)

        return document_similarities
    
    def construct_prompt(self, question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
        """
        Fetch relevant 
        """
        
        # class variables
        SEPARATOR = "\n* "
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        separator_len = len(tokenizer.tokenize(SEPARATOR))
    
        most_relevant_document_sections = self.order_document_sections_by_query_similarity(question, context_embeddings)

        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.        
            document_section = df.loc[section_index]

            chosen_sections_len += document_section.tokens + separator_len
            #print(chosen_sections_len, self.MAX_SECTION_LEN)
            if chosen_sections_len > self.MAX_SECTION_LEN:
                break

            chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))

        # Useful diagnostic information
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))

        return self.header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    
    
    def construct_prompt_from_ans(self, question, context):
        context_len = len(context)
        ques_len = len(question)
        prompt_len = context_len + ques_len
        prompts = []
        max_model_chars = 4 * self.MAX_SECTION_LEN
        
        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
                 
        if prompt_len< max_model_chars:
            prompts = [self.header + context + "\n\n Q: " + question + "\n A:"]
        else:
            no_para = int((context_len + ques_len)/max_model_chars)+1
            index_div = int(context_len/no_para)
            parts = []
            for i in range(0, (context_len-index_div), index_div):
                if(i-self.doc_stride)<=0:
                    st = 0
                else:
                    st = i-self.doc_stride    
                pp = context[st:i+index_div]
                parts.append(pp)
            prompts = [self.header + p + "\n\n Q: " + question + "\n A:" for p in parts]
        
        #print(len(parts), len(prompts))
        ans_list =[]
        COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
        "model": self.COMPLETIONS_MODEL,
        }
        for p in prompts:
            response = openai.Completion.create(
              prompt=p,
              **COMPLETIONS_API_PARAMS
            )
            ans_list.append(response['choices'][0]['text'])
            
        ans_list_unique = list(set(ans_list))
        final_prompt = '\n'.join(ans_list_unique)
        return self.header + final_prompt + "\n\n Q: " + question + "\n A:"  


    def get_ans_long_context(self, context: str, ques: str) -> str:
        '''
        
        '''
    
        COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
        "model": self.COMPLETIONS_MODEL,
        }   
        
        if self.qry_passage_match == True:
            # context embedding
            # break down the context into paragraphs
            para = context.split('\n')
            para_df = pd.DataFrame(para, columns =['content']) # 20 as GPT is asking for payment for full doc. Remove it..
            para_df['tokens'] = para_df['content'].apply(lambda x: len(x.split(' ')))
            # calculate embeddings for paragraphs
            para_embed = self.compute_doc_embeddings(para_df)

            # calculate similarity against query
            # prompt
            prompt = self.construct_prompt(ques,para_embed, para_df)
        else:
            prompt = self.construct_prompt_from_ans(ques, context)
            
        print(prompt)
        print(len(prompt.split(' ')))
        print(COMPLETIONS_API_PARAMS)

        response = openai.Completion.create(
                    prompt=prompt,
                    **COMPLETIONS_API_PARAMS
                )

        return response["choices"][0]["text"].strip(" \n")
    
    