import json
import openai
import pandas as pd
import numpy as np
import pickle
from transformers import GPT2TokenizerFast
from typing import List, Dict, Tuple
import os
#import argparse
import datetime as dt
import logging
from chatGPT_embedding_class import chatGPT_QA

logger = logging.getLogger('main_driver_logs')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def event_handler(event):
    try:
        # CHECKS
        print(event["context_filepath"])
        if not os.path.isfile(event["context_filepath"]):
            raise FileNotFoundError("File doesn't exist")

        if not event["context_filepath"].split(".")[-1]=="txt":
            raise TypeError("Not a text file")

        if not event["ques_filepath"].split(".")[-1]=="json":
            raise TypeError("Not a json file")

        time_start = dt.datetime.now()
        # Loop through files in specified location
        logger.info('Processing --- start %s', time_start)
        with open(event["context_filepath"]) as f:
            lines = f.readlines()
        context = ' '.join(lines)


        ques_list= json.load(open(event['ques_filepath']))

        chatGPT = chatGPT_QA(event['completion_model'], event['max_prompt_len'],event['temperature'], event['max_tokens'], 
                             event['qry_passage_match'], event['doc_embed_model'],event['qry_embed_model'], event['doc_stride'],
                             event['header'])
        ans_list = []
        for q in ques_list['ques']:
            print(q)
            ans = chatGPT.get_ans_long_context(context, q)
            ans_list.append(ans)

        print(ans_list)
        output_file = os.path.join(event['output_dir'], 'ans.txt') 
        
        q_list = ques_list['ques']
        with open(output_file, 'w') as f:
            for i in range(len(q_list)):
                f.write(f"Question {i+1}: {q_list[i]}\nAnswer: {ans_list[i]}\n\n")

#        logger.info('   --- File %s --- duration %s', dt.datetime.now() - time_start)

    except TypeError as e1:
        logger.exception(e1)

    except FileNotFoundError as e2:
        logger.exception(e2)    
    
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('doc_filepath')
#     parser.add_argument('ques_filepath')
#     parser.add_argument('output_dir')
#     parser.add_argument('doc_embed_model')
#     parser.add_argument('qry_embed_model')
#     parser.add_argument('completion_model')
#     parser.add_argument('max_prompt_len') 
# #     parser.add_argument('temperature')
# #     parser.add_argument('max_tokens')
#     args = parser.parse_args()

#     event = {'context_filepath': args.doc_filepath,
#              'ques_filepath': args.ques_filepath,
#              'output_dir': args.output_dir,
#              'doc_embed_model': args.doc_embed_model,
#              'qry_embed_model': args.qry_embed_model,
#              'completion_model': args.completion_model,
#              'max_prompt_len': args.max_prompt_len
# #              'temperature': args.temperature,
# #              'max_tokens': args.max_tokens
#             }
             
#     # Create the folders for the pages
#     os.makedirs(event["output_dir"], exist_ok=True)
    
#     event_handler(event)

