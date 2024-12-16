
import json, csv, pymupdf4llm

import torch
from torch.utils.data import Dataset

from collections import defaultdict
pymu_cache = defaultdict(str)

from tqdm import tqdm
import os
import sys

# 부모 부모 폴더의 Retriever.py 파일 임포트
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from Retriever import get_marktable_relevant

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = """
        You are a helpful AI assistant specializing in finance. Please answer the user's questions kindly and accurately based on the given financial documents and tables. 
        당신은 재정 분야를 전문으로 하는 유능한 AI 어시스턴트입니다. 주어진 재정 문서와 표를 바탕으로 사용자의 질문에 정확하게 답변해주세요.
        """
        data = []

        with open(fname, 'r', newline='', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            title = next(reader)  
            
            temp = list(reader)  

            for info in temp:
                file_path = info[-3].split("/")[-1]
                question = info[-2]
                answer = info[-1]
   
                data.append([file_path,question,answer])

        def make_chat(example):
            file_path = example[0]
            question = example[1]

            input_sentence = ["\n[재정문서]"]

            tables, markdowns = get_marktable_relevant(pdf_path = file_path.replace(".pdf",""), query = question)
            for markdown in markdowns:
                input_sentence.append(markdown)

            input_sentence.append("\n[표]")

            for table in tables:
                input_sentence.append(table)

            input_sentence.append("\n[사용자 질문]")
            input_sentence.append(question)
            input_sentence = "\n".join(input_sentence)

            input_sentence = input_sentence + "\n\n"

            return input_sentence
        
        print("데이터 전처리 중 (학습할 Q-A Pair 생성)...")
        for example in tqdm(data):
            file_path = example[0]
            question = example[1]
            answer = example[2]

            chat = make_chat(example)

            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]

            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = answer
            if target != "":
                target += tokenizer.eos_token

            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
