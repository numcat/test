import os
import json
from retrieval_trainer import DocumentGroundedDialogRetrievalTrainer

from modelscope.hub.snapshot_download import snapshot_download
with open('dev.json', encoding='utf-8') as f_in:
    with open('input.jsonl', 'w', encoding='utf-8') as f_out:
        for line in f_in.readlines():
            sample = json.loads(line)
            sample['positive'] = ''
            sample['negative'] = ''
            #f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            f_out.write(json.dumps(sample) + '\n')

with open('input.jsonl', encoding='utf-8') as f:
    eval_dataset = [json.loads(line) for line in f.readlines()]

all_passages = []
for file_name in ['fr', 'vi']:
    with open(f'all_passages/{file_name}.json', encoding='utf-8') as f:
        all_passages += json.load(f)

#cache_path = './DAMO_ConvAI/nlp_convai_retrieval_pretrain'
cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir='./')
trainer = DocumentGroundedDialogRetrievalTrainer(
    model=cache_path,
    train_dataset=None,
    eval_dataset=eval_dataset,
    all_passages=all_passages
)

trainer.evaluate(
    checkpoint_path=os.path.join(trainer.model.model_dir,
                                 'finetuned_model.bin'))
