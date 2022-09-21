from transformers import AutoTokenizer
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead

from ml_preprocessing import HF_FILES
from ml_training import GPT2_HF_DEEPESP, GPT2_DEEPESP
import os

class Gpt2TrainerDeepesp:
    def __init__(self):

        self.train_path = HF_FILES + '/train.txt'
        self.test_path = HF_FILES + '/test.txt'
        self.model_files = GPT2_DEEPESP

        self.tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
        print('Tokenizer cargado')

    def load_dataset(self):
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=self.train_path,
            block_size=128)

        test_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=self.test_path,
            block_size=128)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False,
        )
        return train_dataset,test_dataset,data_collator

    def train_model(self):

        print('Cargando dataset...')
        train_dataset,test_dataset,data_collator = self.load_dataset()
        model = AutoModelWithLMHead.from_pretrained("DeepESP/gpt2-spanish")
        print('Modelo cargado')


        training_args = TrainingArguments(
            output_dir=self.model_files,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            eval_steps = 400,
            save_steps=800,
            warmup_steps=500,
            prediction_loss_only=True,
            )


        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        print('Fine tuning has started train')
        trainer.train()
        trainer.save_model(GPT2_HF_DEEPESP)

trainer = Gpt2TrainerDeepesp()
trainer.train_model()