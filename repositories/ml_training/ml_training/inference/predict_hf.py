from transformers import pipeline


class PredictGPT2:
    def __init__(self, model_name):
        self.model_name = model_name
        self.max_length = 100
        self.num_return_sequences = 1
        self.pipeline_type = 'text-generation'

        if self.model_name == 'datificate':
            print('Cargando datificate/gpt2-small-spanish')
            self.model = 'datificate/gpt2-small-spanish'
            self.generator = pipeline(self.pipeline_type, model=self.model)
        elif self.model_name == 'deepesp':
            print('Cargando DeepESP/gpt2-spanish')
            self.model = 'DeepESP/gpt2-spanish'
            self.generator = pipeline(self.pipeline_type, model=self.model)
        elif self.model_name == 'pretrained':
            print('Cargando gpt-2 tolkien pretrained')
            self.model = './gpt2-fine_tuned'
            self.generator = pipeline("text-generation",
                            model='./gpt2-fine_tuned',
                            tokenizer="datificate/gpt2-small-spanish")
        else:
            print('Inténtalo con uno de los siguientes modelos: datificate, deepesp, pretrained')
            print('Usa el nombre de los modelos siempre en minúsculas')

    def predict(self, seed_text):
        if self.model_name == 'datificate':
            generate_sentence = self.generator(seed_text, max_length=self.max_length, num_return_sequences=self.num_return_sequences)
            return generate_sentence[0]['generated_text']
        elif self.model_name == 'deepesp':
            generate_sentence = self.generator(seed_text, max_length=self.max_length, num_return_sequences=self.num_return_sequences)
            return generate_sentence[0]['generated_text']
        elif self.model_name == 'pretrained':
            return self.generator(seed_text)[0]['generated_text']
        else:
            print('Inténtalo con uno de los siguientes modelos: datificate, deepesp, pretrained')
            print('Usa el nombre de los modelos siempre en minúsculas')