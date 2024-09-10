import tensorflow as tf
import tensorflow_hub as hub
from markdown_plain_text.extention import convert_to_plain_text
import dtlpy as dl
import json
import logging

logger = logging.getLogger('google-NNLM')


@dl.Package.decorators.module(description='Model Adapter for Goggle NNLM text embedding',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        model_url = self.configuration.get('model_url', 'https://tfhub.dev/google/nnlm-en-dim128/2')
        self.model = hub.load(model_url)

    def prepare_item_func(self, item):
        return item

    def embed(self, batch, **kwargs):
        embedings = []

        for item in batch:
            filename = item.download(overwrite=True)
            logger.info(f'Downloaded item: {filename}')
            if item.mimetype == 'text/plain':
                with open(filename, 'r') as f:
                    text = f.read()
                    text = text.replace('\n', ' ')
            elif item.mimetype == 'text/markdown':
                with open(filename, 'r') as f:
                    text = f.read()
                    text = convert_to_plain_text(text)
                    text = text.replace('\n', ' ')
            elif item.mimetype == 'application/json':
                buffer = json.loads(filename)
                _, prompt_content = list(buffer['prompts'].items())[0]
                _, question = list(prompt_content.items())[0]

                if question["mimetype"] == dl.PromptType.TEXT:
                    text = question["value"]
            else:
                raise ValueError(f'Unsupported mimetype: {item.mimetype}')

            logger.info(f'Extracted text from item: {item.id}')

            if text is not None:
                try:
                    embeding = self.model([text]).numpy().tolist()[0]
                    embedings.append(embeding)
                    logger.info(f'Extracted embeddings for from text: {text}')
                except Exception as e:
                    logger.error(f'Failed to extract embeddings from text: {text}')
                    logger.error(e)
                    raise e

            else:
                logger.error(f'No text found in item: {item.id}')
                raise ValueError(f'No text found in item: {item.id}')

        return embedings
