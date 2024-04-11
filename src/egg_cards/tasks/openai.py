import luigi
import base64
import requests
import json
import requests_cache
import pandas as pd
from pathlib import Path

from egg_cards.config import CACHE_DIR, RAW_DATA_DIR, INTERMEDIATE_DATA_DIR, OPENAI_API_KEY
from egg_cards.utils import camelcase

requests_cache.install_cache(CACHE_DIR / 'openai')

class BaseTask(luigi.Task):
    
    force = luigi.BoolParameter(default=False, significant=False)

    def complete(self):
        if self.force:
            return False
        
        return super().complete() 

class OpenAIImageTask(BaseTask):

    image_path = luigi.PathParameter()  

    def output(self):
        file_name = self.image_path.stem.lower()  
        return luigi.LocalTarget(INTERMEDIATE_DATA_DIR / f'{file_name}.json')  
    
    @staticmethod
    def encode(image_path: Path):
        with image_path.open("rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')    

    def run(self):
        response = self.call_api()

        if response.get('error'):
            raise Exception(response['error']['message'])

        with self.output().open('w') as f:
            json.dump(response, f)

    def call_api(self):
        base64_image = self.encode(self.image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "Extract data from this image. Fields to include are: Registration Number, Locality, Collector, Date, Set Mark and Number of eggs. The values are written inside the field boxes. Structure the response as JSON"
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()
    
class OpenAITask(BaseTask):
    image_paths = luigi.ListParameter(significant=False)  
    limit = luigi.IntParameter(default=10)  

    def requires(self):  
        image_paths = self.image_paths[:self.limit] if self.limit else self.image_paths              
        for image_path in image_paths:
            yield OpenAIImageTask(image_path=image_path)     

    def run(self):
        results = []
        for json_input in self.input():
            json_input_path = Path(json_input.path)
            with json_input_path.open() as f:

                json_data = json.load(f)

                print(json_data)

                content = json_data['choices'][0]['message']['content']
                content = content.replace('json\n', '').replace('```', '')

                data = json.loads(content)
                data['Image ID'] = json_input_path.stem

                data = {camelcase(k.replace('_' ,' ')):v for k,v in data.items()}
                results.append(data)

        df = pd.DataFrame(results) 
        df.to_csv(self.output().path, index=False) 




    def output(self):
        input_path_stems = [Path(i.path).stem for i in self.input()]
        if len(input_path_stems) == 1:
            output_suffix = input_path_stems[0]
        else:
            output_suffix = hash(''.join(input_path_stems)) 
        return luigi.LocalTarget(INTERMEDIATE_DATA_DIR / f'openai-{output_suffix}.csv')  
 



if __name__ == "__main__":   
    image_paths = [p for p in (RAW_DATA_DIR / 'drawer_1').glob('*.png')]
    image_paths.sort()

    verbatim_ids = pd.read_excel(RAW_DATA_DIR / f'd1_manual_metrics.xlsx')['Drawer/ID'].to_list()
    
    image_paths = [str(p) for p in image_paths if p.stem in verbatim_ids]

    # luigi.build([OpenAITask(image_paths=image_paths)], local_scheduler=True)                  

    # image_path = RAW_DATA_DIR / 'drawer_1/001-0042.png'
    # image_path2 = RAW_DATA_DIR / 'drawer_1/001-0135.png'
    # # luigi.build([OpenAIImageTask(image_path=image_path, force=True)], local_scheduler=True)


    images = ['001-0042.png', '001-0096.png']
    image_paths = [str(RAW_DATA_DIR / 'drawer_1' / i) for i in images]

    luigi.build([OpenAITask(image_paths=image_paths, force=True)], local_scheduler=True)                  