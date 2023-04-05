using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using Python.Deployment;
using Python.Runtime;

namespace GP_import_and_prep_models
{
    internal class Program
    {
        static string model_path = "D:/ComputerScience/4th_Year/GradutionProject/model-3-epochs/checkpoints-base.pth";

        static async Task Main(string[] args)
        {


            Installer.LogMessage += Console.WriteLine;
            await Installer.SetupPython();
            Python.Runtime.Runtime.PythonDLL = "python37.dll";
            var z = Installer.EmbeddedPythonHome;
            var x = await Installer.TryInstallPip();
            //await  Installer.PipInstallModule("tensorflow",force:true);
            Console.WriteLine("SAd");
            //await Installer.PipInstallModule("--upgrade pip");
            await Installer.PipInstallModule("transformers");
            PythonEngine.Initialize();
            Console.ReadLine();
            using (Py.GIL())
            {
                // create a Python scope
                using (var scope = Py.CreateScope())
                {
                    scope.Set("model_path", model_path);
                    var passage = "As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut.";
                    var answer = "September 1876";
                    scope.Set("passage", passage);
                    scope.Set("answer", answer);

                    scope.Exec(@"
import torch
import transformers
import spacy
from pathlib import Path
import numpy as np

nlp = spacy.load('en_core_web_sm')

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')


model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8).to(device)
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

class FFN(torch.nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, num_classes).to(device)

    def forward(self, x):
        out = self.fc1(x)
        return out

entity_type_dict = {'PERSON': [1, 0, 0, 0, 0], 'CARDINAL': [0, 1, 0, 0, 0], 'DATE': [0, 0, 1, 0, 0], 'ORG': [0, 0, 0, 1, 0], 'GPE': [0, 0, 0, 0, 1], 'None': [0, 0, 0, 0, 0]}
mapping_vector = { 0 : 'who', 1 : 'what', 2 : 'when', 3 : 'where', 4 : 'why', 5 : 'how',6 : 'which',7 : 'other'}

ffn = FFN(hidden_size=773, num_classes=8)
softmax=torch.nn.Softmax(dim=-1)
model_path = Path(r'/content/drive/MyDrive/Datasets/model-3-epochs/checkpoints-base.pth')


def encode_passage_answer(passage,answer):
  encoded_dict = tokenizer.encode_plus(
                          passage,
                          answer,
                          add_special_tokens = True,
                          max_length = 512,
                          pad_to_max_length = True,
                          return_attention_mask = True,
                          return_tensors = 'pt'
                    )
  return encoded_dict['input_ids'] , encoded_dict['attention_mask']
  
def extract_ner_answer(answer):
  entity_type = ""
  doc = nlp(answer)
  if doc.ents:
      entity_types = doc.ents[0].label_
  else:
      entity_types = 'None'
  entity_type_embeddings_train = np.zeros((1, 5))
  try:
      entity_type_embeddings_train = np.array(entity_type_dict[entity_type])
  except:  
      entity_type_embeddings_train = np.array(entity_type_dict['None'])

  return entity_type_embeddings_train

def load_model(model_path,model,ffn):
  checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint['model_state_dict'])
  ffn.load_state_dict(checkpoint['ffn_state_dict'])
  return model , ffn

model , ffn = load_model(model_path , model , ffn)
input_ids , attention_masks = encode_passage_answer(passage , answer)
entity_type_embeddings = extract_ner_answer(answer)
entity_type_embeddings = entity_type_embeddings.reshape(1,-1)

# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
entity_type_embeddings = torch.from_numpy(entity_type_embeddings)

def evaluation(model , ffn , input_ids , attention_masks , entity_type_embeddings):
  model.eval()
  outputs = model(input_ids, attention_mask = attention_masks,output_hidden_states=True)
  cls_tensor = outputs.hidden_states[-1]
  cls_tensor = cls_tensor[:,0,:]
  ner_tensor = entity_type_embeddings.to(torch.float32)
  concat_tensor = torch.cat((cls_tensor, ner_tensor),dim=1).to(device)
  logits = ffn(concat_tensor)  
  prediction = softmax(logits)
  prediction = int(torch.argmax(prediction, dim=1))

  return mapping_vector[prediction]

prediction = evaluation(model , ffn ,input_ids , attention_masks , entity_type_embeddings)


");

                    dynamic zzz = scope.Get("prediction");
                    Console.WriteLine($"{zzz}");
                }
            }

        }

    }
}
