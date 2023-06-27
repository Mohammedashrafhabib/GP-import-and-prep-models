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
        static string model_path = "C:\\Users\\mando\\Downloads\\onnx(answerExtraction)-20230330T174005Z-001\\checkpoints-base.pth";

        static async Task Main(string[] args)
        {


            Installer.LogMessage += Console.WriteLine;
            await Installer.SetupPython();
            Python.Runtime.Runtime.PythonDLL = "python37.dll";
            var z = Installer.EmbeddedPythonHome;
            var x = await Installer.TryInstallPip();
            //await Installer.PipInstallModule("tensorflow",force:true);
            //await Installer.PipInstallModule("transformers", force: true);
            //await Installer.PipInstallModule("torch", force: true);
            //await Installer.PipInstallModule("spacy", force: true);
           // await Installer.PipInstallModule("urllib3", force: true,version: "1.26.16");


            Console.WriteLine("SAd");
            //await Installer.PipInstallModule("--upgrade pip");
            //await Installer.PipInstallModule(@"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz");
            //Installer.RunCommand("python -m spacy download en_core_web_sm");
            PythonEngine.Initialize();
           // Console.ReadLine();
            using (Py.GIL())
            {
                // create a Python scope
                using (var scope = Py.CreateScope())
                {
                    scope.Set("model_path", model_path);
                    var passage = "As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut.";
                    var answer = @"September 1876";
                    scope.Set("passage", passage);
                    scope.Set("answer", answer);

                    scope.Exec(@"
import transformers
import torch
import spacy
nlp = spacy.load('en_core_web_sm')

'''# Enable Cuda Device'''

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')

'''# FFN'''

class FFN(torch.nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, num_classes).to(device)

    def forward(self, x):
        out = self.fc1(x)
        return out

'''#Loading Saved Model'''

model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8).to(device)


checkpoint = torch.load(model_path,map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])

ffn = FFN(hidden_size=773, num_classes=8)
ffn.load_state_dict(checkpoint['ffn_state_dict'])

'''# Passage Answer Encoding'''

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

encoded_input  = tokenizer.encode_plus(
                      passage,
                      answer,
                      max_length = 512,
                      truncation=True,
                      truncation_strategy='only_first',
                      padding='max_length',
                      return_attention_mask = True,
                      return_tensors = 'pt'
                    )
input_ids ,attention_mask = encoded_input['input_ids'] , encoded_input['attention_mask']

'''# Named Entity For Answer'''

entity_type_dict = {'PERSON': [1, 0, 0, 0, 0], 'CARDINAL': [0, 1, 0, 0, 0], 'DATE': [0, 0, 1, 0, 0], 'ORG': [0, 0, 0, 1, 0], 'GPE': [0, 0, 0, 0, 1], 'None': [0, 0, 0, 0, 0]}
entity_type_embeddings = list()
doc = nlp(answer)
if doc.ents:
  entity_types = set([token.ent_type_ if token.ent_type_ != '' else 'None' for token in doc])
else:
  entity_types = set(['None'])
entities = set()
entity_list = list()
for ent in entity_types:
  if ent in entity_type_dict:
    entities.add(ent)
  else :
    entities.add('None')

for ent in entities:
  try:
    entity = (entity_type_dict[ent])
  except:
    entity = (entity_type_dict['None'])
  entity_list.append(entity)

for i in range(6- len(entity_list)):
  entity_list.append(entity_type_dict['None'])

entity_type_embeddings.append(entity_list)

'''# Evaluation'''

# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
entity_type_embeddings = torch.tensor(entity_type_embeddings)

mapping_vector = { 0 : 'who', 1 : 'what', 2 : 'when', 3 : 'where', 4 : 'why', 5 : 'how',6 : 'which',7 : 'other'}

model.eval()

with torch.no_grad():
  outputs = model(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)

cls_tensor = outputs.hidden_states[-1]
cls_tensor = cls_tensor[:,:6,:]
ner_tensor = entity_type_embeddings

# Concatenate the tensors along the first dimension
concat_tensor = torch.cat((cls_tensor, ner_tensor),dim=2).to(device)

logits = ffn(concat_tensor)
logits=(torch.max(logits,dim=1).values)

prediction = torch.argmax(logits, dim=1)
prediction = mapping_vector[int(prediction)]

");

                    dynamic zzz = scope.Get("prediction");
                    Console.WriteLine($"{zzz}");
                }
            }
            Console.ReadLine();

        }

    }
}
