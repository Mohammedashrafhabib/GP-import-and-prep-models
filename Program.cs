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
            //await  Installer.PipInstallModule("sense2vec",force:true, version: "2.0.2");
            //await Installer.PipInstallModule("urllib3", force: true, version: "1.26.16");
            //await  Installer.PipInstallModule("word2number");
            //Console.WriteLine("SAd");
            //await Installer.PipInstallModule("--upgrade pip");
            //await Installer.PipInstallModule(@"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz");
           // await Installer.PipInstallModule(@"https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0.tar.gz");
            // Installer.RunCommand("python -m spacy download en_core_web_sm");
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
                    scope.Set("answer", answer);
                    string S2V = "C:\\Users\\mando\\Downloads\\distractors\\s2v_old";
                    scope.Set("S2VPath", S2V);

                    scope.Exec(@"

#####Distractors

from sense2vec import Sense2Vec
import spacy
import re
from word2number import w2n
from collections import OrderedDict

#spacy.cli.download('en_core_web_lg')
nlp = spacy.load('en_core_web_lg')
s2v = nlp.add_pipe('sense2vec')
s2v.from_disk(S2VPath)
def common(s0, s1):
  s0 = s0.lower()
  s1 = s1.lower()
  s0List = s0.split(' ')
  s1List = s1.split(' ')
  return len(list(set(s0List)&set(s1List)))
def sense2vec_get_words(context):
  doc=nlp(context)
  output2 = []
  for ent in doc.ents:
    try:
      output = []

      # print(ent.text, ent.end_char, ent.label_)
      most_similar=ent._.s2v_most_similar(100)
      for each_word in most_similar:
        if each_word[0][1]==ent.label_:
          append_word = each_word[0][0]

          new_append_word = re.sub(r'[^\w]', ' ', append_word.lower())
          new_word = re.sub(r'[^\w]', ' ', ent.text.lower())
          new_word2=''
          if ent.label_=='CARDINAL':
            new_word2=str(w2n.word_to_num(new_word))
          if new_append_word not in new_word and new_word not in new_append_word and common(new_word,new_append_word)==0 and common(new_word2,new_append_word)==0 :
              output.append(append_word.title())
      #print( list(OrderedDict.fromkeys(output)))
      output2.append( [ent.text,list(OrderedDict.fromkeys(output))])
      #print(output2)
    except:
      continue
  return output2
");
                    scope.Set("context", passage);
                    scope.Exec(@"
distractors = sense2vec_get_words(context)
len=len(distractors)
");
                    var distractors = scope.Get<PyList> ("distractors");
                    var len = scope.Get<int>("len");

                    List<KeyValuePair<string, string[]>> ret = new List<KeyValuePair<string, string[]>>();
                    for (int i = -1; i < len; i++)
                    {
                        Console.WriteLine($"{distractors[i][0].ToString()}");
                        Console.WriteLine($"{distractors[i][1].As<string[]>()}");

                        ret.Add(new KeyValuePair<string, string[]>(distractors[i][0].ToString(), distractors[i][1].As<string[]>()));
                    }
                    //                    dynamic zzz = scope.Get("prediction");
                    Console.WriteLine($"{distractors}");
                }
            }
            Console.ReadLine();

        }

    }
}
