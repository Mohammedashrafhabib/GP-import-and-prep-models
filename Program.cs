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
            await  Installer.PipInstallModule("sense2vec",force:true);
            Console.WriteLine("SAd");
            //await Installer.PipInstallModule("--upgrade pip");
            //await Installer.PipInstallModule(@"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz");
           // Installer.RunCommand("python -m spacy download en_core_web_sm");
            PythonEngine.Initialize();
            //Console.ReadLine();
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
                    string S2V = "C:\\Users\\mando\\Downloads\\distractors\\s2v_old";
                    scope.Set("S2VPath", S2V);

//                    scope.Exec(@"

//#####Distractors

//from sense2vec import Sense2Vec
//print('sad')

//s2v = Sense2Vec().from_disk(S2VPath)
//print('sad')
//def sense2vec_get_words(word,s2v):
//    output = []
//    word = word.lower()
//    word = word.replace('' '', '_')
//    sense = s2v.get_best_sense(word)
//    if sense is None:
//      return list()
//    most_similar = s2v.most_similar(sense, n=4)


//    for each_word in most_similar:
//      # if each_word[1]<=0.6:
//        append_word = each_word[0].split(''|'')[0].replace('_', ' ').lower()
//        if append_word.lower() != word:
//            output.append(append_word.title())

//    out = list(output)
//    return out
//print('sad')


//");

//                    dynamic zzz = scope.Get("prediction");
//                    Console.WriteLine($"{zzz}");
                }
            }
            Console.ReadLine();

        }

    }
}
