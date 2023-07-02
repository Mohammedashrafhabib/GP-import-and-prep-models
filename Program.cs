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


        static async Task Main(string[] args)
        {


            Installer.LogMessage += Console.WriteLine;
            await Installer.SetupPython();
            Python.Runtime.Runtime.PythonDLL = "python37.dll";
            var z = Installer.EmbeddedPythonHome;
            var x = await Installer.TryInstallPip();
         
           // await Installer.PipInstallModule("nltk", force: true);
        
          
            PythonEngine.Initialize();
           // Console.ReadLine();
            using (Py.GIL())
            {
                // create a Python scope
                using (var scope = Py.CreateScope())
                {
                    var passage = "As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut.";
       
                    scope.Exec(@"
import nltk
nltk.download('wordnet')
nltk.download('punkt')

def split_passage_to_sentences(passage):
    sentences = nltk.sent_tokenize(passage)

    return sentences

");
                    scope.Set("passage", passage);
                    scope.Exec(@"
sentences = split_passage_to_sentences(passage)

");
                    var distractors = scope.Get<string[]> ("sentences");

                    //                    dynamic zzz = scope.Get("prediction");
                    Console.WriteLine($"{distractors}");
                }
            }
            Console.ReadLine();

        }

    }
}
