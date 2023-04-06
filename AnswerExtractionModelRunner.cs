using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GP_import_and_prep_models
{

    internal class AnswerExtractionModelRunner
    {
        static string MODEL_PATH = @"C:\Users\Mostafa\Downloads\t5-small-epochs=4";
        static string Token_Path = @"C:\Users\Mostafa\Downloads\t5-small-tokenizer";
        public static async Task<String> Run(String context)
        {
            var watch = new System.Diagnostics.Stopwatch();
            watch.Start();
            //Installer.LogMessage += Console.WriteLine;
            //await Installer.SetupPython();
            Python.Runtime.Runtime.PythonDLL = @"C:\Users\Mostafa\AppData\Local\python-3.7.3-embed-amd64\python37.dll";
            //var z = Installer.EmbeddedPythonHome;
            //var x = await Installer.TryInstallPip();
            // await  Installer.PipInstallModule("tensorflow",force:true);
            //Console.WriteLine("SAd");
            // await Installer.PipInstallModule("--upgrade pip");
            //await Installer.PipInstallModule("transformers", force: true);
            PythonEngine.Initialize();
            //Console.ReadLine();
            String answers;
            using (Py.GIL())
            {
                // create a Python scope
                using (var scope = Py.CreateScope())
                {
                    scope.Set("MODEL_PATH", MODEL_PATH);
                    scope.Set("Token_Path", Token_Path);
                    //var context = "The Boys are led by Billy Butcher, who despises all superheroes, and the Seven are led by the unstable and violent superheroes. At the start of the series, the Boys are joined by Hughie Campbell after the superhero A-Train accidentally kills girlfriend while high on drugs. Elsewhere, the Seven are joined by Annie January, a young and hopeful heroine forced to face the truth about those she admires. Other members of the Seven include the disillusioned Queen Maeve, the insecure Deep, the mysterious Black Noir, and the white supremacist Stormfront. The Boys are rounded out by tactical planner Mother's \"MM\" Milk, weapons specialist Frenchie, and super-powered test subject Kimiko. Overseeing the Seven is Vought executive Madelyn Stillwell, who is later succeeded by publicist Ashley Barrett, themselves overseen by Vought CEO Stan Edgar, who also maintains controlled governmental opposition via his adoptive daughter Congresswoman Victoria Neuman.so with all what's going on what is it that's going to happen is it enough or will murder and mayhem find it way?";
                    scope.Set("context", context);
                    scope.Exec(@"
import tensorflow
import transformers
from transformers import AutoTokenizer, TFT5ForConditionalGeneration

task_prefix = 'extract answers: '
encoder_max_len = 250
decoder_max_len = 70

model=TFT5ForConditionalGeneration.from_pretrained(MODEL_PATH,local_files_only=True)
model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(Token_Path,local_files_only=True)
input_text = task_prefix + 'context: ' + context

encoded_query = tokenizer(input_text, return_tensors='tf', pad_to_max_length=True, truncation=True, max_length=encoder_max_len)
input_ids = encoded_query['input_ids']
attention_mask = encoded_query['attention_mask']

generated_answers = model.generate(input_ids, attention_mask=attention_mask, max_length=decoder_max_len, top_p=0.95, top_k=50, repetition_penalty=float(2))
decoded_answers = tokenizer.decode(generated_answers.numpy()[0], skip_special_tokens=True)
");
                    answers = scope.Get("decoded_answers").ToString();
                    watch.Stop();
                    string elapsedMs = watch.ElapsedMilliseconds.ToString();
                    Console.WriteLine(elapsedMs);
                    Console.WriteLine($"{answers}");
                }
            }
            PythonEngine.Shutdown();
            return answers;
        }
    }
}
