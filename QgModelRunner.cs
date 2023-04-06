using Python.Runtime;
using System;
using System.Runtime.Remoting;
using System.Threading.Tasks;

namespace GP_import_and_prep_models
{
    internal class QgModelRunner
    {
        static string MODEL_PATH = @"C:\Users\Mostafa\Downloads\QG-With-t5-small-epochs=3";
        static string Token_Path = @"C:\Users\Mostafa\Downloads\t5-small-tokenizer";
        public static async Task<string[]> Run(String passage, String[] answers, String[] question_words)
        {
            var watch = new System.Diagnostics.Stopwatch();
            watch.Start();
            //Installer.LogMessage += Console.WriteLine;
            //await Installer.SetupPython();
            Python.Runtime.Runtime.PythonDLL = @"C:\Users\Mostafa\AppData\Local\python-3.7.3-embed-amd64\python37.dll";
            //var z = Installer.EmbeddedPythonHome;
            //var x = await Installer.TryInstallPip();
            //await  Installer.PipInstallModule("tensorflow",force:true);
            //Console.WriteLine("SAd");
            //await Installer.PipInstallModule("--upgrade pip");
            //await Installer.PipInstallModule("transformers", force: true);
            PythonEngine.Initialize();
            //Console.ReadLine();
            string[] questions;
            using (Py.GIL())
            {
                // create a Python scope
                using (var scope = Py.CreateScope())
                {
                    scope.Set("MODEL_PATH", MODEL_PATH);
                    scope.Set("Token_Path", Token_Path);
                    //var passage = "Software quality assurance(SQA) is the ongoing process that ensures the software product meets and complies with the organization's established and standardized quality specifications its published in 2001";
                    //var answer = "2001";
                    //var question_word = "what";
                    scope.Set("passage", passage);
                    scope.Set("answers", answers);
                    scope.Set("question_words", question_words);
                    scope.Exec(@"
import tensorflow
import transformers
from transformers import AutoTokenizer, TFT5ForConditionalGeneration

task_prefix = 'generate question using question word: '
encoder_max_len = 250
decoder_max_len = 70
model=TFT5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(Token_Path, local_files_only=True)

inputs = tokenizer([task_prefix + question_word + ' ' + 'answer:' + ' '+ answer + ' ' + 'context: ' + passage for answer, question_word in zip(answers, question_words)], return_tensors='tf', padding=True, truncation=True, max_length=encoder_max_len)

generated_questions = model.generate(inputs['input_ids'], attention_mask = inputs['attention_mask'], max_length = decoder_max_len, top_p = 0.95, top_k = 50, repetition_penalty = float(2))
decoded_questions = tokenizer.batch_decode(generated_questions, skip_special_tokens = True)
");
                    questions = scope.Get<string[]>("decoded_questions");
                    watch.Stop();
                    string elapsedMs = watch.ElapsedMilliseconds.ToString();
                    Console.WriteLine(elapsedMs);
                    Console.WriteLine($"{questions.Length}");
                }
            }
            PythonEngine.Shutdown();
            return questions;
        }
    }
}
