using System;
using System.Collections.Generic;
using System.Runtime.Remoting;
using System.Threading.Tasks;
using Python.Runtime;

namespace GP_import_and_prep_models
{
    struct TmpStruct
    {
        string answer;
        string question_word;
        string question;

        public TmpStruct(string answer, string question_word, string question)
        {
            this.answer = answer;
            this.question_word = question_word;
            this.question = question;
        }

        public override string ToString()
        {
            return $"{{answer: {answer},\nquestion_word: {question_word},\nquestion: {question}}}";
        }
    }

    internal class Program
    {
        static async Task Main(string[] args)
        {
            String context = "Software quality assurance(SQA) is the ongoing process that ensures the software product meets and complies with the organization's established and standardized quality specifications its published in 2001";
            String answerString = await AnswerExtractionModelRunner.Run(context);
            string[] answersArray = answerString.Split(',');
            //TODO: This should be replaced with Interrogative word classifier runner code
            string[] question_words = new string[answersArray.Length];
            for (int i = 0; i < answersArray.Length; i++)
            {
                question_words[i] = "what";
            }
            List<TmpStruct> tmp = new List<TmpStruct>();
            string[] questions = await QgModelRunner.Run(context, answersArray, question_words);
            for (int i = 0; i < answersArray.Length; i++)
            {
                tmp.Add(new TmpStruct(answersArray[i], question_words[i], questions[i]));
            }
            Console.WriteLine("*************************************");
            Console.WriteLine("Output Summary:");
            Console.WriteLine($"context: {context}");
            foreach (TmpStruct example in tmp)
            {
                Console.WriteLine(example.ToString());
                Console.WriteLine("-----------------");
            }
        }
    }
}