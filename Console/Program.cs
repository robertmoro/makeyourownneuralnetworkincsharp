using NeuralNetworkDomain;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetworkConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork neuralNetwork = new NeuralNetwork(784, 100, 10, 0.1f);

            var trainingData = File.ReadLines("mnist_train.csv");

            var data = new List<(Matrix Input, Matrix Output)>();

            foreach (var tdata in trainingData)
            {
                data.Add((ExtensionMethods.Transform(ExtensionMethods.ParseData(tdata.Substring(2))).ToMatrix(), ExtensionMethods.CreateTargetOutput(tdata[0].GetNumericValue()).ToMatrix()));
            }

            // Train the neural network
            var epochs = 5;

            foreach (var epoch in Enumerable.Range(0, epochs))
            {
                //foreach (var digit in trainingData)
                foreach(var d in data)
                {
                    //neuralNetwork.Train(Transform(ParseData(digit.Substring(2))), CreateTargetOutput(digit[0].GetNumericValue()));
                    neuralNetwork.Train(d.Input, d.Output);
                }
            }

            int correct = 0;
            int incorrect= 0;

            // Query neural network
            var testData = File.ReadLines("mnist_test.csv");
            foreach (var handWrittenDigit in testData)
            {
                // Known value
                var expectedValue = handWrittenDigit[0].GetNumericValue();
                //Console.WriteLine($"Value of handwritten digit: {expectedValue}");

                // Top three read by neutal network
                List<float> results = neuralNetwork.Query(ExtensionMethods.Transform(ExtensionMethods.ParseData(handWrittenDigit.Substring(2))).ToMatrix());
                
                int i = 0;
                var dic = results.ToDictionary(r => i++, r => r);
                int result = dic.OrderByDescending(r => r.Value).First().Key;

                if(expectedValue == result)
                {
                    correct++;
                }
                else
                {
                    incorrect++;
                }
                /*
                Console.Write($"Value read by neural network: ");
                Console.WriteLine(string.Join(", ", dic.OrderByDescending(r => r.Value).Take(3).Select(d => string.Format($"{d.Key} ({d.Value*100f:0.#}%)" ))));
                */
            }

            Console.WriteLine($"Total correct: {correct}");
            Console.WriteLine($"Total wrong: {incorrect}");
            Console.WriteLine($"Percentage correct: {((float)correct / (incorrect + correct)) * 100f}");

            Console.ReadKey();
        }
    }
}
