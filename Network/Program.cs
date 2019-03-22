using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace neuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                var trainingExamples = DataLoader.LoadFromFiles("./data/train-labels-idx1-ubyte", "./data/train-images-idx3-ubyte");
                var testExamples = DataLoader.LoadFromFiles("./data/t10k-labels-idx1-ubyte", "./data/t10k-images-idx3-ubyte");
                var activation = new Sigmoid();
                var net = new Network(new List<int>() { 784, 30, 10 }, activation);

                trainingExamples = trainingExamples.GetRange(0, 10000);
                net.SGD(trainingExamples, 30, 10, 3.0, testExamples);
            }
            catch (System.Exception ex)
            {
                System.Console.WriteLine(ex.ToString());
            }


        }
    }
}
