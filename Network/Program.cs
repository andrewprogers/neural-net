using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Control.UseNativeMKL();
                var trainingExamples = DataLoader.LoadFromFiles("./data/train-labels-idx1-ubyte", "./data/train-images-idx3-ubyte");
                var testExamples = DataLoader.LoadFromFiles("./data/t10k-labels-idx1-ubyte", "./data/t10k-images-idx3-ubyte");
                var activation = new Sigmoid();
                var init = new ScaledInitializer();
                var net = new Network(new List<int>() { 784, 30, 10 }, activation, init);

                net.SGD(trainingExamples, 30, 10, 3.0, testExamples);
            }
            catch (System.Exception ex)
            {
                System.Console.WriteLine(ex.ToString());
            }


        }
    }
}
