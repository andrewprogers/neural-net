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
                //Control.UseNativeMKL();   // Uncomment if MKL installed locally
                var trainingExamples = DataLoader.LoadFromFiles("./data/train-labels-idx1-ubyte", "./data/train-images-idx3-ubyte");
                var testExamples = DataLoader.LoadFromFiles("./data/t10k-labels-idx1-ubyte", "./data/t10k-images-idx3-ubyte");
                var activation = new Sigmoid();
                var init = new ScaledInitializer();
                var net = new Network(new List<int>() { 784, 30, 10 }, activation, init);

                // These properties allow the definition of callback functions for training events
                net.OnEpochComplete = delegate (int epoch, TimeSpan duration)
                {
                    Console.WriteLine($"Finished epoch number {epoch} in {duration.TotalSeconds} seconds");
                };

                net.OnEpochTestComplete = delegate (int epoch, int testCount, int successCount)
                {
                    var percent = ((double)successCount / testCount).ToString("0.00%");
                    System.Console.WriteLine($"\tTest Results: {successCount}/{testCount} examples correct. ({percent})");
                };


                net.SGD(trainingExamples, 30, 10, 3.0, testExamples);
            }
            catch (System.Exception ex)
            {
                System.Console.WriteLine(ex.ToString());
            }
        }
    }
}
