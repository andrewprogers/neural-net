using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;

namespace NeuralNet
{
    public static class DataLoader
    {
        
        public static List<Example> LoadFromFiles(string labelFilePath, string imagesFilePath)
        {
            var labels = ReadLabels(labelFilePath);
            var images = ReadImages(imagesFilePath);
            var examples = labels.Zip(images, (l, i) => new Example(i, l)).ToList();
            return examples;
        }

        private static void PrintExample((Vector<double> label, Vector<double> image) example)
        {
            var map = Vector<Double>.Build.DenseOfArray(new double[] {0,1,2,3,4,5,6,7,8,9});
            System.Console.WriteLine($"Label: {example.label.DotProduct(map)}");

            for (int r = 0; r < 28; r++)
            {
                for (int c = 0; c < 28; c++)
                {
                    var pixel = example.image[r * 28 + c];
                    System.Console.Write(pixel > 50 ? '.' : ' ');
                }
                System.Console.Write('\n');
            }
        }

        private static List<Vector<Double>> ReadImages(string imagesFilePath)
        {
            using (var file = File.OpenRead(imagesFilePath))
            {
                file.Seek(4, SeekOrigin.Begin); // Skip magic number at beginning
                var numItems = file.Read32BitInteger();
                var rows = file.Read32BitInteger();
                var cols = file.Read32BitInteger();
                var numPixels = rows * cols;

                var images = new List<Vector<Double>>(numItems);
                for (int i = 0; i < numItems; i++)
                {
                    var pixels = new byte[numPixels];
                    file.Read(pixels, 0, numPixels);
                    var imageVector = Vector<Double>.Build.DenseOfArray(pixels.Select(b => (double)b).ToArray());
                    images.Add(imageVector);
                }
                return images;
            }
        }

        private static List<Vector<Double>> ReadLabels(string labelFilePath)
        {
            using (var file = File.OpenRead(labelFilePath))
            {
                file.Seek(4, SeekOrigin.Begin); // Skip magic number at beginning

                var numItems = file.Read32BitInteger();
                var labels = new List<Vector<Double>>(numItems);

                for (int i = 0; i < numItems; i++)
                {
                    var labelVector = Vector<Double>.Build.Dense(10,0);
                    var label = file.ReadByte();
                    labelVector[label] = 1;
                    labels.Add(labelVector);
                }
                return labels;
            }
        }
    }

    public class Example
    {
        public Vector<Double> Image { get; set; }
        public Vector<Double> Label { get; set; }

        public Example(Vector<Double> image, Vector<Double> label)
        {
            Image = image; Label = label;
        }
    }
}