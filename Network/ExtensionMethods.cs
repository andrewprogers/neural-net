using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNet
{
    public static class ExtensionMethods
    {
        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            var rng = new Random();

            while (n > 1)
            {
                n--;
                var swap = rng.Next(n + 1);
                var temp = list[swap];
                list[swap] = list[n];
                list[n] = temp;
            }
        }

        public static int Read32BitInteger(this Stream f)
        {
            var buffer = new byte[4];
            f.Read(buffer);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(buffer);
            }
            return BitConverter.ToInt32(buffer);
        }
    }
}
