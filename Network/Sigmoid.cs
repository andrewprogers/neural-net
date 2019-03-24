using System;
using System.Collections.Generic;
using System.IO;
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;

namespace NeuralNet
{
    public interface Activator
    {
        Vec Activate(Vec vector);
        Vec ActivatePrime(Vec vector);
    }

    public class Sigmoid : Activator
    {
        public Vec Activate(Vec ZValues)
        {
            return 1.0 / (1.0 + Vec.Exp(-ZValues));
        }

        public  Vec ActivatePrime(Vec ZValues)
        {
            var sigmoid = Activate(ZValues);
            var ones = Vec.Build.Dense(sigmoid.Count, 1);
            return sigmoid.PointwiseMultiply(ones - sigmoid);
        }
    }
}