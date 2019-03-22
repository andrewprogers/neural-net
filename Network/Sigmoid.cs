using System;
using System.Collections.Generic;
using System.IO;
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;

namespace neuralNet
{
    public interface Activation
    {
        Func<Vec, Vec> GetActivation();
        Func<Vec, Vec> GetActivationPrime();
    }

    public class Sigmoid : Activation
    {
        public Func<Vec, Vec> GetActivation() {
            return SigmoidFunction;
        }

        public Func<Vec, Vec> GetActivationPrime() {
            return SigmoidPrime;
        }

        private static Vec SigmoidFunction(Vec ZValues)
        {
            return 1.0 / (1.0 + Vec.Exp(-ZValues));
        }

        private static Vec SigmoidPrime(Vec ZValues)
        {
            var sigmoid = SigmoidFunction(ZValues);
            var ones = Vec.Build.Dense(sigmoid.Count, 1);
            return sigmoid.PointwiseMultiply(ones - sigmoid);
        }
    }
}