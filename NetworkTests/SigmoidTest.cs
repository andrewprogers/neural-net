using System;
using System.Collections.Generic;
using neuralNet;
using Xunit;
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;

namespace NetworkTests
{
    public class SigmoidTest
    {
        [Theory]
        [InlineData(new double[] {0, 1, 2}, new double[] {0.5, 0.73105, 0.8807})]
        public void CalculatesSigmoid(double[] values, double[] expected)
        {
            var vec = Vec.Build.DenseOfArray(values);
            var funcSigmoid = new Sigmoid().GetActivation();
            var result = funcSigmoid(vec);

            var epsilon = 0.0001;
            for (int i = 0; i < result.Count; i++)
            {
                Assert.InRange<double>(result[i], expected[i] - epsilon, expected[i] + epsilon);
            }
            
        }

        [Fact]
        public void CalculatesSigmoidPrime()
        {
            var vec = Vec.Build.DenseOfArray(new double[] {0, 1, 2, -5.4});
            var expected = new double[] {0.25, 0.1966, 0.1049, 0.0044};
            var SigmoidPrime = new Sigmoid().GetActivationPrime();
            var result = SigmoidPrime(vec);

            var epsilon = 0.0001;
            for (int i = 0; i < result.Count; i++)
            {
                Assert.InRange<double>(result[i], expected[i] - epsilon, expected[i] + epsilon);
            }        
        }
    }
}
