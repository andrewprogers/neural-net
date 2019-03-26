using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<System.Double>;

namespace NeuralNet
{
    public class Layer
    {
        public Vec Biases { get; set; } = null;
        public int NeuronCount { get; private set; }
        public Activator Activator { get; private set; }

        public Layer NextLayer { get; set; }
        public Layer PreviousLayer { get; set; }

        public Vec Activations { get; set; }
        public Vec ZValues { get; private set; }

        private Matrix _weights = null;
        private Vec _errors = null;

        public Layer(int size, Layer next = null, Activator activator = null)
        {
            NeuronCount = size;
            Activations = Vec.Build.Dense(size, 0);
            Activator = activator;

            if (next != null)
            {
                NextLayer = next;
                next.PreviousLayer = this;
                next.Weights = Matrix.Build.Random(next.NeuronCount, this.NeuronCount);
                next.Biases = Vec.Build.Random(next.NeuronCount);
            }
        }

        public Vec Activate(Vec inputActivations)
        {
            _errors = null;
            ZValues = Weights.Multiply(inputActivations) + Biases;
            if (Activator == null)
            {
                Activations =  ZValues;
            } else {
                Activations = Activator.Activate(ZValues);
            }
            return Activations;
        }

        public Vec Activate()
        {
            if (PreviousLayer == null)
            {
                throw new ArgumentException("Cannot activate on input node");
            } else {
                return Activate(PreviousLayer.Activations);
            }
        }

        public Vec Backpropagate()
        {
            if (NextLayer == null)
            {
                throw new NotSupportedException("Cannot backpropagate errors on output layer");
            }
            if (PreviousLayer == null)
            {
                throw new NotSupportedException("Cannot backpropagate errors on input layer");
            }
            if (NextLayer.Errors == null)
            {
                throw new InvalidOperationException("Cannot backpropagate when next layer's errors are null");
            }
            if (ZValues == null)
            {
                throw new InvalidOperationException("Cannot backpropagate before forward propagation");
            }
            var primeOfZs = Activator.ActivatePrime(ZValues);
            _errors = NextLayer.Weights.TransposeThisAndMultiply(NextLayer.Errors).PointwiseMultiply(primeOfZs);
            return _errors;
        }

        public void Print()
        {
            if (PreviousLayer != null)
            {
                Console.WriteLine(Weights.Append(Biases.ToColumnMatrix()));
            } else {
                Console.WriteLine($"Input layer of {NeuronCount} neurons");
            }
        }

        // Property Implementations

        public Matrix Weights { 
            get { return _weights; } 
            set {
                ValidateWeights(value);
                _weights = value;
            }
        }

        public Vec Errors {
            get { return _errors; }
            set {
                if (NextLayer != null)
                {
                    throw new InvalidOperationException("Cannot set errors for non-output layer");
                }
                _errors = value;
            }
        }

        public Vec DCostDBiases {
            get {
                if (_errors == null)
                {
                    throw new InvalidOperationException("Cannot get Cost derivatives before backpropagation");
                }
                return _errors;
            }

            protected set { }
        }

        public Matrix DCostDWeights {
            get {
                if (_errors == null)
                {
                    throw new InvalidOperationException("Cannot get Cost derivatives before backpropagation");
                }
                var errorsCol = _errors.ToColumnMatrix();
                var activationsInRow = PreviousLayer.Activations.ToRowMatrix();
                return errorsCol.Multiply(activationsInRow);
            }
        }

        // Private Methods
        
        private void ValidateWeights(Matrix weights)
        {
            if (weights.RowCount != NeuronCount)
            {
                throw new ArgumentException($"Cannot assign a Weight matrix with {weights.RowCount} rows to a Layer of {NeuronCount} neurons.");
            } else if (weights.ColumnCount != PreviousLayer.NeuronCount) {
                var m = $"Cannot assign a Weight matrix with {weights.ColumnCount} columns to a Layer with an input of {PreviousLayer.NeuronCount} neurons.";
                throw new ArgumentException(m);
            }
        }
    }
}