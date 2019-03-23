using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworkUsingMathLibrary
{
    public class NeuralNetwork
    {
        private readonly float _learningRate;
        private readonly Random _random = new Random();
        private Matrix<float> _linkWeightsInputHidden;
        private Matrix<float> _linkWeightsHiddenOutput;

        public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, float learningRate)
        {
            _learningRate = learningRate;
            MathNet.Numerics.Distributions.IContinuousDistribution distribution = new MathNet.Numerics.Distributions.Normal(0d, Math.Pow(inputNodes, -0.5));
            _linkWeightsInputHidden = Matrix<float>.Build.Random(hiddenNodes, inputNodes, distribution);
            _linkWeightsHiddenOutput = Matrix<float>.Build.Random(outputNodes, hiddenNodes, distribution);
        }

        public void Train(Matrix<float> inputs, Matrix<float> targets)
        {
            // Calculate signals into hidden layer
            var hidden_inputs = _linkWeightsInputHidden.Multiply(inputs);
            // Calculate the signals emerging from hidden layer
            var hidden_outputs = hidden_inputs.ActivationFunction();

            // Calculate signals into final output layer
            var final_inputs = _linkWeightsHiddenOutput.Multiply(hidden_outputs);
            // Calculate the signals emerging from final output layer
            var final_outputs = final_inputs.ActivationFunction();

            // Output layer error is the (target - actual)
            var output_errors = targets - final_outputs;
            // Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
            var hidden_errors = _linkWeightsHiddenOutput.Transpose().Multiply(output_errors);

            // Update the weights for the links between the hidden and output layers
            _linkWeightsHiddenOutput = _linkWeightsHiddenOutput + _learningRate * (output_errors.Apply(final_outputs, (oe, fo) => { return oe * fo * (1.0f - fo); }).Multiply(hidden_outputs.Transpose()));
            // Update the weights for the links between the input and hidden layers
            _linkWeightsInputHidden = _linkWeightsInputHidden + _learningRate * (hidden_errors.Apply(hidden_outputs, (he, ho) => { return he * ho * (1.0f - ho); }).Multiply(inputs.Transpose()));
        }

        /// <summary>
        /// Query the neural network
        /// </summary>
        public float[] Query(Matrix<float> input)
        {
            // Calculate signals into hidden layer
            var hidden_inputs = _linkWeightsInputHidden.Multiply(input);

            // Calculate the signals emerging from hidden layer
            var hidden_outputs = hidden_inputs.ActivationFunction();

            // Calculate signals into final output layer
            var final_inputs = _linkWeightsHiddenOutput.Multiply(hidden_outputs);

            // Calculate the signals emerging from final output layer
            var final_outputs = final_inputs.ActivationFunction();

            return final_outputs.ToRowMajorArray();
        }
    }
}
