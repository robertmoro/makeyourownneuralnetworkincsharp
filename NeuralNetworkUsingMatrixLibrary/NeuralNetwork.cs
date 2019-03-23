using System;
using System.Collections.Generic;

namespace NeuralNetworkDomain
{
    public class NeuralNetwork
    {
        private readonly float _learningRate;
        private readonly Random _random = new Random();

        public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, float learningRate)
        {
            _learningRate = learningRate;

            LinkWeightsInputHidden = new Matrix(hiddenNodes, inputNodes);
            LinkWeightsInputHidden.Init(() => (float)_random.NextGaussian(1.0, 4.0));
            //InitLinkWeights(_linkWeightsInputHidden, 1.0, 4.0);

            LinkWeightsHiddenOutput = new Matrix(outputNodes, hiddenNodes);
            LinkWeightsHiddenOutput.Init(() => (float)_random.NextGaussian(1.0, 4.0));
            //InitLinkWeights(_linkWeightsHiddenOutput, 1.0, 4.0);
        }

        public Matrix LinkWeightsInputHidden { get; set; }
        public Matrix LinkWeightsHiddenOutput { get; set; }

        //public void Train(Matrix inputs, Matrix targets)
        //{
        //    // Calculate signals into hidden layer
        //    var hidden_inputs = _linkWeightsInputHidden.Multiply(inputs);
        //    // Calculate the signals emerging from hidden layer
        //    var hidden_outputs = hidden_inputs.ActivationFunction();

        //    // Calculate signals into final output layer
        //    var final_inputs = _linkWeightsHiddenOutput.Multiply(hidden_outputs);
        //    // Calculate the signals emerging from final output layer
        //    var final_outputs = final_inputs.ActivationFunction();

        //    // Output layer error is the (target - actual)
        //    var output_errors = targets - final_outputs;
        //    // Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        //    var hidden_errors = _linkWeightsHiddenOutput.Transpose().Multiply(output_errors);

        //    // Update the weights for the links between the hidden and output layers
        //    _linkWeightsHiddenOutput = _linkWeightsHiddenOutput + _learningRate * (output_errors.Apply(final_outputs, (oe, fo) => { return oe * fo * (1.0f - fo); }).Multiply(hidden_outputs.Transpose()));
        //    // Update the weights for the links between the input and hidden layers
        //    _linkWeightsInputHidden = _linkWeightsInputHidden + _learningRate * (hidden_errors.Apply(hidden_outputs, (he, ho) => { return he * ho * (1.0f - ho); }).Multiply(inputs.Transpose()));
        //}

        public void Train(Matrix inputs, Matrix targets)
        {
            var result = Process(inputs, targets);

            // Update the weights for the links between the hidden and output layers
            LinkWeightsHiddenOutput = result.LinkWeightsHiddenOutput;
            // Update the weights for the links between the input and hidden layers
            LinkWeightsInputHidden = result.LinkWeightsInputHidden;
        }

        public (Matrix LinkWeightsHiddenOutput, Matrix LinkWeightsInputHidden) Process(Matrix inputs, Matrix targets)
        {
            // Calculate signals into hidden layer
            var hidden_inputs = LinkWeightsInputHidden.Multiply(inputs);
            // Calculate the signals emerging from hidden layer
            var hidden_outputs = hidden_inputs.ActivationFunction();

            // Calculate signals into final output layer
            var final_inputs = LinkWeightsHiddenOutput.Multiply(hidden_outputs);
            // Calculate the signals emerging from final output layer
            var final_outputs = final_inputs.ActivationFunction();

            // Output layer error is the (target - actual)
            var output_errors = targets - final_outputs;
            // Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
            var hidden_errors = LinkWeightsHiddenOutput.Transpose().Multiply(output_errors);

            // Update the weights for the links between the hidden and output layers
            var linkWeightsHiddenOutput = LinkWeightsHiddenOutput + _learningRate * (output_errors.Apply(final_outputs, (oe, fo) => { return oe * fo * (1.0f - fo); }).Multiply(hidden_outputs.Transpose()));
            // Update the weights for the links between the input and hidden layers
            var linkWeightsInputHidden = LinkWeightsInputHidden + _learningRate * (hidden_errors.Apply(hidden_outputs, (he, ho) => { return he * ho * (1.0f - ho); }).Multiply(inputs.Transpose()));

            return (linkWeightsHiddenOutput, linkWeightsInputHidden);
        }

        public void UpdateNeuralNetwork(IEnumerable<Matrix> linkWeightsHiddenOutputs, IEnumerable<Matrix> linkWeightsInputHiddens)
        {
            LinkWeightsHiddenOutput = linkWeightsHiddenOutputs.Average();
            LinkWeightsInputHidden = linkWeightsInputHiddens.Average();
        }

        /// <summary>
        /// Query the neural network
        /// </summary>
        public List<float> Query(Matrix input)
        {
            // Calculate signals into hidden layer
            var hidden_inputs = LinkWeightsInputHidden.Multiply(input);

            // Calculate the signals emerging from hidden layer
            var hidden_outputs = hidden_inputs.ActivationFunction();

            // Calculate signals into final output layer
            var final_inputs = LinkWeightsHiddenOutput.Multiply(hidden_outputs);

            // Calculate the signals emerging from final output layer
            var final_outputs = final_inputs.ActivationFunction();

            return final_outputs.Flatten();
        }

        private void InitLinkWeights(Matrix matrix, double mean, double stdDev)
        {
            for (int rowIndex = 0; rowIndex < matrix.RowCount; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < matrix.ColumnCount; columnIndex++)
                {
                    matrix[rowIndex, columnIndex] = (float)_random.NextGaussian(mean, stdDev);
                }
            }
        }
    }
}
