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

            LinkWeightsHiddenOutput = new Matrix(outputNodes, hiddenNodes);
            LinkWeightsHiddenOutput.Init(() => (float)_random.NextGaussian(1.0, 4.0));
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

            UpdateNeuralNetwork(result.LinkWeightsHiddenOutput, result.LinkWeightsInputHidden);

            // Update the weights for the links between the hidden and output layers
            LinkWeightsHiddenOutput = result.LinkWeightsHiddenOutput;
            // Update the weights for the links between the input and hidden layers
            LinkWeightsInputHidden = result.LinkWeightsInputHidden;
        }

        public (Matrix LinkWeightsHiddenOutput, Matrix LinkWeightsInputHidden) Process(Matrix inputs, Matrix targets)
        {
            // Calculate signals into hidden layer
            var hiddenInputs = LinkWeightsInputHidden.Multiply(inputs);
            // Calculate the signals emerging from hidden layer
            var hiddenOutputs = hiddenInputs.ActivationFunction();

            // Calculate signals into final output layer
            var finalInputs = LinkWeightsHiddenOutput.Multiply(hiddenOutputs);
            // Calculate the signals emerging from final output layer
            var finalOutputs = finalInputs.ActivationFunction();

            // Output layer error is the (target - actual)
            var outputErrors = targets - finalOutputs;
            // Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
            var hiddenErrors = LinkWeightsHiddenOutput.Transpose().Multiply(outputErrors);

            // Update the weights for the links between the hidden and output layers
            var linkWeightsHiddenOutput = LinkWeightsHiddenOutput + _learningRate * (outputErrors.Apply(finalOutputs, (oe, fo) => { return oe * fo * (1.0f - fo); }).Multiply(hiddenOutputs.Transpose()));
            // Update the weights for the links between the input and hidden layers
            var linkWeightsInputHidden = LinkWeightsInputHidden + _learningRate * (hiddenErrors.Apply(hiddenOutputs, (he, ho) => { return he * ho * (1.0f - ho); }).Multiply(inputs.Transpose()));

            return (linkWeightsHiddenOutput, linkWeightsInputHidden);
        }

        public void UpdateNeuralNetwork(IEnumerable<Matrix> linkWeightsHiddenOutputs, IEnumerable<Matrix> linkWeightsInputHiddens)
        {
            LinkWeightsHiddenOutput = linkWeightsHiddenOutputs.Average();
            LinkWeightsInputHidden = linkWeightsInputHiddens.Average();
        }

        public void UpdateNeuralNetwork(Matrix linkWeightsHiddenOutput, Matrix linkWeightsInputHidden)
        {
            LinkWeightsHiddenOutput = linkWeightsHiddenOutput;
            LinkWeightsInputHidden = linkWeightsInputHidden;
        }

        /// <summary>
        /// Query the neural network
        /// </summary>
        public List<float> Query(Matrix input)
        {
            // Calculate signals into hidden layer
            var hiddenInputs = LinkWeightsInputHidden.Multiply(input);

            // Calculate the signals emerging from hidden layer
            var hiddenOutputs = hiddenInputs.ActivationFunction();

            // Calculate signals into final output layer
            var finalInputs = LinkWeightsHiddenOutput.Multiply(hiddenOutputs);

            // Calculate the signals emerging from final output layer
            var finalOutputs = finalInputs.ActivationFunction();

            return finalOutputs.Flatten();
        }
    }
}
