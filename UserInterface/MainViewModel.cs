using MathNet.Numerics.LinearAlgebra;
using MnistDatabase;
using NeuralNetworkUsingMathLibrary;
using ReactiveUI;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reactive;
using System.Reactive.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetworkUserInterface
{
    public class MainViewModel : ReactiveObject
    {
        private readonly NeuralNetwork _neuralNetwork;
        private CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();
        private List<(Vector<float> Input, byte ExpectedDigit, Vector<float> ExpectedOutput)> _trainingData = new List<(Vector<float> Input, byte ExpectedDigit, Vector<float> ExpectedOutput)>();
        private List<(Vector<float> Input, byte ExpectedOutput)> _testData = new List<(Vector<float> Input, byte ExpectedOutput)>();
        private string _locationOfMnistFiles;
        private int _trainingSetSizeValue;
        private float _learingRateValue;
        private int _epochValue;
        private int _minibatchValue;
        private long _loadTrainingProgressBarMax;
        private int _trainingSetCount;
        private int _runTrainingProgress;
        private int _runTrainingCount;
        private long _loadTestProgressBarMax;
        private int _testSetCount;
        private int _testSetSize;
        private int _runTestProgress;
        private float _accuracy;
        private readonly IMnistReader _mnistReader;

        public MainViewModel()
        {
            LearningRateValue = 0.1f;
            LoadTrainingProgressBarMax = 1; // Avoid full progress bar after start
            RunTrainingProgress = 0;
            RunTrainingCount = 1;
            LoadTestProgressBarMax = 1;     // Avoid full progress bar after start

            _mnistReader = new MnistReader();

            Progress<int> runTrainingProgress = new Progress<int>(i => RunTrainingProgress = i);
            Progress<int> runTestProgress = new Progress<int>(i => RunTestProgress = i);

            _neuralNetwork = new NeuralNetwork(784, 100, 10, LearningRateValue);

            LocationOfMnistFiles = Properties.Settings.Default.LocationOfMnistFiles;

            this.WhenAnyValue(x => x.EpochValue, x => x.TrainingSetSizeValue)
                .Select(tuple => tuple.Item1 * tuple.Item2)
                .Do(total => RunTrainingCount = total)
                .Subscribe();

            this.WhenAnyValue(x => x.LocationOfMnistFiles)
                .Select(x => Unit.Default)
                .InvokeCommand(this, x => x.StoreSettingsCommand);

            BrowseMnistDatabaseFolderCommand = ReactiveCommand.Create(BrowseMnistDatabaseFolder);
            LoadTrainingSet = ReactiveCommand.CreateFromTask(() => LoadTrainingSetCommandAsync());
            StopLoadTrainingSet = ReactiveCommand.Create(StopLoadTrainingSetCommand);
            RunTraining = ReactiveCommand.CreateFromTask(() => RunTrainingCommandAsync(runTrainingProgress), this.WhenAnyValue(x => x.TrainingSetSizeValue).Select(x => x > 0));
            LoadTestSet = ReactiveCommand.CreateFromTask(() => LoadTestSetCommandAsync());
            RunTest = ReactiveCommand.CreateFromTask(() => RunTestCommandAsync(runTestProgress), this.WhenAnyValue(x => x.TestSetSizeValue).Select(x => x > 0));
            StoreSettingsCommand = ReactiveCommand.Create(StoreSettings);
        }

        private void StoreSettings()
        {
            Properties.Settings.Default.LocationOfMnistFiles = LocationOfMnistFiles;
            Properties.Settings.Default.Save();
        }

        public ReactiveCommand<Unit, Unit> BrowseMnistDatabaseFolderCommand { get; }
        public ReactiveCommand<Unit, Unit> LoadTrainingSet { get; }
        public ReactiveCommand<Unit, Unit> StopLoadTrainingSet { get; }
        public ReactiveCommand<Unit, Unit> RunTraining { get; }
        public ReactiveCommand<Unit, Unit> CancelTraining { get; }
        public ReactiveCommand<Unit, Unit> LoadTestSet { get; }
        public ReactiveCommand<Unit, Unit> RunTest { get; }
        public ReactiveCommand<Unit, Unit> CancelTest { get; }
        public ReactiveCommand<Unit, Unit> StoreSettingsCommand { get; }

        public string LocationOfMnistFiles
        {
            get => _locationOfMnistFiles;
            set => this.RaiseAndSetIfChanged(ref _locationOfMnistFiles, value);
        }

        public int TrainingSetCount
        {
            get => _trainingSetCount;
            set => this.RaiseAndSetIfChanged(ref _trainingSetCount, value);
        }
        public float LearningRateValue
        {
            get => _learingRateValue;
            set => this.RaiseAndSetIfChanged(ref _learingRateValue, value);
        }
        public long LoadTrainingProgressBarMax
        {
            get => _loadTrainingProgressBarMax;
            set => this.RaiseAndSetIfChanged(ref _loadTrainingProgressBarMax, value);
        }
        public int TrainingSetSizeValue
        {
            get => _trainingSetSizeValue;
            set => this.RaiseAndSetIfChanged(ref _trainingSetSizeValue, value);
        }
        public int EpochValue
        {
            get => _epochValue;
            set => this.RaiseAndSetIfChanged(ref _epochValue, value);
        }
        public int MinibatchValue
        {
            get => _minibatchValue;
            set => this.RaiseAndSetIfChanged(ref _minibatchValue, value);
        }
        public int RunTrainingProgress
        {
            get => _runTrainingProgress;
            set => this.RaiseAndSetIfChanged(ref _runTrainingProgress, value);
        }
        public int RunTrainingCount
        {
            get => _runTrainingCount;
            set => this.RaiseAndSetIfChanged(ref _runTrainingCount, value);
        }
        public long LoadTestProgressBarMax
        {
            get => _loadTestProgressBarMax;
            set => this.RaiseAndSetIfChanged(ref _loadTestProgressBarMax, value);
        }
        public int TestSetCount
        {
            get => _testSetCount;
            set => this.RaiseAndSetIfChanged(ref _testSetCount, value);
        }
        public int TestSetSizeValue
        {
            get => _testSetSize;
            set => this.RaiseAndSetIfChanged(ref _testSetSize, value);
        }
        public int RunTestProgress
        {
            get => _runTestProgress;
            set => this.RaiseAndSetIfChanged(ref _runTestProgress, value);
        }
        public float Accuracy
        {
            get => _accuracy;
            set => this.RaiseAndSetIfChanged(ref _accuracy, value);
        }

        private void BrowseMnistDatabaseFolder()
        {
            FolderBrowserDialog folderDialog = new FolderBrowserDialog();
            folderDialog.SelectedPath = "C:\\";

            if (folderDialog.ShowDialog() == DialogResult.OK)
            {
                LocationOfMnistFiles = folderDialog.SelectedPath;
            }
        }

        private void StopLoadTrainingSetCommand()
        {
            _cancellationTokenSource.Cancel();
        }

        // Train neural network
        private async Task RunTrainingCommandAsync(IProgress<int> progress)
        {
            await Task.Run(() =>
            {
                int runTrainingProgress = 0;

                foreach (var epoch in Enumerable.Range(0, EpochValue))
                {
                    foreach (var d in _trainingData.Take(TrainingSetSizeValue))
                    {
                        _neuralNetwork.Train(d.Input, d.ExpectedOutput);

                        progress.Report(++runTrainingProgress);
                    }
                }
            });
        }

        private async Task LoadTrainingSetCommandAsync()
        {
            TrainingSetCount = 0;

            var expectedValues = _mnistReader.LoadLabels(Path.Combine(LocationOfMnistFiles, "train-labels-idx1-ubyte.gz"));

            var images = await LoadImages(Path.Combine(LocationOfMnistFiles, "train-images-idx3-ubyte.gz"), v => LoadTrainingProgressBarMax = v, () => TrainingSetCount++);

            _trainingData.Clear();

            _trainingData = images.Zip(expectedValues, (image, ev) => (image, ev, ExtensionMethods.CreateTargetOutput(ev).ToVector())).ToList();
        }

        private async Task LoadTestSetCommandAsync()
        {
            TestSetCount = 0;

            var expectedValues = _mnistReader.LoadLabels(Path.Combine(LocationOfMnistFiles, "t10k-labels-idx1-ubyte.gz"));

            var images = await LoadImages(Path.Combine(LocationOfMnistFiles, "t10k-images-idx3-ubyte.gz"), v => LoadTestProgressBarMax = v, () => TestSetCount++);

            _testData.Clear();

            _testData = images.Zip(expectedValues, (image, ev) => (image, ev)).ToList();
        }

        private async Task<IEnumerable<Vector<float>>> LoadImages(string fileName, Action<long> setProgressBarMax, Action incrementCounter)
        {
            var images = new List<Vector<float>>();
            FileInfo fileToDecompress = new FileInfo(fileName);

            using (FileStream originalFileStream = fileToDecompress.OpenRead())
            using (GZipStream decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress))
            {
                byte[] buffer = new byte[28 * 28];

                // Read file header
                await ReadBytes(decompressionStream, buffer, 16);
                // Read number of images from header
                setProgressBarMax(buffer.ReadBigEndianInt32(4));

                // Read all the images
                while (await ReadBytes(decompressionStream, buffer, buffer.Length))
                {
                    images.Add(ExtensionMethods.Transform(buffer).ToVector());
                    incrementCounter();
                }
            }

            return images;
        }

        // Query neural network
        private async Task RunTestCommandAsync(IProgress<int> progress)
        {
            var results = new bool[TestSetSizeValue];

            await Task.Run(() =>
            {
                int runTestProgress = 0;

                Parallel.For(0, TestSetSizeValue, i =>
                {
                    var expectedValue = _testData[i].ExpectedOutput;
                    var actualValue = _neuralNetwork.Query(_testData[i].Input).Result();
                    results[i] = expectedValue == actualValue;

                    Interlocked.Increment(ref runTestProgress);
                    progress.Report(runTestProgress);
                });
            });

            Result result = new Result();
            results.ToList().ForEach(result.Increment);
            Accuracy = result.Accuracy;
        }

        private async Task<bool> ReadBytes(Stream decompressionStream, byte[] buffer, int count)
        {
            int offset = 0;
            int bytesRead = 0;
            while ((bytesRead = await decompressionStream.ReadAsync(buffer, offset, count - offset)) > 0)
            {
                if (bytesRead + offset < count)
                {
                    offset += bytesRead;
                }
                else
                {
                    return true;
                }
            }
            return false;
        }

        private class Result
        {
            private int _correct;
            private int _incorrect;

            public float Accuracy => ((float)_correct / (_incorrect + _correct)) * 100f;

            internal void Increment(bool result)
            {
                if (result)
                {
                    _correct++;
                }
                else
                {
                    _incorrect++;
                }
            }
        }
    }
}
