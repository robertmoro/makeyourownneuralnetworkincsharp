using ReactiveUI;
using System.Reactive.Disposables;

namespace NeuralNetworkUserInterface
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainView : ReactiveWindow<MainViewModel>
    {
        public MainView()
        {
            InitializeComponent();

            ViewModel = new MainViewModel();

            this.WhenActivated(disposableRegistration =>
            {
                // Browse MNIST folder command
                this.BindCommand(ViewModel,
                    viewModel => viewModel.BrowseMnistDatabaseFolderCommand,
                    view => view.BrowseMnistDatabaseFolder)
                    .DisposeWith(disposableRegistration);

                this.Bind(ViewModel,
                    viewModel => viewModel.LocationOfMnistFiles,
                    view => view.LocationOfMnistFiles.Text)
                    .DisposeWith(disposableRegistration);

                // Load training command
                this.BindCommand(ViewModel,
                    viewModel => viewModel.LoadTrainingSet,
                    view => view.LoadTrainingSet)
                    .DisposeWith(disposableRegistration);

                // Load training progressbar
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.LoadTrainingProgressBarMax,
                    view => view.LoadTrainingSetProgressBar.Maximum)
                    .DisposeWith(disposableRegistration);

                this.Bind(ViewModel,
                    viewModel => viewModel.TrainingSetCount,
                    view => view.LoadTrainingSetProgressBar.Value)
                    .DisposeWith(disposableRegistration);

                //
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.TrainingSetCount,
                    view => view.TrainingSetCount.Text)
                    .DisposeWith(disposableRegistration);

                // Stop load button
                this.BindCommand(ViewModel,
                    viewModel => viewModel.StopLoadTrainingSet,
                    view => view.StopLoadTrainingSet)
                    .DisposeWith(disposableRegistration);

                // Training set size selection using slider
                this.Bind(ViewModel,
                    viewModel => viewModel.TrainingSetSizeValue,
                    view => view.LoadTrainingSetSlider.Value)
                    .DisposeWith(disposableRegistration);

                // Training set slider maximum
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.TrainingSetCount,
                    view => view.LoadTrainingSetSlider.Maximum)
                    .DisposeWith(disposableRegistration);

                this.OneWayBind(ViewModel,
                    viewModel => viewModel.TrainingSetSizeValue,
                    view => view.TrainingSetSizeValue.Text)
                    .DisposeWith(disposableRegistration);

                // Learning rate
                this.Bind(ViewModel,
                    viewModel => viewModel.LearningRateValue,
                    view => view.LearningRateSlider.Value)
                    .DisposeWith(disposableRegistration);

                this.OneWayBind(ViewModel,
                    viewModel => viewModel.LearningRateValue,
                    view => view.LearningRateValue.Text,
                    ViewModelToViewConverterFunc)
                    .DisposeWith(disposableRegistration);

                // Epoch slider value as position
                this.Bind(ViewModel,
                    viewModel => viewModel.EpochValue,
                    view => view.EpochSlider.Value)
                    .DisposeWith(disposableRegistration);

                // Epoch slider value as text
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.EpochValue,
                    view => view.EpochValue.Text)
                    .DisposeWith(disposableRegistration);

                // Minibatch slider position
                this.Bind(ViewModel,
                    viewModel => viewModel.MinibatchValue,
                    view => view.MinibatchSlider.Value)
                    .DisposeWith(disposableRegistration);

                // Minibatch slider value as text
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.MinibatchValue,
                    view => view.MinibatchValue.Text)
                    .DisposeWith(disposableRegistration);

                // Run training command
                this.BindCommand(ViewModel,
                    viewModel => viewModel.RunTraining,
                    view => view.RunTraining)
                    .DisposeWith(disposableRegistration);

                // Run training progressbar
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.RunTrainingCount,
                    view => view.RunTrainingProgressBar.Maximum)
                    .DisposeWith(disposableRegistration);

                this.Bind(ViewModel,
                    viewModel => viewModel.RunTrainingProgress,
                    view => view.RunTrainingProgressBar.Value)
                    .DisposeWith(disposableRegistration);

                this.OneWayBind(ViewModel,
                    viewModel => viewModel.RunTrainingProgress,
                    view => view.RunTrainingValue.Text)
                    .DisposeWith(disposableRegistration);

                this.OneWayBind(ViewModel,
                    viewModel => viewModel.RunTrainingCount,
                    view => view.RunTrainingCount.Text)
                    .DisposeWith(disposableRegistration);

                // Load test command
                this.BindCommand(ViewModel,
                    viewModel => viewModel.LoadTestSet,
                    view => view.LoadTestSet)
                    .DisposeWith(disposableRegistration);

                // Load test progressbar
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.LoadTestProgressBarMax,
                    view => view.LoadTestSetProgressBar.Maximum)
                    .DisposeWith(disposableRegistration);

                this.Bind(ViewModel,
                    viewModel => viewModel.TestSetCount,
                    view => view.LoadTestSetProgressBar.Value)
                    .DisposeWith(disposableRegistration);

                //
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.TestSetCount,
                    view => view.TestSetCount.Text)
                    .DisposeWith(disposableRegistration);

                // Test set size selection using slider
                this.Bind(ViewModel,
                    viewModel => viewModel.TestSetSizeValue,
                    view => view.LoadTestSetSlider.Value)
                    .DisposeWith(disposableRegistration);

                // Test set slider maximum
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.TestSetCount,
                    view => view.LoadTestSetSlider.Maximum)
                    .DisposeWith(disposableRegistration);

                this.OneWayBind(ViewModel,
                    viewModel => viewModel.TestSetSizeValue,
                    view => view.TestSetSizeValue.Text)
                    .DisposeWith(disposableRegistration);

                // Accuracy
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.Accuracy,
                    view => view.Accuracy.Text,
                    ViewModelToViewConverterFunc)
                    .DisposeWith(disposableRegistration);

                // Run test command
                this.BindCommand(ViewModel,
                    viewModel => viewModel.RunTest,
                    view => view.RunTest)
                    .DisposeWith(disposableRegistration);

                // Run test progressbar
                this.OneWayBind(ViewModel,
                    viewModel => viewModel.TestSetSizeValue,
                    view => view.RunTestProgressBar.Maximum)
                    .DisposeWith(disposableRegistration);

                this.Bind(ViewModel,
                    viewModel => viewModel.RunTestProgress,
                    view => view.RunTestProgressBar.Value)
                    .DisposeWith(disposableRegistration);

                this.OneWayBind(ViewModel,
                    viewModel => viewModel.RunTestProgress,
                    view => view.RunTestValue.Text)
                    .DisposeWith(disposableRegistration);

                this.OneWayBind(ViewModel,
                    viewModel => viewModel.TestSetSizeValue,
                    view => view.RunTestCount.Text)
                    .DisposeWith(disposableRegistration);
            });
        }
        private string ViewModelToViewConverterFunc(float value)
        {
            return value.ToString("F");
        }
    }
}
