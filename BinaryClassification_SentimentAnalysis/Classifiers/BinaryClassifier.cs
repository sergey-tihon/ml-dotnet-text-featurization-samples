using System;
using System.Collections.Generic;
using System.IO;

using Microsoft.ML;
using Microsoft.ML.Data;

using TextFeaturization.Common;

namespace TextFeaturization.Classifiers
{
    /// <summary>
    /// This code is based on Sentient Analysis sample for ML.NET
    /// see - https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/BinaryClassification_SentimentAnalysis
    /// </summary>
    public class BinaryClassifier
    {
        public BinaryClassifier(string modelName = "FeaturizeText_Default")
        {
            _outputFolder = Path.Combine(Helpers.OutputFolder, modelName);
            if (!Directory.Exists(_outputFolder))
                Directory.CreateDirectory(_outputFolder);

            _modelFile = Path.Combine(_outputFolder, "model.zip");

            // Create MLContext to be shared across the model creation workflow objects
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            Context = new MLContext(seed: 1);
        }

        private readonly string _outputFolder;
        private readonly string _modelFile;

        protected readonly MLContext Context;

        protected virtual IEstimator<ITransformer> GetDataProcessPipeline() =>
            // https://github.com/dotnet/machinelearning/blob/master/docs/code/MlNetCookBook.md#how-do-i-train-my-model-on-textual-data
            Context.Transforms.Text.FeaturizeText(outputColumnName: "Features", nameof(SentimentIssue.Text));

        public void TrainModel()
        {
            // STEP 1: Common data loading configuration
            var dataView = Context.Data.LoadFromTextFile<SentimentIssue>(Helpers.DataPath, hasHeader: true);

            var trainTestSplit = Context.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainingData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = GetDataProcessPipeline();

            // STEP 3: Set the training algorithm, then create and config the modelBuilder
            var trainer = Context.BinaryClassification.Trainers.SdcaLogisticRegression();
            //var trainer = Context.BinaryClassification.Trainers.LightGbm();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);

            // STEP 5: Evaluate the model and show accuracy stats
            var predictions = trainedModel.Transform(testData);
            var metrics = Context.BinaryClassification.Evaluate(data: predictions);
            PrintBinaryClassificationMetrics(metrics);

            // STEP 6: Save/persist the trained model to a .ZIP file
            Context.Model.Save(trainedModel, trainingData.Schema, _modelFile);
            Console.WriteLine($"The model is saved to {_modelFile}");
        }

        public void SaveTransformedView()
        {
            var dataView = Context.Data.LoadFromTextFile<SentimentIssue>(Helpers.DataPath, hasHeader: true);
            var dataProcessPipeline = GetDataProcessPipeline();
            var finalView = dataProcessPipeline.Fit(dataView).Transform(dataView);

            using var stream = File.Create(Path.Combine(_outputFolder, "transformedView.tsv"));
            Context.Data.SaveAsText(finalView, stream);
        }

        public void Errors()
        {
            var trainedModel = Context.Model.Load(_modelFile, out _);
            // Create prediction engine related to the loaded trained model
            var predEngine = Context.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

            var dataView = Context.Data.LoadFromTextFile<SentimentIssue>(Helpers.DataPath, hasHeader: true);
            var issues = Context.Data.CreateEnumerable<SentimentIssue>(dataView, reuseRowObject: false);

            var errors = new List<SentimentIssue>();
            foreach (var issue in issues)
            {
                var pred = predEngine.Predict(issue);
                if (pred.Prediction != issue.Label)
                    errors.Add(issue);
            }

            dataView = Context.Data.LoadFromEnumerable(errors);
            using var stream = File.Create(Path.Combine(_outputFolder, "errors.tsv"));
            Context.Data.SaveAsText(dataView, stream);
        }

        private void PrintBinaryClassificationMetrics(CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {GetType().Name} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"************************************************************");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine($"************************************************************");
        }
    }
}
