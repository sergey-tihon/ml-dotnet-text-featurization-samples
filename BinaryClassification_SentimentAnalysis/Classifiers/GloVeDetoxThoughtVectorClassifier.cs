using System;
using System.IO;
using System.Text;

using Microsoft.ML;
using Microsoft.ML.Data;

using TextFeaturization.Common;

namespace TextFeaturization.Classifiers
{
    public class GloVeDetoxThoughtVectorClassifier: BinaryClassifier
    {
        public GloVeDetoxThoughtVectorClassifier()
            : base("Glove_DetoxVectors")
        {}

        class InputRow
        {
            public string[] Tokens { get; set; }
        }
        class OutputRow
        {
            [VectorType(20)]
            public float[] ThoughtVector { get; set; }
        }

        protected override IEstimator<ITransformer> GetDataProcessPipeline()
        {
            var glove = new GloVeDictionary(
                Path.Combine(Helpers.DataFolder, "glove-detox/vectors.txt"),
                20, 40_000);

            void Mapping(InputRow input, OutputRow output)
                => output.ThoughtVector = glove.GetThoughtVector(input.Tokens);

            return GetTokenizationPipeline()
                .Append(Context.Transforms.CustomMapping((Action<InputRow, OutputRow>) Mapping, nameof(Mapping)))
                .Append(Context.Transforms.CopyColumns("Features", "ThoughtVector"))
                .Append(Context.Transforms.DropColumns("NormalizedMessage", "Tokens", "ThoughtVector"))
                .AppendCacheCheckpoint(Context);
        }

        private IEstimator<ITransformer> GetTokenizationPipeline() =>
            Context.Transforms.Text.NormalizeText("NormalizedMessage", nameof(SentimentIssue.Text), keepPunctuations: false, keepNumbers: false)
                .Append(Context.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedMessage",new[] { ' ', '=', '!', '~', '|'}))
                .Append(Context.Transforms.Text.RemoveDefaultStopWords("Tokens"));

        public void PrepareGloveTrainingData()
        {
            var dataView = Context.Data.LoadFromTextFile<SentimentIssue>(Helpers.DataPath, hasHeader: true);
            var trainedModel = GetTokenizationPipeline().Fit(dataView);
            var tokenizedData = trainedModel.Transform(dataView);

            var sb = new StringBuilder();
            foreach (var row in tokenizedData.GetColumn<string[]>("Tokens"))
            {
                foreach (var word in row)
                {
                    sb.Append(word);
                    sb.Append(" ");
                }

                sb.Append("\n");
            }
            File.WriteAllText(Path.Combine(Helpers.DataFolder, "glove-detox/tokens.txt"), sb.ToString());
        }
    }
}
