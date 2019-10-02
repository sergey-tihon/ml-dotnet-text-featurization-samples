using System;
using System.IO;

using Microsoft.ML;
using Microsoft.ML.Data;

using TextFeaturization.Common;

namespace TextFeaturization.Classifiers
{
    public class MixedClassifier: BinaryClassifier
    {
        public MixedClassifier()
            : base("Mixed")
        {}

        class InputRow
        {
            public string[] Tokens { get; set; }
        }
        class OutputRow
        {
            [VectorType(200)]
            public float[] ThoughtVector { get; set; }
        }

        protected override IEstimator<ITransformer> GetDataProcessPipeline()
        {
            var glove = new GloVeDictionary(
                Path.Combine(Helpers.DataFolder, "glove-twitter/glove.twitter.27B.200d.txt"),
                200, 1_200_000);

            void Mapping(InputRow input, OutputRow output)
                => output.ThoughtVector = glove.GetThoughtVector(input.Tokens);

            return Context.Transforms.Text.FeaturizeText("TextFeatures", FeaturizeTextOptionsClassifier.Options, nameof(SentimentIssue.Text))
                .Append(Context.Transforms.Text.NormalizeText("NormalizedMessage", nameof(SentimentIssue.Text), keepPunctuations:false, keepNumbers:false))
                .Append(Context.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedMessage", new[] { ' ', '=', '!', '~', '|'}))
                .Append(Context.Transforms.Text.RemoveDefaultStopWords("Tokens"))
                .Append(Context.Transforms.CustomMapping((Action<InputRow, OutputRow>) Mapping, nameof(Mapping)))

                .Append(Context.Transforms.Concatenate("Features", "ThoughtVector", "TextFeatures"))
                .Append(Context.Transforms.DropColumns("NormalizedMessage", "Tokens", "ThoughtVector", "TextFeatures"))
                .AppendCacheCheckpoint(Context);
        }
    }
}
