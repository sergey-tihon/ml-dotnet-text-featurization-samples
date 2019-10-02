using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

using TextFeaturization.Common;

namespace TextFeaturization.Classifiers
{
    class FeaturizeTextOptionsClassifier : BinaryClassifier
    {
        public FeaturizeTextOptionsClassifier()
            : base("FeaturizeText_Options")
        {}

        protected override IEstimator<ITransformer> GetDataProcessPipeline() {
            // https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.text.textfeaturizingestimator?view=ml-dotnet
            // https://github.com/dotnet/machinelearning/blob/ec6ff086edf685bdb64e244d14fd4ff8f5e6af4d/src/Microsoft.ML.Transforms/Text/TextFeaturizingEstimator.cs
            return Context.Transforms.Text.FeaturizeText("Features", Options, nameof(SentimentIssue.Text));
        }

        public static TextFeaturizingEstimator.Options Options =>
            new TextFeaturizingEstimator.Options
            {
                Norm = TextFeaturizingEstimator.NormFunction.L2,
                CaseMode = TextNormalizingEstimator.CaseMode.Lower,
                KeepDiacritics = false,
                KeepNumbers = false, // true
                KeepPunctuations = false, // true
                StopWordsRemoverOptions =  null,

                WordFeatureExtractor = new WordBagEstimator.Options
                {
                    NgramLength = 1,
                    SkipLength = 0,
                    UseAllLengths = true,
                    MaximumNgramsCount = new []{ 10000000 },
                    Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf // Tf
                },
                CharFeatureExtractor = new WordBagEstimator.Options()
                {
                    NgramLength = 3,
                    SkipLength = 0,
                    UseAllLengths = false,
                    MaximumNgramsCount = new []{ 10000000 },
                    Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf // Tf
                }
            };
    }
}
