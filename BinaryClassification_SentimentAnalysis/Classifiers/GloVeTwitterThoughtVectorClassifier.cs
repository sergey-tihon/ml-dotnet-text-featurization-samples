using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

using Microsoft.ML;
using Microsoft.ML.Data;

using TextFeaturization.Common;

namespace TextFeaturization.Classifiers
{
    public class GloVeTwitterThoughtVectorClassifier: BinaryClassifier
    {
        public GloVeTwitterThoughtVectorClassifier()
            : base("Glove_TwitterVectors")
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

            return Context.Transforms.Text.NormalizeText("NormalizedMessage", nameof(SentimentIssue.Text), keepPunctuations:false, keepNumbers:false)
                .Append(Context.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedMessage"))
                //.Append(Context.Transforms.Text.RemoveDefaultStopWords("Tokens"))
                .Append(Context.Transforms.CustomMapping((Action<InputRow, OutputRow>) Mapping, nameof(Mapping)))

                .Append(Context.Transforms.CopyColumns("Features", "ThoughtVector"))
                .Append(Context.Transforms.DropColumns("NormalizedMessage", "Tokens", "ThoughtVector"))
                .AppendCacheCheckpoint(Context);
        }
    }
}
