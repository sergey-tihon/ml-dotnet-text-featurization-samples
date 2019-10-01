using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading.Tasks.Dataflow;

namespace TextFeaturization.Common
{
    public class GloVeDictionary
    {
        public GloVeDictionary(string filePath, int wordVectorLength, int capacity)
        {
            _wordVectorLength = wordVectorLength;
            _wordVectors = new float[capacity, wordVectorLength];
            _word2Index = new Dictionary<string, int>(capacity, StringComparer.InvariantCultureIgnoreCase);

            var updateDict = new ActionBlock<(string word, int pos)>(x => _word2Index.TryAdd(x.word, x.pos));

            var parseLine = new TransformBlock<(string line, int pos), (string word, int pos) >(x =>
            {
                var ind = x.line.IndexOf(' ');
                var word = x.line.Substring(0, ind);

                var span = x.line.AsSpan();
                for (var i = 0; i < wordVectorLength; i++)
                {
                    while (span[ind] == ' ') ind++;
                    var pos = ind;
                    while (pos < span.Length && span[pos] != ' ') pos++;

                    var s = span.Slice(ind, pos - ind);
                    _wordVectors[x.pos, i] = float.Parse(s);
                    ind = pos;
                }

                return(word, x.pos);
            }, new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = Environment.ProcessorCount});

            parseLine.LinkTo(updateDict, new DataflowLinkOptions {PropagateCompletion = true});

            var position = 0;
            foreach(var line in File.ReadLines(filePath))
                parseLine.Post((line, position++));

            parseLine.Complete();
            updateDict.Completion.GetAwaiter().GetResult();
        }

        private readonly int _wordVectorLength;
        private readonly float[,] _wordVectors;
        private readonly Dictionary<string, int> _word2Index;

        private readonly Regex _regex = new Regex("[^a-zA-Z']", RegexOptions.Compiled);
        private bool TryGetWordVector(string word, out int index)
        {
            if (_word2Index.TryGetValue(word, out index))
                return true;

            word = _regex.Replace(word, string.Empty);
            return _word2Index.TryGetValue(word, out index);
        }

        public float[] GetThoughtVector(IEnumerable<string> words)
        {
            var result = new float[_wordVectorLength];
            foreach (var word in words)
            {
                if (!TryGetWordVector(word, out var ind))
                    continue;

                for (var i = 0; i < _wordVectorLength; i++)
                    result[i] += _wordVectors[ind, i];
            }
            return result;
        }
    }
}
