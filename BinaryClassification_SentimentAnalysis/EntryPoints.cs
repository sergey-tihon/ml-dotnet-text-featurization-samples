using NUnit.Framework;

using TextFeaturization.Classifiers;

namespace TextFeaturization
{
    [TestFixture]
    public class EntryPoints
    {
        private static BinaryClassifier[] _classifiers =
        {
            new BinaryClassifier(),
            new FeaturizeTextOptionsClassifier(),
            //new GloVeThoughtVectorClassifier(),
        };

        [TestCaseSource(nameof(_classifiers))]
        public void TrainModel(BinaryClassifier classifier)
            => classifier.TrainModel();

        [TestCaseSource(nameof(_classifiers))]
        public void SaveFinalView(BinaryClassifier classifier)
            => classifier.SaveTransformedView();

        [TestCaseSource(nameof(_classifiers))]
        public void SaveErrors(BinaryClassifier classifier)
            => classifier.Errors();
    }
}
