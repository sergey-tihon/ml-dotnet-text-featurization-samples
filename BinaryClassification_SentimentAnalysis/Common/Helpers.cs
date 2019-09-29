using System.Diagnostics;
using System.IO;

namespace TextFeaturization.Common
{
    public class Helpers
    {
        public static readonly string DataPath = GetAbsolutePath("../../../../data/wikiDetoxAnnotated40kRows.tsv");
        public static readonly string OutputFolder = GetAbsolutePath("../../../Output");

        public static readonly string GloVeFolder = GetAbsolutePath("../../../GloVe");

        private static string GetAbsolutePath(string relativePath)
        {
            var dataRoot = new FileInfo(typeof(Helpers).Assembly.Location);
            Debug.Assert(dataRoot.Directory != null, "dataRoot.Directory != null");

            var assemblyFolderPath = dataRoot.Directory.FullName;
            return Path.Combine(assemblyFolderPath , relativePath);
        }
    }
}
