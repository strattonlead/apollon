using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.Linq;

namespace ConsoleTester
{
    public class Programm
    {
        public static void Main(string[] args)
        {
            var items = new ModelInput[]
            {
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 1), Reason = "KR", Absence = 1 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 2), Reason = "KR", Absence = 1 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 3), Reason = "KR", Absence = 1 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 4), Reason = "KR", Absence = 1 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 5), Reason = "KR", Absence = 1 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 6), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 7), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 8), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 9), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 10), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 11), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 12), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 13), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 14), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 15), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 16), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 17), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 18), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 19), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 20), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 21), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 22), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 23), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 24), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 25), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 26), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 27), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 28), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 29), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 30), Reason = null, Absence = 0 },
                new ModelInput() { Age = 18, Date = new DateTime(2022, 1, 31), Reason = null, Absence = 0 }
            };

            var mlContext = new MLContext();
            var dataView = mlContext.Data.LoadFromEnumerable(items);

            var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedAbsence",
                inputColumnName: "Absence",
                windowSize: 14,
                seriesLength: 30,
                trainSize: 365,
                horizon: 14,
                confidenceLevel: 0.95f,
                confidenceLowerBoundColumn: "LowerBoundAbsence",
                confidenceUpperBoundColumn: "UpperBoundAbsence");

            var forecaster = forecastingPipeline.Fit(dataView);

            Evaluate(secondYearData, forecaster, mlContext);

            var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
            forecastEngine.CheckPoint(mlContext, "myModel.zip");

            Forecast(secondYearData, 7, forecastEngine, mlContext);
        }

        public static void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
        {
            var predictions = model.Transform(testData);

            var actual = mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
                .Select(observed => observed.Absence);

            var forecast = mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
                    .Select(prediction => prediction.ForecastedAbsence[0]);

            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

            var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
            var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------");
            Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
            Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
        }

        public static void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
        {
            ModelOutput forecast = forecaster.Predict();

            var forecastOutput = mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
                .Take(horizon)
                .Select((ModelInput rental, int index) =>
                {
                    string rentalDate = rental.Date.ToShortDateString();
                    float actualAbsence = rental.Absence;
                    float lowerEstimate = Math.Max(0, forecast.LowerBoundAbsence[index]);
                    float estimate = forecast.ForecastedAbsence[index];
                    float upperEstimate = forecast.UpperBoundAbsence[index];
                    return $"Date: {rentalDate}\n" +
                    $"Actual Absence: {actualAbsence}\n" +
                    $"Lower Estimate: {lowerEstimate}\n" +
                    $"Forecast: {estimate}\n" +
                    $"Upper Estimate: {upperEstimate}\n";
                });

            Console.WriteLine("Absence Forecast");
            Console.WriteLine("---------------------");
            foreach (var prediction in forecastOutput)
            {
                Console.WriteLine(prediction);
            }
        }
    }

    public class ModelInput
    {
        public int Age { get; set; }
        public DateTime Date { get; set; }
        public string Reason { get; set; }
        public float Absence { get; set; }
    }

    public class ModelOutput
    {
        public float[] ForecastedAbsence { get; set; }

        public float[] LowerBoundAbsence { get; set; }

        public float[] UpperBoundAbsence { get; set; }
    }
}