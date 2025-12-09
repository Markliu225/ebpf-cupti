/*
 *  Copyright 2024 NVIDIA Corporation. All rights reserved
 *
 * This sample demonstrates the usage of the PM sampling feature in the CUDA Profiling Tools Interface (CUPTI).
 * There are two parts to this sample:
 * 1) Querying the available metrics for a chip and their properties.
 * 2) Collecting PM sampling data for a CUDA workload.
 *
 * The PM sampling feature allows users to collect sampling data at a specific interval for the CUDA workload
 * launched in the application.
 * For the continuous collection usecase, two separate threads are required:
 * 1. A main thread for launching the CUDA workload.
 * 2. A decode thread for decoding the collected data at a certain interval.
 *
 * The user is responsible for continuously calling the decode API, which frees up the hardware buffer for storing new data.
 *
 * In this sample, In the main thread the CUDA workload is launched. This workload is a simple vector addition implemented
 * in the `VectorAdd` kernel.
 *
 * The decode thread where we call the `DecodeCounterData` API. This API decodes the raw PM sampling data stored
 * in the hardware to a counter data image that the user has allocated.
 *
 */

#include <atomic>
#include <chrono>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <thread>
#include <signal.h>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include "pm_sampling.h"

std::atomic<bool> stopDecodeThread(false);
std::atomic<bool> samplingStopRequested(false);
// Monitoring-only tool: no built-in workload.

struct ParsedArgs
{
    bool isDeviceIndexSet = false;
    bool isChipNameSet = false;
    int deviceIndex = 0;
    int queryBaseMetrics = 0;
    int queryMetricProperties = 0;
    std::string chipName;
    uint64_t samplingInterval = 10000000; // 100us
    size_t hardwareBufferSize = 512 * 1024 * 1024; // 512MB
    uint64_t maxSamples = 100000;
    uint64_t durationSec = 10;          // monitor duration
    uint64_t decodeIntervalMs = 1000;    // decode + print interval
    int livePrint = 1;                  // print results during decode loop
    int holdOnExit = 1;                 // hold console open after completion
    int writeCsv = 1;                   // export samples to CSV after capture
    std::string csvOutputPath = "pm_sampling_metrics.csv";
    std::vector<const char*> metrics =
    {
        "sm__inst_executed_realtime.sum.per_cycle_active",  // Inst Executed per Active Cycle
        "sm__cycles_active.sum",                            // SM Active Cycles
        "l1tex__t_bytes.sum",                               // L1 Tex Bytes
        "sm__inst_executed_realtime.sum",                  // Inst Executed
        "dram__bytes.sum",                                 // DRAM throughput (memory power proxy)
        "dram__cycles_active.sum",                          // DRAM active cycles (memory controller load)
        // "gpu__time_active.sum",                             // GPU active time (device utilization)
    };
};

ParsedArgs parseArgs(int argc, char *argv[]);
void PmSamplingDeviceSupportStatus(CUdevice device);
int PmSamplingCollection(std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args);
int PmSamplingQueryMetrics(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args);
void DecodeCounterData(
    std::vector<uint8_t>& counterDataImage,
    std::vector<const char*> metricsList,
    CuptiPmSampling& cuptiPmSamplingTarget,
    CuptiProfilerHost& pmSamplingHost,
    uint64_t decodeIntervalMs,
    int livePrint,
    CUptiResult& result
);

void HandleSignal(int)
{
    samplingStopRequested.store(true);
}

int main(int argc, char *argv[])
{
    signal(SIGINT, HandleSignal);
    signal(SIGTERM, HandleSignal);
    ParsedArgs args = parseArgs(argc, argv);
    DRIVER_API_CALL(cuInit(0));

    std::string chipName = args.chipName;
    std::vector<uint8_t> counterAvailibilityImage;
    if ((args.isDeviceIndexSet && args.deviceIndex >= 0) || !args.isChipNameSet)
    {
        CUdevice cuDevice;
        DRIVER_API_CALL(cuDeviceGet(&cuDevice, args.deviceIndex));
        PmSamplingDeviceSupportStatus(cuDevice);

        CuptiPmSampling::GetChipName(args.deviceIndex, chipName);
        CuptiPmSampling::GetCounterAvailabilityImage(args.deviceIndex, counterAvailibilityImage);
    }

    int rc = 0;
    if (args.queryBaseMetrics || args.queryMetricProperties)
    {
        rc = PmSamplingQueryMetrics(chipName, counterAvailibilityImage, args);
    }
    else
    {
        rc = PmSamplingCollection(counterAvailibilityImage, args);
    }

    if (args.holdOnExit)
    {
        printf("\nPress Enter to exit...\n");
        fflush(stdout);
        int c;
        do { c = getchar(); } while (c != '\n' && c != '\r' && c != EOF);
    }
    return rc;
}

int PmSamplingQueryMetrics(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args)
{
    CuptiProfilerHost pmSamplingHost;
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);

    if (args.queryBaseMetrics)
    {
        std::vector<std::string> baseMetrics;
        CUPTI_API_CALL(pmSamplingHost.GetSupportedBaseMetrics(baseMetrics));
        printf("Base Metrics:\n");
        for (const auto& metric : baseMetrics)
        {
            printf("  %s\n", metric.c_str());
        }
        return 0;
    }

    if (args.queryMetricProperties)
    {
        for (const auto& metricName : args.metrics)
        {
            std::vector<std::string> subMetrics;
            CUPTI_API_CALL(pmSamplingHost.GetSubMetrics(metricName, subMetrics));
            printf("Sub Metrics for %s:\n", metricName);
            for (const auto& metric : subMetrics) {
                printf("  %s\n", metric.c_str());
            }

            std::string metricDescription;
            CUpti_MetricType metricType;
            CUPTI_API_CALL(pmSamplingHost.GetMetricProperties(metricName, metricType,metricDescription));

            printf("Metric Description: %s\n", metricDescription.c_str());
            printf("Metric Type: %s\n", metricType == CUPTI_METRIC_TYPE_COUNTER ? "Counter" : (metricType == CUPTI_METRIC_TYPE_RATIO) ? "Ratio" : "Throughput");
            printf("\n");
        }
        return 0;
    }

    pmSamplingHost.TearDown();
    return 0;
}

int PmSamplingCollection(std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args)
{
    samplingStopRequested = false;
    std::string chipName;
    CuptiPmSampling::GetChipName(args.deviceIndex, chipName);

    CuptiProfilerHost pmSamplingHost;
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);

    std::vector<uint8_t> configImage;
    CUPTI_API_CALL(pmSamplingHost.CreateConfigImage(args.metrics, configImage));

    CuptiPmSampling cuptiPmSamplingTarget;
    cuptiPmSamplingTarget.SetUp(args.deviceIndex);

    // 1. Enable PM sampling and set config for the PM sampling data collection.
    CUPTI_API_CALL(cuptiPmSamplingTarget.EnablePmSampling(args.deviceIndex));
    CUPTI_API_CALL(cuptiPmSamplingTarget.SetConfig(configImage, args.hardwareBufferSize, args.samplingInterval));

    // Time synchronization
    uint64_t cuptiStartTimestamp;
    CUPTI_API_CALL(cuptiGetTimestamp(&cuptiStartTimestamp));
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t monotonicStartTimestamp = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    std::cout << "TIME_SYNC: CUPTI=" << cuptiStartTimestamp << " MONOTONIC=" << monotonicStartTimestamp << std::endl;

    // 2. Create counter data image
    std::vector<uint8_t> counterDataImage;
    CUPTI_API_CALL(cuptiPmSamplingTarget.CreateCounterDataImage(args.maxSamples, args.metrics, counterDataImage));

    CUptiResult threadFuncResult;
    // 3. Launch the decode thread
    std::thread decodeThread(DecodeCounterData,
                            std::ref(counterDataImage),
                            std::ref(args.metrics),
                            std::ref(cuptiPmSamplingTarget),
                            std::ref(pmSamplingHost),
                            args.decodeIntervalMs,
                            args.livePrint,
                            std::ref(threadFuncResult));

    // 4. Start the PM sampling and monitor for duration
    CUPTI_API_CALL(cuptiPmSamplingTarget.StartPmSampling());
    stopDecodeThread = false;
    auto waitStep = std::chrono::milliseconds(200);
    if (args.durationSec == 0)
    {
        while (!samplingStopRequested.load())
        {
            std::this_thread::sleep_for(waitStep);
        }
    }
    else
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(args.durationSec);
        while (!samplingStopRequested.load() && std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(waitStep);
        }
    }

    // 5. Stop the PM sampling and join the decode thread
    CUPTI_API_CALL(cuptiPmSamplingTarget.StopPmSampling());
    stopDecodeThread = true;
    decodeThread.join();
    if (threadFuncResult != CUPTI_SUCCESS)
    {
        const char *errstr;
        cuptiGetResultString(threadFuncResult, &errstr);
        std::cerr << "DecodeCounterData Thread failed with error " << errstr << std::endl;
        return 1;
    }

    // 6. Print the sample ranges for the collected metrics
    pmSamplingHost.PrintSampleRanges();

    const size_t exportedSamples = pmSamplingHost.GetSamplerRanges().size();
    if (args.writeCsv)
    {
        if (pmSamplingHost.ExportSamplesToCsv(args.csvOutputPath, args.metrics))
        {
            std::cout << "Exported " << exportedSamples << " samples to CSV: " << args.csvOutputPath << "\n";
            if (exportedSamples == 0)
            {
                std::cout << "No samples were collected; CSV contains only headers.\n";
            }
            else
            {
                std::cout << "To visualize, run: python visualize_pm_sampling.py --csv " << args.csvOutputPath << "\n";
            }
        }
        else
        {
            std::cerr << "Failed to export samples to CSV at " << args.csvOutputPath << "\n";
        }
    }

    // 7. Disable PM sampling for release all the resources allocated in CUPTI
    CUPTI_API_CALL(cuptiPmSamplingTarget.DisablePmSampling());

    // 8. Clean up
    cuptiPmSamplingTarget.TearDown();
    pmSamplingHost.TearDown();
    return 0;
}

void DecodeCounterData( std::vector<uint8_t>& counterDataImage,
                        std::vector<const char*> metricsList,
                        CuptiPmSampling& cuptiPmSamplingTarget,
                        CuptiProfilerHost& pmSamplingHost,
                        uint64_t decodeIntervalMs,
                        int livePrint,
                        CUptiResult& result)
{
    while (!stopDecodeThread)
    {
        const char *errstr;
        result = cuptiPmSamplingTarget.DecodePmSamplingData(counterDataImage);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "DecodePmSamplingData failed with error " << errstr << std::endl;
            return;
        }

        CUpti_PmSampling_GetCounterDataInfo_Params counterDataInfo {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
        counterDataInfo.pCounterDataImage = counterDataImage.data();
        counterDataInfo.counterDataImageSize = counterDataImage.size();
        result = cuptiPmSamplingGetCounterDataInfo(&counterDataInfo);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "cuptiPmSamplingGetCounterDataInfo failed with error " << errstr << std::endl;
            return;
        }

        for (size_t sampleIndex = 0; sampleIndex < counterDataInfo.numCompletedSamples; ++sampleIndex)
        {
            pmSamplingHost.EvaluateCounterData(cuptiPmSamplingTarget.GetPmSamplerObject(), sampleIndex, metricsList, counterDataImage);
        }
        if (livePrint && counterDataInfo.numCompletedSamples > 0)
        {
            pmSamplingHost.PrintLastSamples(counterDataInfo.numCompletedSamples);
        }
        result = cuptiPmSamplingTarget.ResetCounterDataImage(counterDataImage);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "ResetCounterDataImage failed with error " << errstr << std::endl;
            return;
        }
        if (decodeIntervalMs > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(decodeIntervalMs));
        }
    }
}

void PrintHelp()
{
    printf("Usage:\n");
    printf("  Query Metrics:\n");
    printf("    List Base Metrics : ./pm_sampling --device/-d <deviceIndex> --chip/-c <chipname> --queryBaseMetrics/-q\n");
    printf("    List submetrics   : ./pm_sampling --device/-d <deviceIndex> --chip/-c <chipname> --metrics/-m <metric1,metric2,...> --queryMetricsProp/-p\n");
    printf("  Note: when device index flag is passed, the chip name flag will be ignored.\n");
    printf("  PM Sampling:\n");
    printf("    Monitor: ./pm_sampling --device/-d <deviceIndex> --samplingInterval/-i <ns> --maxsamples/-s <maxSamples> --hardwareBufferSize/-b <bytes> --metrics/-m <metric1,metric2,...> --durationSec/-t <sec> --decodeIntervalMs/-I <ms> [--livePrint/-L 0|1]\n");
    printf("  Misc:\n");
    printf("    Export samples to CSV (default pm_sampling_metrics.csv): --csv/-o <path>\n");
    printf("    Disable CSV export: --no-csv\n");
    printf("    Hold console after finish: --hold/-H [0|1], disable hold with --no-hold\n");
}

ParsedArgs parseArgs(int argc, char *argv[])
{
    ParsedArgs args;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--device" || arg == "-d")
        {
            args.deviceIndex = std::stoi(argv[++i]);
            args.isDeviceIndexSet = true;
        }
        else if (arg == "--samplingInterval" || arg == "-i")
        {
            args.samplingInterval = std::stoull(argv[++i]);
        }
        else if (arg == "--maxsamples" || arg == "-s")
        {
            args.maxSamples = std::stoull(argv[++i]);
        }
        else if (arg == "--durationSec" || arg == "-t")
        {
            args.durationSec = std::stoull(argv[++i]);
        }
        else if (arg == "--decodeIntervalMs" || arg == "-I")
        {
            args.decodeIntervalMs = std::stoull(argv[++i]);
        }
        else if (arg == "--hardwareBufferSize" || arg == "-b")
        {
            args.hardwareBufferSize = std::stoull(argv[++i]);
        }
        else if (arg == "--chip" || arg == "-c")
        {
            args.chipName = std::string(argv[++i]);
            args.isChipNameSet = true;
        }
        else if (arg == "--queryBaseMetrics" || arg == "-q")
        {
            args.queryBaseMetrics = 1;
        }
        else if (arg == "--queryMetricsProp" || arg == "-p")
        {
            args.queryMetricProperties = 1;
        }
        else if (arg == "--metrics" || arg == "-m")
        {
            std::stringstream ss(argv[++i]);
            std::string metric;
            args.metrics.clear();
            while (std::getline(ss, metric, ','))
            {
                args.metrics.push_back(strdup(metric.c_str()));
            }
        }
        else if (arg == "--livePrint" || arg == "-L")
        {
            args.livePrint = std::stoi(argv[++i]);
        }
        else if (arg == "--csv" || arg == "-o")
        {
            args.csvOutputPath = std::string(argv[++i]);
            args.writeCsv = 1;
        }
        else if (arg == "--no-csv")
        {
            args.writeCsv = 0;
        }
        else if (arg == "--hold" || arg == "-H")
        {
            if (i + 1 < argc && argv[i + 1][0] != '-')
            {
                args.holdOnExit = std::stoi(argv[++i]) != 0;
            }
            else
            {
                args.holdOnExit = 1;
            }
        }
        else if (arg == "--no-hold")
        {
            args.holdOnExit = 0;
        }
        else if (arg == "--help" || arg == "-h")
        {
            PrintHelp();
            exit(EXIT_SUCCESS);
        }
        else
        {
            fprintf(stderr, "Invalid argument: %s\n", arg.c_str());
            PrintHelp();
            exit(EXIT_FAILURE);
        }
    }
    return args;
}

void PmSamplingDeviceSupportStatus(CUdevice device)
{
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = device;
    params.api = CUPTI_PROFILER_PM_SAMPLING;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << device << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }

        exit(EXIT_WAIVED);
    }
}
