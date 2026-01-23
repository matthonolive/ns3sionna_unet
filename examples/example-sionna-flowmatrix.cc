/*
 * FlowMonitor matrix test with selectable propagation model + timing.
 *
 * Modes:
 *   --plModel=sionna : ns3sionna (requires Python server)
 *   --plModel=friis  : FriisPropagationLossModel (no server)
 *
 * Inputs:
 *   --environment=.../scene.xml         (used only for plModel=sionna)
 *   --placements=.../placements.csv     (tx,x,y,z then sta,x,y,z lines)
 *
 * Traffic:
 *   Creates one UDP flow per ordered pair i->j (i!=j) using OnOff -> PacketSink.
 *
 * Outputs:
 *   --outFlowCsv: per-flow (directed) aggregate stats
 *   --outPairCsv: per-unordered-pair aggregate stats (both directions combined)
 *
 * Timing outputs:
 *   --timingCsv: one-row CSV summary with wall-clock and (optional) benchmark stats
 *   --benchLinks: run a propagation micro-benchmark over all ordered pairs
 *   --benchRepeats: repeats for benchmark loop
 *   --benchOnly: if 1, skip traffic simulation and only benchmark propagation
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iomanip>
#include <cmath>
#include <cctype>
#include <chrono>

// ns3sionna models (only used if plModel=sionna)
#include "ns3/sionna-helper.h"
#include "ns3/sionna-propagation-cache.h"
#include "ns3/sionna-propagation-loss-model.h"
#include "ns3/sionna-propagation-delay-model.h"
#include "ns3/sionna-spectrum-propagation-loss-model.h"

// ns-3 modules
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/wifi-spectrum-phy-interface.h"
#include "ns3/wifi-module.h"

// Friis + delay (used if plModel=friis)
#include "ns3/propagation-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ExampleSionnaFlowMatrix");

static std::string Trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

struct PlacementRow
{
    std::string tag; // "tx" or "sta"
    Vector pos;
};

static std::vector<PlacementRow>
ReadPlacementsCsv(const std::string& path)
{
    std::ifstream f(path);
    NS_ABORT_MSG_IF(!f.is_open(), "Could not open placements file: " << path);

    std::vector<PlacementRow> rows;
    std::string line;
    while (std::getline(f, line))
    {
        line = Trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string tag, xs, ys, zs;
        if (!std::getline(ss, tag, ',')) continue;
        if (!std::getline(ss, xs, ',')) continue;
        if (!std::getline(ss, ys, ',')) continue;
        if (!std::getline(ss, zs, ',')) continue;

        tag = Trim(tag);
        for (auto& c : tag) c = std::tolower(static_cast<unsigned char>(c));
        if (tag == "ap") tag = "tx";
        if (tag == "rx") tag = "sta";

        double x = std::stod(Trim(xs));
        double y = std::stod(Trim(ys));
        double z = std::stod(Trim(zs));

        rows.push_back({tag, Vector(x, y, z)});
    }

    NS_ABORT_MSG_IF(rows.empty(), "Placements CSV is empty: " << path);

    uint32_t nTx = 0, nSta = 0;
    for (auto& r : rows)
    {
        if (r.tag == "tx") nTx++;
        if (r.tag == "sta") nSta++;
    }
    NS_ABORT_MSG_IF(nTx != 1, "Placements must contain exactly one tx row (found " << nTx << ")");
    NS_ABORT_MSG_IF(nSta < 1, "Placements must contain at least one sta row");

    return rows;
}

struct PairAgg
{
    uint64_t txPackets = 0, rxPackets = 0, lostPackets = 0;
    uint64_t txBytes = 0, rxBytes = 0;
    double delaySum_s = 0.0;
    double jitterSum_s = 0.0;
    uint64_t rxForDelay = 0;
    uint64_t rxForJitter = 0;
};

static void
WriteFlowMonitorCsv(Ptr<FlowMonitor> monitor,
                    FlowMonitorHelper& helper,
                    const std::unordered_map<uint32_t, uint32_t>& ipToIndex,
                    const std::vector<std::string>& labels,
                    uint16_t portMin,
                    uint16_t portMax,
                    const std::string& outFlowCsv,
                    const std::string& outPairCsv)
{
    monitor->CheckForLostPackets();

    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(helper.GetClassifier());

    std::ofstream flowOut(outFlowCsv);
    NS_ABORT_MSG_IF(!flowOut.is_open(), "Could not open outFlowCsv: " << outFlowCsv);

    flowOut << "flowId,src_i,dst_j,label_i,label_j,src_ip,dst_ip,src_port,dst_port,"
            << "txPackets,rxPackets,lostPackets,txBytes,rxBytes,"
            << "throughput_Mbps,meanDelay_ms,meanJitter_ms\n";

    std::map<std::pair<uint32_t, uint32_t>, PairAgg> pairAgg;

    auto stats = monitor->GetFlowStats();
    for (const auto& kv : stats)
    {
        FlowId flowId = kv.first;
        const FlowMonitor::FlowStats& st = kv.second;
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);

        // Only keep our synthetic UDP flows by destination port range
        if (t.destinationPort < portMin || t.destinationPort > portMax)
            continue;

        auto itS = ipToIndex.find(t.sourceAddress.Get());
        auto itD = ipToIndex.find(t.destinationAddress.Get());
        if (itS == ipToIndex.end() || itD == ipToIndex.end())
            continue;

        uint32_t i = itS->second;
        uint32_t j = itD->second;

        double duration_s = (st.timeLastRxPacket - st.timeFirstTxPacket).GetSeconds();
        if (duration_s <= 0.0) duration_s = 0.0;

        double throughput_Mbps = 0.0;
        if (duration_s > 0.0)
            throughput_Mbps = (st.rxBytes * 8.0) / duration_s / 1e6;

        double meanDelay_ms = 0.0;
        if (st.rxPackets > 0)
            meanDelay_ms = (st.delaySum.GetSeconds() / st.rxPackets) * 1e3;

        // Common mean jitter convention: jitterSum/(rxPackets-1) if rxPackets>1
        double meanJitter_ms = 0.0;
        if (st.rxPackets > 1)
            meanJitter_ms = (st.jitterSum.GetSeconds() / (st.rxPackets - 1)) * 1e3;

        flowOut << flowId << ","
                << i << "," << j << ","
                << labels[i] << "," << labels[j] << ","
                << t.sourceAddress << "," << t.destinationAddress << ","
                << t.sourcePort << "," << t.destinationPort << ","
                << st.txPackets << "," << st.rxPackets << "," << st.lostPackets << ","
                << st.txBytes << "," << st.rxBytes << ","
                << std::fixed << std::setprecision(6)
                << throughput_Mbps << ","
                << meanDelay_ms << ","
                << meanJitter_ms << "\n";

        // Unordered pair aggregate
        uint32_t a = std::min(i, j);
        uint32_t b = std::max(i, j);
        auto& pa = pairAgg[{a, b}];

        pa.txPackets += st.txPackets;
        pa.rxPackets += st.rxPackets;
        pa.lostPackets += st.lostPackets;
        pa.txBytes += st.txBytes;
        pa.rxBytes += st.rxBytes;

        if (st.rxPackets > 0)
        {
            pa.delaySum_s += st.delaySum.GetSeconds();
            pa.rxForDelay += st.rxPackets;
        }
        if (st.rxPackets > 1)
        {
            pa.jitterSum_s += st.jitterSum.GetSeconds();
            pa.rxForJitter += (st.rxPackets - 1);
        }
    }

    flowOut.close();

    std::ofstream pairOut(outPairCsv);
    NS_ABORT_MSG_IF(!pairOut.is_open(), "Could not open outPairCsv: " << outPairCsv);

    pairOut << "i,j,label_i,label_j,txPackets,rxPackets,lostPackets,txBytes,rxBytes,"
            << "meanDelay_ms,meanJitter_ms\n";

    for (const auto& kv : pairAgg)
    {
        uint32_t i = kv.first.first;
        uint32_t j = kv.first.second;
        const PairAgg& pa = kv.second;

        double meanDelay_ms = 0.0;
        if (pa.rxForDelay > 0)
            meanDelay_ms = (pa.delaySum_s / pa.rxForDelay) * 1e3;

        double meanJitter_ms = 0.0;
        if (pa.rxForJitter > 0)
            meanJitter_ms = (pa.jitterSum_s / pa.rxForJitter) * 1e3;

        pairOut << i << "," << j << ","
                << labels[i] << "," << labels[j] << ","
                << pa.txPackets << "," << pa.rxPackets << "," << pa.lostPackets << ","
                << pa.txBytes << "," << pa.rxBytes << ","
                << std::fixed << std::setprecision(6)
                << meanDelay_ms << ","
                << meanJitter_ms << "\n";
    }

    pairOut.close();
}

struct BenchResult
{
    uint64_t calls = 0;
    double total_s = 0.0;
    double avg_us = 0.0;
};

// Micro-benchmark propagation calls over all ordered pairs i->j (i!=j).
// Times (CalcRxPower + GetDelay) together per call.
static BenchResult
BenchmarkLinks(uint32_t N,
               const NodeContainer& nodes,
               Ptr<PropagationLossModel> loss,
               Ptr<PropagationDelayModel> delay,
               double txPowerDbm,
               uint32_t repeats)
{
    using clock = std::chrono::steady_clock;

    uint64_t calls = 0;
    auto t0 = clock::now();

    volatile double sink = 0.0; // prevent compiler eliminating work

    for (uint32_t r = 0; r < repeats; ++r)
    {
        for (uint32_t i = 0; i < N; ++i)
        {
            Ptr<MobilityModel> mi = nodes.Get(i)->GetObject<MobilityModel>();
            for (uint32_t j = 0; j < N; ++j)
            {
                if (i == j) continue;
                Ptr<MobilityModel> mj = nodes.Get(j)->GetObject<MobilityModel>();

                double rxDbm = loss->CalcRxPower(txPowerDbm, mi, mj);
                Time d = delay->GetDelay(mi, mj);

                sink += rxDbm + d.GetSeconds();
                calls++;
            }
        }
    }

    auto t1 = clock::now();
    (void)sink;

    double total_s = std::chrono::duration<double>(t1 - t0).count();
    double avg_us = (calls > 0) ? (total_s * 1e6 / calls) : 0.0;

    return {calls, total_s, avg_us};
}

static void
AppendTimingCsv(const std::string& timingCsv,
                bool fileExists,
                const std::string& plModel,
                uint32_t N,
                bool enableTraffic,
                double simTime,
                double wall_setup_s,
                double wall_run_s,
                double wall_total_s,
                bool benchLinks,
                uint32_t benchRepeats,
                const BenchResult& bench)
{
    std::ofstream f;
    f.open(timingCsv, std::ios::app);
    NS_ABORT_MSG_IF(!f.is_open(), "Could not open timingCsv: " << timingCsv);

    if (!fileExists)
    {
        f << "plModel,N,numOrderedPairs,enableTraffic,simTime_s,"
          << "wall_setup_s,wall_run_s,wall_total_s,"
          << "benchLinks,benchRepeats,benchCalls,benchTotal_s,benchAvg_us_per_call\n";
    }

    uint32_t numOrderedPairs = N * (N - 1);

    f << plModel << ","
      << N << ","
      << numOrderedPairs << ","
      << (enableTraffic ? 1 : 0) << ","
      << std::fixed << std::setprecision(6)
      << simTime << ","
      << wall_setup_s << ","
      << wall_run_s << ","
      << wall_total_s << ","
      << (benchLinks ? 1 : 0) << ","
      << benchRepeats << ","
      << bench.calls << ","
      << bench.total_s << ","
      << bench.avg_us
      << "\n";

    f.close();
}

int
main(int argc, char* argv[])
{
    using clock = std::chrono::steady_clock;

    auto T_total0 = clock::now();

    // Inputs
    std::string plModel = "sionna"; // "sionna" or "friis"
    std::string environment = "seed0000/scene.xml"; // only used for sionna
    std::string placementsPath = "seed0000/placements.csv";
    std::string serverAddr = "tcp://localhost:5555";

    // Wi-Fi
    int wifi_channel_num = 42; // ~5210 MHz
    int channelWidth = 80;     // MHz
    double txPowerDbm = 20.0;
    bool caching = true;       // sionna cache only
    int heMcs = 7;             // ConstantRate HeMcs for repeatability

    // Traffic
    bool enableTraffic = true;
    double simTime = 6.0;
    double trafficStart = 1.5;
    double trafficStopMargin = 0.5;
    uint32_t pktSize = 512;
    double appRateMbps = 0.1;
    double stagger = 0.01;
    uint16_t basePort = 9000;

    // Outputs
    std::string outFlowCsv = "flow_stats.csv";
    std::string outPairCsv = "pair_stats.csv";

    // Timing outputs / benchmark
    std::string timingCsv = "timing_summary.csv";
    bool benchLinks = false;
    uint32_t benchRepeats = 1;
    bool benchOnly = false;

    bool verbose = true;

    CommandLine cmd(__FILE__);
    cmd.AddValue("plModel", "Propagation model: sionna|friis", plModel);
    cmd.AddValue("environment", "scene.xml path (sionna only)", environment);
    cmd.AddValue("placements", "placements.csv path", placementsPath);
    cmd.AddValue("server", "Python server addr (sionna only)", serverAddr);
    cmd.AddValue("channel", "WiFi channel number", wifi_channel_num);
    cmd.AddValue("channelWidth", "WiFi channel width (MHz)", channelWidth);
    cmd.AddValue("txPowerDbm", "TX power (dBm)", txPowerDbm);
    cmd.AddValue("caching", "Enable SionnaPropagationCache caching (sionna only)", caching);
    cmd.AddValue("heMcs", "ConstantRate HeMcs index", heMcs);
    cmd.AddValue("enableTraffic", "Enable UDP flows for all ordered pairs", enableTraffic);
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("trafficStart", "Traffic start time (s)", trafficStart);
    cmd.AddValue("pktSize", "UDP packet size (bytes)", pktSize);
    cmd.AddValue("appRateMbps", "Per-flow offered rate (Mbps)", appRateMbps);
    cmd.AddValue("stagger", "Flow start staggering (s)", stagger);
    cmd.AddValue("basePort", "Base UDP port", basePort);
    cmd.AddValue("outFlowCsv", "Output per-flow CSV", outFlowCsv);
    cmd.AddValue("outPairCsv", "Output per-pair CSV", outPairCsv);

    cmd.AddValue("timingCsv", "Append timing summary row to this CSV", timingCsv);
    cmd.AddValue("benchLinks", "Benchmark propagation calls over all ordered pairs (0/1)", benchLinks);
    cmd.AddValue("benchRepeats", "Repeats for benchmark loop", benchRepeats);
    cmd.AddValue("benchOnly", "If 1, skip traffic sim and only benchmark propagation", benchOnly);

    cmd.AddValue("verbose", "Enable logs", verbose);
    cmd.Parse(argc, argv);

    for (auto& c : plModel) c = std::tolower(static_cast<unsigned char>(c));
    bool useSionna = (plModel == "sionna");
    NS_ABORT_MSG_IF(!useSionna && plModel != "friis", "plModel must be 'sionna' or 'friis'");

    if (verbose)
    {
        LogComponentEnable("ExampleSionnaFlowMatrix", LOG_INFO);
        if (useSionna)
        {
            LogComponentEnable("SionnaPropagationCache", LOG_INFO);
            LogComponentEnable("SionnaPropagationLossModel", LOG_INFO);
            LogComponentEnable("SionnaSpectrumPropagationLossModel", LOG_INFO);
            LogComponentEnable("SionnaPropagationDelayModel", LOG_INFO);
        }
    }

    // Read placements; reorder to [TX, STA0..]
    auto rows = ReadPlacementsCsv(placementsPath);
    Vector txPos;
    std::vector<Vector> staPos;
    for (auto& r : rows)
    {
        if (r.tag == "tx") txPos = r.pos;
        if (r.tag == "sta") staPos.push_back(r.pos);
    }

    uint32_t N = 1 + staPos.size();
    NS_ABORT_MSG_IF(N < 2, "Need at least 2 nodes");

    NodeContainer nodes;
    nodes.Create(N);

    std::vector<std::string> labels;
    labels.reserve(N);
    labels.push_back("TX");
    for (uint32_t k = 0; k < staPos.size(); ++k)
        labels.push_back("STA" + std::to_string(k));

    // Channel
    Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();

    // Prop models as base pointers (for benchmarking)
    Ptr<PropagationLossModel> lossBase;
    Ptr<PropagationDelayModel> delayBase;

    // Optional ns3sionna objects
    std::unique_ptr<SionnaHelper> sionnaHelper;
    Ptr<SionnaPropagationCache> propagationCache;
    Ptr<SionnaPropagationLossModel> sionnaLoss;
    Ptr<SionnaSpectrumPropagationLossModel> sionnaSpec;
    Ptr<SionnaPropagationDelayModel> sionnaDelay;

    // Optional Friis objects
    Ptr<FriisPropagationLossModel> friisLoss;
    Ptr<ConstantSpeedPropagationDelayModel> constDelay;

    if (useSionna)
    {
        sionnaHelper = std::make_unique<SionnaHelper>(environment, serverAddr);

        propagationCache = CreateObject<SionnaPropagationCache>();
        propagationCache->SetSionnaHelper(*sionnaHelper);
        propagationCache->SetCaching(caching);

        sionnaLoss = CreateObject<SionnaPropagationLossModel>();
        sionnaLoss->SetPropagationCache(propagationCache);
        spectrumChannel->AddPropagationLossModel(sionnaLoss);

        sionnaSpec = CreateObject<SionnaSpectrumPropagationLossModel>();
        sionnaSpec->SetPropagationCache(propagationCache);
        spectrumChannel->AddSpectrumPropagationLossModel(sionnaSpec);

        sionnaDelay = CreateObject<SionnaPropagationDelayModel>();
        sionnaDelay->SetPropagationCache(propagationCache);
        spectrumChannel->SetPropagationDelayModel(sionnaDelay);

        lossBase = sionnaLoss;
        delayBase = sionnaDelay;
    }
    else
    {
        friisLoss = CreateObject<FriisPropagationLossModel>();
        spectrumChannel->AddPropagationLossModel(friisLoss);

        constDelay = CreateObject<ConstantSpeedPropagationDelayModel>();
        spectrumChannel->SetPropagationDelayModel(constDelay);

        lossBase = friisLoss;
        delayBase = constDelay;
    }

    // Wi-Fi (Adhoc) over Spectrum channel
    Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelWidth",
                UintegerValue(channelWidth));

    SpectrumWifiPhyHelper spectrumPhy;
    spectrumPhy.SetChannel(spectrumChannel);
    spectrumPhy.SetErrorRateModel("ns3::NistErrorRateModel");
    spectrumPhy.Set("TxPowerStart", DoubleValue(txPowerDbm));
    spectrumPhy.Set("TxPowerEnd", DoubleValue(txPowerDbm));

    WifiHelper wifi;
    WifiStandard wifi_standard = WIFI_STANDARD_80211ax;
    wifi.SetStandard(wifi_standard);

    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("HeMcs" + std::to_string(heMcs)),
                                 "ControlMode", StringValue("HeMcs0"));

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    std::string channelStr =
        "{" + std::to_string(wifi_channel_num) + ", " + std::to_string(channelWidth) + ", BAND_5GHZ, 0}";
    spectrumPhy.Set("ChannelSettings", StringValue(channelStr));

    NetDeviceContainer dev = wifi.Install(spectrumPhy, mac, nodes);

    // Mobility
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::SionnaMobilityModel");
    mobility.Install(nodes);

    nodes.Get(0)->GetObject<MobilityModel>()->SetPosition(txPos);
    for (uint32_t k = 0; k < staPos.size(); ++k)
        nodes.Get(1 + k)->GetObject<MobilityModel>()->SetPosition(staPos[k]);

    // Internet stack + IPs
    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ifs = address.Assign(dev);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    std::unordered_map<uint32_t, uint32_t> ipToIndex;
    for (uint32_t i = 0; i < N; ++i)
        ipToIndex[ifs.GetAddress(i).Get()] = i;

    // Configure frequency-dependent pieces
    double fc = get_center_freq(dev.Get(0));
    if (useSionna)
    {
        sionnaHelper->Configure(fc,
                                channelWidth,
                                getFFTSize(wifi_standard, channelWidth),
                                getSubcarrierSpacing(wifi_standard));
        sionnaHelper->SetMode(SionnaHelper::MODE_P2P);
    }
    else
    {
        friisLoss->SetAttribute("Frequency", DoubleValue(fc));
    }

    // (Optional) propagation micro-benchmark
    BenchResult bench{0, 0.0, 0.0};
    if (benchLinks)
    {
        if (useSionna)
        {
            // Important: ensure helper is started so benchmark can trigger server requests
            sionnaHelper->Start();
        }

        bench = BenchmarkLinks(N, nodes, lossBase, delayBase, txPowerDbm, benchRepeats);

    }

    auto T_setup1 = clock::now();

    // If we only wanted benchmarking, skip simulation+flowmonitor
    Ptr<FlowMonitor> monitor;
    FlowMonitorHelper flowmonHelper;

    if (!benchOnly)
    {
        // Ports for all ordered flows i->j: basePort + i*N + j
        uint32_t maxFlows = N * (N - 1);
        NS_ABORT_MSG_IF(basePort + maxFlows >= 65535, "basePort too high for N");

        uint16_t portMin = basePort;
        uint16_t portMax = basePort + static_cast<uint16_t>(N * N); // safe upper bound

        if (enableTraffic)
        {
            // Install sinks for every ordered pair (i->j) at destination j
            for (uint32_t i = 0; i < N; ++i)
            {
                for (uint32_t j = 0; j < N; ++j)
                {
                    if (i == j) continue;
                    uint16_t port = basePort + static_cast<uint16_t>(i * N + j);

                    PacketSinkHelper sink("ns3::UdpSocketFactory",
                                          InetSocketAddress(Ipv4Address::GetAny(), port));
                    auto apps = sink.Install(nodes.Get(j));
                    apps.Start(Seconds(0.5));
                    apps.Stop(Seconds(simTime));
                }
            }

            // Install OnOff sources
            uint32_t k = 0;
            double stopTime = std::max(trafficStart + 0.1, simTime - trafficStopMargin);

            for (uint32_t i = 0; i < N; ++i)
            {
                for (uint32_t j = 0; j < N; ++j)
                {
                    if (i == j) continue;

                    uint16_t port = basePort + static_cast<uint16_t>(i * N + j);

                    OnOffHelper onoff("ns3::UdpSocketFactory",
                                      InetSocketAddress(ifs.GetAddress(j), port));
                    onoff.SetAttribute("PacketSize", UintegerValue(pktSize));
                    onoff.SetAttribute("DataRate",
                                       DataRateValue(DataRate(static_cast<uint64_t>(appRateMbps * 1e6))));
                    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
                    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));

                    auto apps = onoff.Install(nodes.Get(i));
                    apps.Start(Seconds(trafficStart + stagger * k));
                    apps.Stop(Seconds(stopTime));
                    k++;
                }
            }
        }

        // FlowMonitor
        monitor = flowmonHelper.InstallAll();

        Simulator::Stop(Seconds(simTime));

        // Start ns3sionna helper (connects to Python server) if needed and not already started
        if (useSionna && !(benchLinks))  // if benchLinks we already started above
            sionnaHelper->Start();

        auto T_run0 = clock::now();
        Simulator::Run();
        auto T_run1 = clock::now();

        // Export FlowMonitor metrics
        auto T_export0 = clock::now();
        WriteFlowMonitorCsv(monitor,
                            flowmonHelper,
                            ipToIndex,
                            labels,
                            portMin,
                            portMax,
                            outFlowCsv,
                            outPairCsv);
        auto T_export1 = clock::now();

        double wall_export_s = std::chrono::duration<double>(T_export1 - T_export0).count();
        NS_LOG_UNCOND("Export time: " << wall_export_s << " s");


        Simulator::Destroy();

        // Timing summary
        double wall_setup_s = std::chrono::duration<double>(T_setup1 - T_total0).count();
        double wall_run_s = std::chrono::duration<double>(T_run1 - T_run0).count();
        double wall_total_s = std::chrono::duration<double>(clock::now() - T_total0).count();

        // Append timing CSV
        {
            std::ifstream test(timingCsv);
            bool exists = test.good();
            AppendTimingCsv(timingCsv, exists,
                            plModel, N, enableTraffic, simTime,
                            wall_setup_s, wall_run_s, wall_total_s,
                            benchLinks, benchRepeats, bench);
        }

        if (useSionna)
        {
            if (propagationCache) propagationCache->PrintStats();
            sionnaHelper->Destroy();
        }
    }
    else
    {
        // benchOnly timing summary (no simulation)
        double wall_setup_s = std::chrono::duration<double>(T_setup1 - T_total0).count();
        double wall_run_s = 0.0;
        double wall_total_s = std::chrono::duration<double>(clock::now() - T_total0).count();

        std::ifstream test(timingCsv);
        bool exists = test.good();
        AppendTimingCsv(timingCsv, exists,
                        plModel, N, false, 0.0,
                        wall_setup_s, wall_run_s, wall_total_s,
                        benchLinks, benchRepeats, bench);

        if (useSionna)
        {
            if (propagationCache) propagationCache->PrintStats();
            sionnaHelper->Destroy();
        }
    }

    return 0;
}
