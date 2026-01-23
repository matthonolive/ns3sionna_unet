/*
 * FlowMonitor matrix test for ns3sionna
 *
 * - environment: scene.xml
 * - placements: placements.csv with lines:
 *      tx,x,y,z
 *      sta,x,y,z
 *      sta,x,y,z
 *
 * Creates N nodes and runs UDP flows for every ordered pair i->j (i!=j),
 * then exports FlowMonitor aggregate stats:
 *   tx/rx/lost packets, mean delay, mean jitter, throughput.
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

// Sionna models
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
    // aggregate both directions
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

    // aggregate per unordered pair (min,max)
    std::map<std::pair<uint32_t,uint32_t>, PairAgg> pairAgg;

    auto stats = monitor->GetFlowStats();
    for (const auto& kv : stats)
    {
        FlowId flowId = kv.first;
        const FlowMonitor::FlowStats& st = kv.second;

        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);

        // Filter to only our UDP flows by dst port range
        if (t.destinationPort < portMin || t.destinationPort > portMax)
        {
            continue;
        }

        auto itS = ipToIndex.find(t.sourceAddress.Get());
        auto itD = ipToIndex.find(t.destinationAddress.Get());
        if (itS == ipToIndex.end() || itD == ipToIndex.end())
        {
            continue;
        }

        uint32_t i = itS->second;
        uint32_t j = itD->second;

        double duration_s = (st.timeLastRxPacket - st.timeFirstTxPacket).GetSeconds();
        if (duration_s <= 0.0) duration_s = 0.0;

        double throughput_Mbps = 0.0;
        if (duration_s > 0.0)
        {
            throughput_Mbps = (st.rxBytes * 8.0) / duration_s / 1e6;
        }

        double meanDelay_ms = 0.0;
        if (st.rxPackets > 0)
        {
            meanDelay_ms = (st.delaySum.GetSeconds() / st.rxPackets) * 1e3;
        }

        // FlowMonitor jitterSum is accumulated over received packets; a common "mean jitter"
        // is jitterSum/(rxPackets-1) when rxPackets>1.
        double meanJitter_ms = 0.0;
        if (st.rxPackets > 1)
        {
            meanJitter_ms = (st.jitterSum.GetSeconds() / (st.rxPackets - 1)) * 1e3;
        }

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

        // Pair aggregate (unordered)
        uint32_t a = std::min(i, j);
        uint32_t b = std::max(i, j);
        auto& pa = pairAgg[{a,b}];

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
        {
            meanDelay_ms = (pa.delaySum_s / pa.rxForDelay) * 1e3;
        }

        double meanJitter_ms = 0.0;
        if (pa.rxForJitter > 0)
        {
            meanJitter_ms = (pa.jitterSum_s / pa.rxForJitter) * 1e3;
        }

        pairOut << i << "," << j << ","
                << labels[i] << "," << labels[j] << ","
                << pa.txPackets << "," << pa.rxPackets << "," << pa.lostPackets << ","
                << pa.txBytes << "," << pa.rxBytes << ","
                << std::fixed << std::setprecision(6)
                << meanDelay_ms << ","
                << meanJitter_ms << "\n";
    }

    pairOut.close();

    NS_LOG_UNCOND("Wrote FlowMonitor CSV: " << outFlowCsv);
    NS_LOG_UNCOND("Wrote Pair CSV:       " << outPairCsv);
}

int
main(int argc, char *argv[])
{
    // Inputs
    std::string environment = "seed0000/scene.xml";
    std::string placementsPath = "seed0000/placements.csv";
    std::string serverAddr = "tcp://localhost:5555";

    // Wi-Fi
    int wifi_channel_num = 42;
    int channelWidth = 80; // MHz
    double txPowerDbm = 20.0;
    bool caching = true;   // caching in SionnaPropagationCache

    // Rate control (set constant so results are repeatable)
    int heMcs = 7;

    // App traffic
    bool enableTraffic = true;
    double simTime = 6.0;
    double trafficStart = 1.5;
    double trafficStopMargin = 0.5; // stop flows at simTime - margin
    uint32_t pktSize = 512;
    double appRateMbps = 0.1;       // per-flow offered rate
    double stagger = 0.01;          // stagger flow start times to reduce burstiness
    uint16_t basePort = 9000;

    // Outputs
    std::string outFlowCsv = "flow_stats.csv";
    std::string outPairCsv = "pair_stats.csv";

    bool verbose = true;

    CommandLine cmd(__FILE__);
    cmd.AddValue("environment", "scene.xml path", environment);
    cmd.AddValue("placements", "placements.csv path", placementsPath);
    cmd.AddValue("server", "Python server address", serverAddr);
    cmd.AddValue("channel", "WiFi channel number", wifi_channel_num);
    cmd.AddValue("channelWidth", "WiFi channel width (MHz)", channelWidth);
    cmd.AddValue("txPowerDbm", "TX power (dBm)", txPowerDbm);
    cmd.AddValue("caching", "Enable caching in SionnaPropagationCache", caching);
    cmd.AddValue("heMcs", "ConstantRate HeMcs index (0..11 typical)", heMcs);
    cmd.AddValue("enableTraffic", "Enable UDP flows for every ordered pair", enableTraffic);
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("trafficStart", "Traffic start time (s)", trafficStart);
    cmd.AddValue("pktSize", "UDP packet size (bytes)", pktSize);
    cmd.AddValue("appRateMbps", "Per-flow offered rate (Mbps)", appRateMbps);
    cmd.AddValue("stagger", "Stagger flow start times (s)", stagger);
    cmd.AddValue("basePort", "Base UDP port", basePort);
    cmd.AddValue("outFlowCsv", "Output per-flow CSV", outFlowCsv);
    cmd.AddValue("outPairCsv", "Output per-pair CSV", outPairCsv);
    cmd.AddValue("verbose", "Enable logs", verbose);
    cmd.Parse(argc, argv);

    if (verbose)
    {
        LogComponentEnable("ExampleSionnaFlowMatrix", LOG_INFO);
        LogComponentEnable("SionnaPropagationCache", LOG_INFO);
        LogComponentEnable("SionnaPropagationLossModel", LOG_INFO);
        LogComponentEnable("SionnaSpectrumPropagationLossModel", LOG_INFO);
        LogComponentEnable("SionnaPropagationDelayModel", LOG_INFO);
    }

    std::cout << "ns3sionna FlowMonitor matrix test\n";
    std::cout << "Env: " << environment << "\n";
    std::cout << "Placements: " << placementsPath << "\n";

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
    {
        labels.push_back("STA" + std::to_string(k));
    }

    // Sionna helper
    SionnaHelper sionnaHelper(environment, serverAddr);

    // Cache + channel models
    Ptr<SionnaPropagationCache> propagationCache = CreateObject<SionnaPropagationCache>();
    propagationCache->SetSionnaHelper(sionnaHelper);
    propagationCache->SetCaching(caching);

    Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();

    Ptr<SionnaPropagationLossModel> lossModel = CreateObject<SionnaPropagationLossModel>();
    lossModel->SetPropagationCache(propagationCache);
    spectrumChannel->AddPropagationLossModel(lossModel);

    Ptr<SionnaSpectrumPropagationLossModel> spectrumLossModel = CreateObject<SionnaSpectrumPropagationLossModel>();
    spectrumLossModel->SetPropagationCache(propagationCache);
    spectrumChannel->AddSpectrumPropagationLossModel(spectrumLossModel);

    Ptr<SionnaPropagationDelayModel> delayModel = CreateObject<SionnaPropagationDelayModel>();
    delayModel->SetPropagationCache(propagationCache);
    spectrumChannel->SetPropagationDelayModel(delayModel);

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

    // Constant MCS for repeatability
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
    {
        nodes.Get(1 + k)->GetObject<MobilityModel>()->SetPosition(staPos[k]);
    }

    // Internet stack + IPs
    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ifs = address.Assign(dev);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Map IP->node index for later CSV labeling
    std::unordered_map<uint32_t, uint32_t> ipToIndex;
    for (uint32_t i = 0; i < N; ++i)
    {
        ipToIndex[ifs.GetAddress(i).Get()] = i;
    }

    // Applications: PacketSink on each node for each incoming (i->j) flow
    // We use unique dst ports per ordered pair to make filtering/mapping easy.
    uint32_t maxFlows = N * (N - 1);
    NS_ABORT_MSG_IF(basePort + maxFlows >= 65535, "basePort too high for N");

    uint16_t portMin = basePort;
    uint16_t portMax = basePort + static_cast<uint16_t>(maxFlows);

    if (enableTraffic)
    {
        // Install sinks
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
                onoff.SetAttribute("DataRate", DataRateValue(DataRate(static_cast<uint64_t>(appRateMbps * 1e6))));
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
    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();

    // Configure Sionna helper OFDM params (needed when spectrum loss model is in play)
    double fc = get_center_freq(dev.Get(0));
    sionnaHelper.Configure(fc,
                           channelWidth,
                           getFFTSize(wifi_standard, channelWidth),
                           getSubcarrierSpacing(wifi_standard));
    sionnaHelper.SetMode(SionnaHelper::MODE_P2P);

    Simulator::Stop(Seconds(simTime));

    // Start ns3sionna helper (connects to Python server)
    sionnaHelper.Start();

    Simulator::Run();

    // Export FlowMonitor metrics (after sim)
    WriteFlowMonitorCsv(monitor,
                        flowmonHelper,
                        ipToIndex,
                        labels,
                        portMin,
                        portMax,
                        outFlowCsv,
                        outPairCsv);

    Simulator::Destroy();

    propagationCache->PrintStats();
    sionnaHelper.Destroy();

    return 0;
}
