/*
 * Pairwise reciprocity test for ns3sionna (wideband path loss)
 *
 * - Loads a Sionna scene XML (environment)
 * - Loads placements.csv:
 *      tx,x,y,z
 *      sta,x,y,z
 *      sta,x,y,z
 *      ...
 * - Creates one ns-3 node per row
 * - Uses Adhoc Wi-Fi so every node pair is a direct link
 * - Computes path loss both ways for every pair using:
 *      PL(i->j) = Ptx - CalcRxPower(Ptx, i, j)
 * - Optionally sends bidirectional UDP Echo traffic for every pair
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <cmath>

// ns3 + sionna models
#include "ns3/sionna-helper.h"
#include "ns3/sionna-propagation-cache.h"
#include "ns3/sionna-propagation-loss-model.h"
#include "ns3/sionna-propagation-delay-model.h"

// ns-3 modules
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/wifi-spectrum-phy-interface.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ExampleSionnaReciprocityMatrix");

// Optional: map IPv4->NodeId for nicer logging
static std::map<Ipv4Address, uint32_t> g_ipToNodeId;

static void
BuildIpToNodeIdMap()
{
    g_ipToNodeId.clear();
    for (uint32_t i = 0; i < NodeList::GetNNodes(); ++i)
    {
        Ptr<Node> node = NodeList::GetNode(i);
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        if (!ipv4) continue;

        for (uint32_t j = 0; j < ipv4->GetNInterfaces(); ++j)
        {
            for (uint32_t k = 0; k < ipv4->GetNAddresses(j); ++k)
            {
                Ipv4Address addr = ipv4->GetAddress(j, k).GetLocal();
                if (!addr.IsLocalhost())
                {
                    g_ipToNodeId[addr] = node->GetId();
                }
            }
        }
    }
}

static uint32_t
NodeIdFromIpv4(Ipv4Address a)
{
    auto it = g_ipToNodeId.find(a);
    if (it == g_ipToNodeId.end()) return 0xFFFFFFFF;
    return it->second;
}

// Simple RX trace (just confirms packets are flowing both ways)
static void
RxTraceWithAddresses(std::string context,
                     Ptr<const Packet> packet,
                     const Address &from,
                     const Address &to)
{
    InetSocketAddress src = InetSocketAddress::ConvertFrom(from);
    InetSocketAddress dst = InetSocketAddress::ConvertFrom(to);

    uint32_t srcId = NodeIdFromIpv4(src.GetIpv4());
    uint32_t dstId = NodeIdFromIpv4(dst.GetIpv4());

    NS_LOG_INFO(Simulator::Now().GetSeconds()
                << "s: RX " << packet->GetSize() << "B  "
                << src.GetIpv4() << "(" << srcId << "):" << src.GetPort()
                << " -> "
                << dst.GetIpv4() << "(" << dstId << "):" << dst.GetPort()
                << "  [" << context << "]");
}

struct PlacementRow
{
    std::string tag; // "tx" or "sta" (also accepts "ap"/"rx")
    Vector pos;
};

static std::string Trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

static std::vector<PlacementRow>
ReadPlacementsCsv(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        NS_FATAL_ERROR("Could not open placements file: " << path);
    }

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
        double x = std::stod(Trim(xs));
        double y = std::stod(Trim(ys));
        double z = std::stod(Trim(zs));

        // normalize tags
        std::string t = tag;
        for (auto &c : t) c = std::tolower(c);
        if (t == "ap") t = "tx";
        if (t == "rx") t = "sta";

        rows.push_back({t, Vector(x, y, z)});
    }

    NS_ABORT_MSG_IF(rows.empty(), "Placements CSV is empty: " << path);

    // Must contain one tx and at least one sta
    uint32_t nTx = 0, nSta = 0;
    for (auto &r : rows)
    {
        if (r.tag == "tx") nTx++;
        if (r.tag == "sta") nSta++;
    }
    NS_ABORT_MSG_IF(nTx != 1, "Placements CSV must contain exactly one 'tx' row (found " << nTx << ")");
    NS_ABORT_MSG_IF(nSta < 1, "Placements CSV must contain at least one 'sta' row");

    return rows;
}

static void
ComputeAndLogPairwisePlReciprocity(Ptr<SionnaPropagationLossModel> lossModel,
                                  const NodeContainer& nodes,
                                  const std::vector<std::string>& labels,
                                  double txPowerDbm,
                                  const std::string& outCsv)
{
    std::ofstream out;
    if (!outCsv.empty())
    {
        out.open(outCsv);
        NS_ABORT_MSG_IF(!out.is_open(), "Could not open outCsv: " << outCsv);
        out << "i,j,label_i,label_j,pl_i_to_j_db,pl_j_to_i_db,diff_db,abs_diff_db\n";
    }

    double sumAbs = 0.0, sumSq = 0.0, maxAbs = 0.0;
    uint32_t cnt = 0;

    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        for (uint32_t j = i + 1; j < nodes.GetN(); ++j)
        {
            Ptr<MobilityModel> mi = nodes.Get(i)->GetObject<MobilityModel>();
            Ptr<MobilityModel> mj = nodes.Get(j)->GetObject<MobilityModel>();

            double rx_ij = lossModel->CalcRxPower(txPowerDbm, mi, mj);
            double rx_ji = lossModel->CalcRxPower(txPowerDbm, mj, mi);

            double pl_ij = txPowerDbm - rx_ij;
            double pl_ji = txPowerDbm - rx_ji;

            double diff = pl_ij - pl_ji;
            double ad = std::abs(diff);

            sumAbs += ad;
            sumSq  += diff * diff;
            maxAbs = std::max(maxAbs, ad);
            cnt++;

            NS_LOG_UNCOND(std::fixed << std::setprecision(3)
                         << "[PL reciprocity] (" << i << ":" << labels[i]
                         << " <-> " << j << ":" << labels[j] << ")  "
                         << "PL(i->j)=" << pl_ij << " dB, "
                         << "PL(j->i)=" << pl_ji << " dB, "
                         << "diff=" << diff << " dB");

            if (out.is_open())
            {
                out << i << "," << j << ","
                    << labels[i] << "," << labels[j] << ","
                    << pl_ij << "," << pl_ji << ","
                    << diff << "," << ad << "\n";
            }
        }
    }

    if (cnt > 0)
    {
        double meanAbs = sumAbs / cnt;
        double rmse = std::sqrt(sumSq / cnt);
        NS_LOG_UNCOND(std::fixed << std::setprecision(4)
                     << "[PL reciprocity summary] pairs=" << cnt
                     << "  mean|diff|=" << meanAbs << " dB"
                     << "  rmse(diff)=" << rmse << " dB"
                     << "  max|diff|=" << maxAbs << " dB");
    }
}

int
main(int argc, char *argv[])
{
    bool verbose = true;
    bool caching = false;      // recommended false for reciprocity measurement
    bool enableTraffic = true; // send UDP both ways per pair
    std::string environment = "seed0000/scene.xml";
    std::string placementsPath = "seed0000/placements.csv";
    std::string outCsv = "reciprocity_pl.csv";
    std::string serverAddr = "tcp://localhost:5555";

    int wifi_channel_num = 42; // ~5210 MHz
    int channelWidth = 80;     // MHz
    double txPowerDbm = 20.0;
    double simTime = 6.0;

    // UDP echo
    uint32_t pktSize = 512;
    uint16_t basePort = 9000;
    double trafficStart = 1.5;
    double stagger = 0.02;

    CommandLine cmd(__FILE__);
    cmd.AddValue("verbose", "Enable logging", verbose);
    cmd.AddValue("caching", "Enable caching inside SionnaPropagationCache", caching);
    cmd.AddValue("enableTraffic", "Send UDP traffic for every node pair", enableTraffic);
    cmd.AddValue("environment", "XML file of environment", environment);
    cmd.AddValue("placements", "placements.csv (tx + sta rows)", placementsPath);
    cmd.AddValue("outCsv", "Output CSV for PL reciprocity results", outCsv);
    cmd.AddValue("server", "Python server address", serverAddr);
    cmd.AddValue("channel", "WiFi channel number", wifi_channel_num);
    cmd.AddValue("channelWidth", "WiFi channel width in MHz", channelWidth);
    cmd.AddValue("txPowerDbm", "TX power (dBm)", txPowerDbm);
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("pktSize", "UDP packet size (bytes)", pktSize);
    cmd.AddValue("basePort", "Base UDP port for per-node servers", basePort);
    cmd.Parse(argc, argv);

    if (verbose)
    {
        LogComponentEnable("ExampleSionnaReciprocityMatrix", LOG_INFO);
        LogComponentEnable("SionnaPropagationLossModel", LOG_INFO);
        LogComponentEnable("SionnaPropagationDelayModel", LOG_INFO);
        LogComponentEnable("SionnaPropagationCache", LOG_INFO);
    }

    std::cout << "ns3sionna pairwise reciprocity test (adhoc, N nodes)\n";
    std::cout << "Env: " << environment << "\n";
    std::cout << "Placements: " << placementsPath << "\n";

    // Read placements
    auto rows = ReadPlacementsCsv(placementsPath);

    // Reorder to: [tx, sta0, sta1, ...]
    Vector txPos;
    std::vector<Vector> staPos;
    for (auto &r : rows)
    {
        if (r.tag == "tx") txPos = r.pos;
        if (r.tag == "sta") staPos.push_back(r.pos);
    }

    uint32_t N = 1 + staPos.size();

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

    // Cache + models
    Ptr<SionnaPropagationCache> propagationCache = CreateObject<SionnaPropagationCache>();
    propagationCache->SetSionnaHelper(sionnaHelper);
    propagationCache->SetCaching(caching);

    Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();

    Ptr<SionnaPropagationLossModel> lossModel = CreateObject<SionnaPropagationLossModel>();
    lossModel->SetPropagationCache(propagationCache);
    spectrumChannel->AddPropagationLossModel(lossModel);

    Ptr<SionnaPropagationDelayModel> delayModel = CreateObject<SionnaPropagationDelayModel>();
    delayModel->SetPropagationCache(propagationCache);
    spectrumChannel->SetPropagationDelayModel(delayModel);

    // Wi-Fi (Adhoc) with SpectrumWifiPhyHelper
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
    BuildIpToNodeIdMap();

    // Applications: per-node echo server on (basePort + nodeIndex)
    for (uint32_t j = 0; j < N; ++j)
    {
        UdpEchoServerHelper serv(basePort + j);
        auto apps = serv.Install(nodes.Get(j));
        apps.Start(Seconds(0.5));
        apps.Stop(Seconds(simTime));
    }

    // Trace RX (optional)
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::UdpEchoServer/RxWithAddresses",
                    MakeCallback(&RxTraceWithAddresses));

    // Optional traffic: explicit BOTH directions for every unordered pair
    if (enableTraffic)
    {
        uint32_t k = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            for (uint32_t j = i + 1; j < N; ++j)
            {
                // i -> j
                {
                    UdpEchoClientHelper cli(ifs.GetAddress(j), basePort + j);
                    cli.SetAttribute("MaxPackets", UintegerValue(1));
                    cli.SetAttribute("Interval", TimeValue(Seconds(1.0)));
                    cli.SetAttribute("PacketSize", UintegerValue(pktSize));
                    auto apps = cli.Install(nodes.Get(i));
                    apps.Start(Seconds(trafficStart + stagger * k));
                    apps.Stop(Seconds(simTime));
                }

                // j -> i
                {
                    UdpEchoClientHelper cli(ifs.GetAddress(i), basePort + i);
                    cli.SetAttribute("MaxPackets", UintegerValue(1));
                    cli.SetAttribute("Interval", TimeValue(Seconds(1.0)));
                    cli.SetAttribute("PacketSize", UintegerValue(pktSize));
                    auto apps = cli.Install(nodes.Get(j));
                    apps.Start(Seconds(trafficStart + stagger * k + 0.5 * stagger));
                    apps.Stop(Seconds(simTime));
                }

                k++;
            }
        }
    }

    // Configure Sionna helper (OFDM params) – needed by some backends; harmless for PL-only
    double fc = get_center_freq(dev.Get(0));
    sionnaHelper.Configure(fc,
                           channelWidth,
                           getFFTSize(wifi_standard, channelWidth),
                           getSubcarrierSpacing(wifi_standard));
    sionnaHelper.SetMode(SionnaHelper::MODE_P2P);

    Simulator::Stop(Seconds(simTime));

    // Connect to Python server
    sionnaHelper.Start();

    // Run a clean reciprocity sweep before traffic starts
    Simulator::Schedule(Seconds(1.0),
        &ComputeAndLogPairwisePlReciprocity, lossModel, nodes, labels, txPowerDbm, outCsv);

    Simulator::Run();
    Simulator::Destroy();

    propagationCache->PrintStats();
    sionnaHelper.Destroy();

    return 0;
}
