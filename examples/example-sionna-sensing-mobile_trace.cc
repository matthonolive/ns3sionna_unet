/*
* Copyright (c) 2024 Zubow
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: zubow@tkn.tu-berlin.de
 */

// C++
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>


// Sionna models
#include "ns3/sionna-helper.h"
#include "ns3/sionna-propagation-cache.h"
#include "ns3/sionna-propagation-delay-model.h"
#include "ns3/sionna-propagation-loss-model.h"
#include "ns3/sionna-spectrum-propagation-loss-model.h"
#include "ns3/sionna-mobility-model.h"
#include "ns3/cfr-tag.h"

// Ns-3 modules
#include <ns3/wifi-spectrum-phy-interface.h>
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/spectrum-module.h"


/**
 * Advanced example showing how to access the CSI computed by Sionna in a scenario with static AP
 * and mobile STA using 80 MHz channel in simple room scenario. STA sends packets to the AP from
 * which the CSI is retrieved and exported to a file for later plotting.
 * Note: Due to mobility the channel needs to be recomputed. The number of recomputations depends
 * on the speed of the mobile (coherence time) and the traffic pattern.
 *
 * Limitations: only SISO so far
 *
 *  To run: ./example-sionna-sensing-mobile.sh
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ExampleSionnaSensing");

// mapping of IPv4 addr to nodeIds
std::map<Ipv4Address, uint32_t> g_ipToNodeIdMap;

// CSI
std::string csiFname = "example-sionna-sensing-mobile.csv";
std::ofstream ofs_csi(csiFname);
// Pathloss
std::string plFname = "example-sionna-sensing-mobile-pathloss.csv";
std::ofstream ofs_pl(plFname);
// RX (STA) node location
std::string tpFname = "example-sionna-sensing-mobile-time-pos.csv";
std::ofstream ofs_tp(tpFname);


// Replayability

// Replayable node-position trace (0=ap, 1=sta)
std::ofstream ofs_nodeTs;

static std::string Trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> SplitCsv(const std::string& line)
{
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(line.size());
    for (char c : line)
    {
        if (c == ',')
        {
            out.push_back(Trim(cur));
            cur.clear();
        }
        else
        {
            cur.push_back(c);
        }
    }
    out.push_back(Trim(cur));
    return out;
}

// Reads a CSV with at least columns: t_s,node,x,y,z
// Extra columns (like label, throughput, etc.) are ignored.
static std::unordered_map<uint32_t, std::vector<std::pair<double, Vector>>>
ReadMobilityTraceCsv(const std::string& path)
{
    std::ifstream f(path);
    NS_ABORT_MSG_IF(!f.is_open(), "Could not open mobilityTraceIn: " << path);

    std::string line;

    // Skip comments/empty lines until header
    while (std::getline(f, line))
    {
        line = Trim(line);
        if (!line.empty() && line[0] != '#') break;
    }
    NS_ABORT_MSG_IF(line.empty(), "mobilityTraceIn has no header: " << path);

    auto hdr = SplitCsv(line);
    std::unordered_map<std::string, int> col;
    for (int i = 0; i < (int)hdr.size(); ++i)
    {
        std::string h = hdr[i];
        for (auto& c : h) c = std::tolower((unsigned char)c);
        col[h] = i;
    }

    auto need = [&](const std::string& name) -> int {
        auto it = col.find(name);
        NS_ABORT_MSG_IF(it == col.end(), "mobilityTraceIn missing column '" << name << "'");
        return it->second;
    };

    int it_t = need("t_s");
    int it_n = need("node");
    int it_x = need("x");
    int it_y = need("y");
    int it_z = need("z");

    std::unordered_map<uint32_t, std::vector<std::pair<double, Vector>>> trace;

    while (std::getline(f, line))
    {
        line = Trim(line);
        if (line.empty() || line[0] == '#') continue;

        auto toks = SplitCsv(line);
        if ((int)toks.size() <= std::max({it_t, it_n, it_x, it_y, it_z}))
        {
            continue;
        }

        double t = std::stod(toks[it_t]);
        uint32_t n = (uint32_t)std::stoul(toks[it_n]);
        double x = std::stod(toks[it_x]);
        double y = std::stod(toks[it_y]);
        double z = std::stod(toks[it_z]);

        trace[n].push_back({t, Vector(x, y, z)});
    }

    // Sort and remove duplicate timestamps per node
    for (auto& kv : trace)
    {
        auto& v = kv.second;
        std::sort(v.begin(), v.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<std::pair<double, Vector>> dedup;
        dedup.reserve(v.size());

        double lastT = -1.0;
        for (const auto& p : v)
        {
            if (!dedup.empty() && std::abs(p.first - lastT) < 1e-12)
            {
                continue;
            }
            dedup.push_back(p);
            lastT = p.first;
        }
        v.swap(dedup);
    }

    return trace;
}

// Replay by scheduling explicit SetPosition() calls.
// This uses the exact sampled positions from the CSV.
static void ApplyMobilityTraceSchedulePositions(const std::string& path,
                                                const NodeContainer& nodes,
                                                const std::vector<Vector>& initPos)
{
    auto trace = ReadMobilityTraceCsv(path);

    NS_ABORT_MSG_IF(initPos.size() != nodes.GetN(), "initPos size mismatch");

    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        Ptr<MobilityModel> mm = nodes.Get(i)->GetObject<MobilityModel>();
        NS_ABORT_MSG_IF(!mm, "Node " << i << " missing MobilityModel");

        // Initial position: use t=0 trace point if present, otherwise fallback
        Vector p0 = initPos[i];
        auto it = trace.find(i);
        if (it != trace.end() && !it->second.empty() &&
            std::abs(it->second.front().first) < 1e-12)
        {
            p0 = it->second.front().second;
        }

        mm->SetPosition(p0);

        if (it != trace.end())
        {
            for (const auto& tp : it->second)
            {
                if (tp.first <= 0.0) continue;
                Simulator::Schedule(Seconds(tp.first),
                                    &MobilityModel::SetPosition,
                                    mm,
                                    tp.second);
            }
        }
    }
}

// Write a replayable CSV trace of all node positions.
// node indices are local to this example: 0=ap, 1=sta.
static void LogNodePositions(NodeContainer nodes,
                             std::vector<std::string> labels,
                             Time period,
                             Time stopTime)
{
    NS_ABORT_MSG_IF(labels.size() != nodes.GetN(), "labels size mismatch");

    double t = Simulator::Now().GetSeconds();
    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        Ptr<MobilityModel> mm = nodes.Get(i)->GetObject<MobilityModel>();
        NS_ABORT_MSG_IF(!mm, "Node " << i << " missing MobilityModel");

        Vector p = mm->GetPosition();
        ofs_nodeTs << t << "," << i << "," << labels[i] << ","
                   << p.x << "," << p.y << "," << p.z << "\n";
    }

    if (Simulator::Now() + period <= stopTime + NanoSeconds(1))
    {
        Simulator::Schedule(period, &LogNodePositions, nodes, labels, period, stopTime);
    }
}



void BuildIpToNodeIdMap ()
{
    g_ipToNodeIdMap.clear ();
    for (uint32_t i = 0; i < NodeList::GetNNodes (); ++i)
    {
        Ptr<Node> node = NodeList::GetNode (i);
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
        if (ipv4)
        {
            uint32_t nInterfaces = ipv4->GetNInterfaces ();
            for (uint32_t j = 0; j < nInterfaces; ++j)
            {
                uint32_t nAddrs = ipv4->GetNAddresses (j);
                for (uint32_t k = 0; k < nAddrs; ++k)
                {
                    Ipv4Address addr = ipv4->GetAddress (j, k).GetLocal ();
                    if (!addr.IsLocalhost())
                        {
                            NS_LOG_DEBUG("Node Id: " << node->GetId () << ", IP: " << addr);
                            g_ipToNodeIdMap[addr] = node->GetId ();  // Insert (overwrites if duplicate)
                        }
                }
            }
        }
    }
    NS_LOG_DEBUG ("Built IP-to-NodeID map with " << g_ipToNodeIdMap.size () << " entries");
}

// Fast lookup function
uint32_t GetNodeIdFromIpv4Address (Ipv4Address targetAddr)
{
    auto it = g_ipToNodeIdMap.find (targetAddr);
    if (it != g_ipToNodeIdMap.end ())
    {
        return it->second;
    }
    NS_LOG_WARN ("No node found for IPv4 address " << targetAddr);
    return 0xFFFFFFFF;
}

/*
 * Trace application layer, here the UdpEchoServer.
 * For each the received packet the CFR (CSI) is retrieved from the tag and dumped to file.
 */
void RxTraceWithAddresses(std::string context, Ptr<const Packet> packet, const Address &from, const Address &to) {

    // Lookup ID of transmitter & receiver
    uint32_t src_nodeId = GetNodeIdFromIpv4Address(InetSocketAddress::ConvertFrom(from).GetIpv4());
    Ptr<Node> src_node = NodeList::GetNode(src_nodeId);
    Ptr<MobilityModel> mobility = src_node->GetObject<MobilityModel>();
    Vector pos = mobility->GetPosition();
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s: Node: " << src_node->GetId() << ": Pos: (" << pos.x << "," << pos.y << "," << pos.z << ")");

    NS_LOG_INFO("*** " << Simulator::Now().GetSeconds() << "s [" << context << "]: Server received packet of " << packet->GetSize() << " bytes"
        << " from: " << InetSocketAddress::ConvertFrom(from).GetIpv4() << "(" << src_nodeId << ") port "
        << " to: " << InetSocketAddress::ConvertFrom(to).GetIpv4() << "(/) port " << InetSocketAddress::ConvertFrom(to).GetPort());

    // check to see whether packet is tagged with CSI/CFR
    CFRTag tag;
    if (packet->PeekPacketTag(tag))
    {
        // dump CSI
        dumpComplexVecToStream(tag.GetComplexes(), ofs_csi);
        // dump pathloss
        double pathLossDb = tag.GetPathloss();
        ofs_pl << pathLossDb << std::endl;
        // dump rx node position
        ofs_tp << Simulator::Now().GetSeconds() << "," << pos.x << "," << pos.y << "," << pos.z << std::endl;
    }
}


int
main(int argc, char* argv[])
{
    bool verbose = true;
    bool tracing = true;
    bool caching = true;
    std::string environment = "simple_room/simple_room.xml";
    int wifi_channel_num = 42; // center at 5210
    int sim_end_time_sec = 10;
    int channel_width = 80;
    int min_coherence_time_ms = 10; // total sim duration
    std::string mobilityTraceIn = "";
    std::string outNodeTsCsv = "example-sionna-sensing-mobile-node-timeseries.csv";
    int mobilitySampleMs = 50;


    CommandLine cmd(__FILE__);
    cmd.AddValue("verbose", "Enable logging", verbose);
    cmd.AddValue("tracing", "Enable pcap tracing", tracing);
    cmd.AddValue("caching", "Enable caching of propagation delay and loss", caching);
    cmd.AddValue("environment", "Xml file of environment", environment);
    cmd.AddValue("channel", "The WiFi channel number", wifi_channel_num);
    cmd.AddValue("simEndTimeSec", "The total simulation time", sim_end_time_sec);
    cmd.AddValue("channelWidth", "The WiFi channel width in MHz", channel_width);
    cmd.AddValue("minCoherenceTimeMs", "The minimal coherence time in msec", min_coherence_time_ms);
    cmd.AddValue("mobilityTraceIn",
             "Replay mobility from CSV with columns t_s,node,x,y,z (extra columns allowed)",
             mobilityTraceIn);
    cmd.AddValue("outNodeTsCsv",
                "Write replayable node-position CSV trace",
                outNodeTsCsv);
    cmd.AddValue("mobilitySampleMs",
                "Sampling period for replayable node-position CSV",
                mobilitySampleMs);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(mobilitySampleMs <= 0, "mobilitySampleMs must be > 0");

    if (verbose)
    {
        LogComponentEnable("ExampleSionnaSensing", LOG_INFO);
        LogComponentEnable("SionnaPropagationDelayModel", LOG_INFO);
        LogComponentEnable("SionnaPropagationLossModel", LOG_INFO);
        LogComponentEnable("SionnaPropagationCache", LOG_INFO);
        LogComponentEnable("SionnaSpectrumPropagationLossModel", LOG_INFO);
    }

    std::cout << "Example spectrum model wifi scenario with sionna" << std::endl << std::endl;

    SionnaHelper sionnaHelper(environment, "tcp://localhost:5555");

    // Create nodes
    NodeContainer wifiStaNode;
    wifiStaNode.Create(1);

    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelWidth", UintegerValue (channel_width));

    Ptr<SionnaPropagationCache> propagationCache = CreateObject<SionnaPropagationCache>();
    propagationCache->SetSionnaHelper(sionnaHelper);
    propagationCache->SetCaching(caching);

    // new
    Ptr<MultiModelSpectrumChannel> spectrumChannel =
        CreateObject<MultiModelSpectrumChannel>();

    Ptr<SionnaPropagationLossModel> lossModel = CreateObject<SionnaPropagationLossModel>();
    lossModel->SetPropagationCache(propagationCache);

    spectrumChannel->AddPropagationLossModel(lossModel);

    // SISO only
    Ptr<SionnaSpectrumPropagationLossModel> spectrumLossModel = CreateObject<SionnaSpectrumPropagationLossModel>();
    spectrumLossModel->SetPropagationCache(propagationCache);

    spectrumChannel->AddSpectrumPropagationLossModel(spectrumLossModel);

    Ptr<SionnaPropagationDelayModel> delayModel = CreateObject<SionnaPropagationDelayModel>();
    delayModel->SetPropagationCache(propagationCache);
    spectrumChannel->SetPropagationDelayModel(delayModel);

    SpectrumWifiPhyHelper spectrumPhy;
    spectrumPhy.SetChannel(spectrumChannel);
    spectrumPhy.SetErrorRateModel("ns3::NistErrorRateModel");
    spectrumPhy.Set("TxPowerStart", DoubleValue(20));
    spectrumPhy.Set("TxPowerEnd", DoubleValue(20));

    WifiMacHelper mac;
    Ssid ssid = Ssid("ns-3-ssid");

    WifiHelper wifi;

    WifiStandard wifi_standard = WIFI_STANDARD_80211ax; // WIFI6
    wifi.SetStandard(wifi_standard);

    std::string channelStr = "{" + std::to_string(wifi_channel_num) + ", " + std::to_string(channel_width) + ", BAND_5GHZ, 0}";

    std::string wifiManager("Ideal");
    uint32_t rtsThreshold = 999999; // disabled even for large A-MPDU
    wifi.SetRemoteStationManager("ns3::" + wifiManager + "WifiManager",
                                 "RtsCtsThreshold",
                                 UintegerValue(rtsThreshold));

    NetDeviceContainer staDevices;
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    spectrumPhy.Set("ChannelSettings", StringValue(channelStr));
    staDevices = wifi.Install(spectrumPhy, mac, wifiStaNode);

    NetDeviceContainer apDevices;
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid), "BeaconGeneration", BooleanValue(true),
        "BeaconInterval", TimeValue(Seconds(5.120)), "EnableBeaconJitter", BooleanValue(false));
    spectrumPhy.Set("ChannelSettings", StringValue(channelStr));
    apDevices = wifi.Install(spectrumPhy, mac, wifiApNode);

    // Mobility configuration
    MobilityHelper mobility;

    // Local node order for replay/logging CSV:
    //   0 = AP
    //   1 = STA
    NodeContainer allNodes;
    allNodes.Add(wifiApNode.Get(0));
    allNodes.Add(wifiStaNode.Get(0));

    std::vector<std::string> nodeLabels = {"ap", "sta"};
    std::vector<Vector> initPos = {
        Vector(1.0, 2.0, 1.0), // AP
        Vector(4.0, 2.0, 1.0)  // STA
    };

    if (!mobilityTraceIn.empty())
    {
        // Replay mode: use explicit scheduled positions from CSV
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobility.Install(wifiApNode);
        mobility.Install(wifiStaNode);

        ApplyMobilityTraceSchedulePositions(mobilityTraceIn, allNodes, initPos);
    }
    else
    {
        // Native random-walk generation mode
        mobility.SetMobilityModel("ns3::SionnaMobilityModel");
        mobility.Install(wifiApNode);

        mobility.SetMobilityModel("ns3::SionnaMobilityModel",
                                "Model",
                                EnumValue(SionnaMobilityModel::MODEL_RANDOM_WALK),
                                "Speed",
                                StringValue("ns3::ConstantRandomVariable[Constant=1.0]"),
                                "Wall",
                                BooleanValue(true));
        mobility.Install(wifiStaNode);

        wifiApNode.Get(0)->GetObject<MobilityModel>()->SetPosition(initPos[0]);
        wifiStaNode.Get(0)->GetObject<MobilityModel>()->SetPosition(initPos[1]);
    }


    // Set up Internet stack and assign IP addresses
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNode);

    Ipv4AddressHelper address;

    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer wifiStaInterfaces = address.Assign(staDevices);
    Ipv4InterfaceContainer wifiApInterfaces = address.Assign(apDevices);

    BuildIpToNodeIdMap();  // Build once here

    ofs_nodeTs.open(outNodeTsCsv, std::ios::out);
    NS_ABORT_MSG_IF(!ofs_nodeTs.is_open(), "Could not open outNodeTsCsv: " << outNodeTsCsv);
    ofs_nodeTs << "t_s,node,label,x,y,z\n";

    // Start sampling at t=0 so this file can be replayed later.
    // node 0 = AP, node 1 = STA
    LogNodePositions(allNodes,
                    nodeLabels,
                    MilliSeconds(mobilitySampleMs),
                    Seconds(sim_end_time_sec));

    // Set up applications
    UdpEchoServerHelper echoServer(9);

    ApplicationContainer serverApps = echoServer.Install(wifiApNode);
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(30.0));

    // App layer tracing of RX events to capture CSI
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::UdpEchoServer/RxWithAddresses",
                    MakeCallback(&RxTraceWithAddresses));

    Ipv4Address wifi_ip_addr = wifiApInterfaces.GetAddress(0);
    //std::cout << "AP IP: " << wifi_ip_addr << std::endl;

    UdpEchoClientHelper echoClient(wifi_ip_addr, 9);
    echoClient.SetAttribute("MaxPackets", UintegerValue(1000000));
    echoClient.SetAttribute("Interval", TimeValue(MilliSeconds(50)));
    echoClient.SetAttribute("PacketSize", UintegerValue(1024));

    ApplicationContainer clientApps = echoClient.Install(wifiStaNode);
    clientApps.Start(Seconds(1.0));
    clientApps.Stop(Seconds(sim_end_time_sec));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // set center frequency for Sionna
    double fc = get_center_freq(apDevices.Get(0));

    // set center frequency & bandwidth for Sionna
    sionnaHelper.Configure(fc, channel_width,
        getFFTSize(wifi_standard, channel_width), getSubcarrierSpacing(wifi_standard), min_coherence_time_ms);
    sionnaHelper.SetMode(SionnaHelper::MODE_P2P); // compute CSI for P2P only; no look-ahead computation

    // Tracing
    if (tracing)
    {
        std::cout << "Writing pcap files ..." << std::endl;
        spectrumPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        spectrumPhy.EnablePcap("example-sionna-sensing", apDevices.Get(0));
    }

    // Simulation end
    Simulator::Stop(Seconds(sim_end_time_sec));

    sionnaHelper.Start();

    Simulator::Run();
    Simulator::Destroy();

    propagationCache->PrintStats();
    sionnaHelper.Destroy();

    ofs_csi.close();
    ofs_pl.close();
    ofs_tp.close();
    std::cout << "Replayable mobility trace written to: " << outNodeTsCsv << std::endl;
    std::cout << "CSI results can be found in: " << csiFname << std::endl;
    std::cout << "For plotting run: python plot3d_mobile_csi.py " << csiFname << " " << plFname << " " << tpFname << std::endl;

    return 0;
}

