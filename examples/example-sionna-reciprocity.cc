/*
 * Reciprocity test for ns3sionna + SpectrumPropagationLoss
 *
 * Two WiFi nodes (AP + STA), both run UDP EchoServer and EchoClient so that
 * packets go both directions and trigger CSI/CFR requests both ways.
 *
 * Goal: check that tau_rms printed inside ns3unet_spectrum.py is identical
 * for both directions (channel reciprocity), for static nodes.
 */

#include <iostream>
#include <map>
#include <string>
#include <fstream>

// Sionna models
#include "ns3/sionna-helper.h"
#include "ns3/sionna-propagation-cache.h"
#include "ns3/sionna-propagation-delay-model.h"
#include "ns3/sionna-propagation-loss-model.h"
#include "ns3/sionna-spectrum-propagation-loss-model.h"
#include "ns3/cfr-tag.h"

// ns-3 modules
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/wifi-spectrum-phy-interface.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ExampleSionnaReciprocity");

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

// Trace server RX: show direction + whether CFRTag is present
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

    CFRTag tag;
    if (packet->PeekPacketTag(tag))
    {
        auto csi = tag.GetComplexes();
        NS_LOG_INFO("  CFRTag present: N=" << csi.size());
    }
    else
    {
        NS_LOG_INFO("  (no CFRTag found on packet)");
    }
}

int
main(int argc, char *argv[])
{
    bool verbose = true;
    bool tracing = false;
    bool caching = false; // IMPORTANT for reciprocity test: ensure both directions hit Python
    std::string environment = "2_rooms_with_door/2_rooms_with_door_open.xml";

    int wifi_channel_num = 42;   // center ~5210 MHz
    int channelWidth = 80;       // MHz
    double txPowerDbm = 20.0;

    double simTime = 3.0;

    // App setup
    uint32_t maxPackets = 1;
    double interval_s = 0.2;
    uint32_t pktSize = 1024;
    uint16_t portA = 9000; // Node0 server
    uint16_t portB = 9001; // Node1 server
    std::string dumpPlacements = ""; // if non-empty, write placements CSV

    CommandLine cmd(__FILE__);
    cmd.AddValue("verbose", "Enable logging", verbose);
    cmd.AddValue("tracing", "Enable pcap tracing", tracing);
    cmd.AddValue("caching", "Enable caching inside SionnaPropagationCache", caching);
    cmd.AddValue("environment", "Xml file of environment", environment);
    cmd.AddValue("channel", "WiFi channel number", wifi_channel_num);
    cmd.AddValue("channelWidth", "WiFi channel width in MHz", channelWidth);
    cmd.AddValue("txPowerDbm", "TX power (dBm)", txPowerDbm);
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("maxPackets", "Max packets per client", maxPackets);
    cmd.AddValue("interval", "Client interval (s)", interval_s);
    cmd.AddValue("packetSize", "UDP packet size (B)", pktSize);
    cmd.AddValue("dumpPlacements", "CSV path to dump node placements", dumpPlacements);
    cmd.Parse(argc, argv);

    if (verbose)
    {
        LogComponentEnable("ExampleSionnaReciprocity", LOG_INFO);
        LogComponentEnable("SionnaPropagationDelayModel", LOG_INFO);
        LogComponentEnable("SionnaPropagationLossModel", LOG_INFO);
        LogComponentEnable("SionnaPropagationCache", LOG_INFO);
        LogComponentEnable("SionnaSpectrumPropagationLossModel", LOG_INFO);
    }

    std::cout << "ns3sionna reciprocity test (2 nodes, bidirectional UDP)\n\n";

    // --- Sionna helper (Python server must be listening on tcp://localhost:5555) ---
    SionnaHelper sionnaHelper(environment, "tcp://localhost:5555");

    // --- Nodes: Node0 = AP, Node1 = STA ---
    NodeContainer staNode;
    staNode.Create(1);

    NodeContainer apNode;
    apNode.Create(1);

    NodeContainer allNodes;
    allNodes.Add(apNode);
    allNodes.Add(staNode);

    // --- Cache ---
    Ptr<SionnaPropagationCache> propagationCache = CreateObject<SionnaPropagationCache>();
    propagationCache->SetSionnaHelper(sionnaHelper);
    propagationCache->SetCaching(caching);

    // --- Spectrum channel with (loss + spectrum loss + delay) ---
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

    // --- WiFi PHY/MAC using SpectrumWifiPhyHelper ---
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
    Ssid ssid = Ssid("reciprocity-ssid");

    std::string channelStr =
        "{" + std::to_string(wifi_channel_num) + ", " + std::to_string(channelWidth) + ", BAND_5GHZ, 0}";

    NetDeviceContainer staDev, apDev;

    mac.SetType("ns3::StaWifiMac",
                "Ssid", SsidValue(ssid),
                "ActiveProbing", BooleanValue(false));
    spectrumPhy.Set("ChannelSettings", StringValue(channelStr));
    staDev = wifi.Install(spectrumPhy, mac, staNode);

    mac.SetType("ns3::ApWifiMac",
                "Ssid", SsidValue(ssid),
                "BeaconGeneration", BooleanValue(true),
                "BeaconInterval", TimeValue(Seconds(1.024)),
                "EnableBeaconJitter", BooleanValue(false));
    spectrumPhy.Set("ChannelSettings", StringValue(channelStr));
    apDev = wifi.Install(spectrumPhy, mac, apNode);

    // --- Mobility: fixed, using SionnaMobilityModel ---
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::SionnaMobilityModel");
    mobility.Install(allNodes);

    // Positions (edit as you like)
    apNode.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(1.0, 2.0, 1.0));   // Node0 (AP)
    staNode.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(32.0, 2.0, 1.0));  // Node1 (STA)

    // --- Dump placements to CSV (optional) ---
    if (!dumpPlacements.empty())
    {
        std::ofstream f(dumpPlacements);
        if (!f.is_open())
        {
            NS_LOG_UNCOND("ERROR: could not open dumpPlacements file: " << dumpPlacements);
        }
        else
        {
            // First line stores the environment XML so the plotting script can auto-load it
            f << "environment," << environment << "\n";
            f << "role,nodeId,x,y,z\n";

            auto writeOne = [&](const std::string& role, Ptr<Node> n)
            {
                Vector p = n->GetObject<MobilityModel>()->GetPosition();
                f << role << "," << n->GetId() << ","
                << p.x << "," << p.y << "," << p.z << "\n";
            };

            writeOne("AP", apNode.Get(0));
            writeOne("STA", staNode.Get(0));
        }
    }


    // --- Internet stack ---
    InternetStackHelper stack;
    stack.Install(allNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer apIf = address.Assign(apDev);
    Ipv4InterfaceContainer staIf = address.Assign(staDev);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    BuildIpToNodeIdMap();

    // --- Applications: BOTH nodes have server + client (forces traffic both ways) ---
    // Node0(AP) server on portA
    UdpEchoServerHelper servA(portA);
    ApplicationContainer servAppsA = servA.Install(apNode.Get(0));
    servAppsA.Start(Seconds(0.5));
    servAppsA.Stop(Seconds(simTime));

    // Node1(STA) server on portB
    UdpEchoServerHelper servB(portB);
    ApplicationContainer servAppsB = servB.Install(staNode.Get(0));
    servAppsB.Start(Seconds(0.5));
    servAppsB.Stop(Seconds(simTime));

    // Trace server RX to confirm CFRTag presence (optional)
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::UdpEchoServer/RxWithAddresses",
                    MakeCallback(&RxTraceWithAddresses));

    // Node1(STA) client -> Node0(AP) server portA
    UdpEchoClientHelper cliToA(apIf.GetAddress(0), portA);
    cliToA.SetAttribute("MaxPackets", UintegerValue(maxPackets));
    cliToA.SetAttribute("Interval", TimeValue(Seconds(interval_s)));
    cliToA.SetAttribute("PacketSize", UintegerValue(pktSize));
    ApplicationContainer cliAppsToA = cliToA.Install(staNode.Get(0));
    cliAppsToA.Start(Seconds(1.0));
    cliAppsToA.Stop(Seconds(simTime));

    // Node0(AP) client -> Node1(STA) server portB
    UdpEchoClientHelper cliToB(staIf.GetAddress(0), portB);
    cliToB.SetAttribute("MaxPackets", UintegerValue(maxPackets));
    cliToB.SetAttribute("Interval", TimeValue(Seconds(interval_s)));
    cliToB.SetAttribute("PacketSize", UintegerValue(pktSize));
    ApplicationContainer cliAppsToB = cliToB.Install(apNode.Get(0));
    cliAppsToB.Start(Seconds(1.0));
    cliAppsToB.Stop(Seconds(simTime));

    // --- Configure Sionna OFDM params from device ---
    double fc = get_center_freq(apDev.Get(0));
    sionnaHelper.Configure(fc,
                           channelWidth,
                           getFFTSize(wifi_standard, channelWidth),
                           getSubcarrierSpacing(wifi_standard));

    // Ensure we only compute the requested link(s)
    sionnaHelper.SetMode(SionnaHelper::MODE_P2P);

    if (tracing)
    {
        std::cout << "Writing pcap files ...\n";
        spectrumPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        spectrumPhy.EnablePcap("example-sionna-reciprocity", apDev.Get(0));
        spectrumPhy.EnablePcap("example-sionna-reciprocity", staDev.Get(0));
    }

    Simulator::Stop(Seconds(simTime));

    // Start ns3sionna helper (connects to Python server)
    sionnaHelper.Start();

    Simulator::Run();
    Simulator::Destroy();

    propagationCache->PrintStats();
    sionnaHelper.Destroy();

    return 0;
}
