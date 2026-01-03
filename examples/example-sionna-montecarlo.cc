/*
 * Monte Carlo driver: 2 nodes, bidirectional UDP, switchable channel model:
 *   - rt/unet: SionnaPropagationLossModel + SionnaSpectrumPropagationLossModel (needs Python server)
 *   - friis:   FriisPropagationLossModel + FriisSpectrumPropagationLossModel (no server)
 *
 * placementsCsv format (matches your gen_random_suite.py):
 *   tx,x,y,z
 *   sta,x,y,z
 *   sta,x,y,z
 *   ...
 */

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

// ns3 core
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"

// wifi/spectrum
#include "ns3/spectrum-module.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/wifi-spectrum-phy-interface.h"
#include "ns3/wifi-module.h"
#include "ns3/ssid.h"

// sionna models
#include "ns3/sionna-helper.h"
#include "ns3/sionna-propagation-cache.h"
#include "ns3/sionna-propagation-delay-model.h"
#include "ns3/sionna-propagation-loss-model.h"
#include "ns3/sionna-spectrum-propagation-loss-model.h"

// friis models
#include "ns3/propagation-loss-model.h"
#include "ns3/friis-spectrum-propagation-loss.h"
#include "ns3/propagation-delay-model.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ExampleSionnaMonteCarlo");

struct PlacementRow
{
  std::string kind; // "tx" or "sta"
  Vector pos;
};

static std::vector<PlacementRow>
LoadPlacementsCsv(const std::string& path)
{
  std::ifstream f(path);
  if (!f.is_open())
  {
    throw std::runtime_error("Failed to open placementsCsv: " + path);
  }

  std::vector<PlacementRow> rows;
  std::string line;
  while (std::getline(f, line))
  {
    if (line.size() == 0) continue;
    std::stringstream ss(line);
    std::string kind, sx, sy, sz;

    std::getline(ss, kind, ',');
    std::getline(ss, sx, ',');
    std::getline(ss, sy, ',');
    std::getline(ss, sz, ',');

    PlacementRow r;
    r.kind = kind;
    r.pos = Vector(std::stod(sx), std::stod(sy), std::stod(sz));
    rows.push_back(r);
  }
  return rows;
}

static Vector
PickTx(const std::vector<PlacementRow>& rows)
{
  for (auto& r : rows) if (r.kind == "tx") return r.pos;
  throw std::runtime_error("placementsCsv has no 'tx' row");
}

static std::vector<Vector>
PickStas(const std::vector<PlacementRow>& rows)
{
  std::vector<Vector> stas;
  for (auto& r : rows) if (r.kind == "sta") stas.push_back(r.pos);
  if (stas.empty())
  {
    throw std::runtime_error("placementsCsv has no 'sta' rows");
  }
  return stas;
}

int
main(int argc, char* argv[])
{
  // --- CLI ---
  bool verbose = true;
  bool tracing = false;
  bool caching = false;

  std::string environment = "2_rooms_with_door/2_rooms_with_door_open.xml";
  std::string placementsCsv = "";
  std::string propModel = "rt"; // rt | unet | friis (rt/unet both use SionnaSpectrumPropagationLossModel)

  int wifiChannelNum = 42; // for 80 MHz at 5 GHz, 42 is typical; for 160 use 50 or 114
  int channelWidth = 80;   // 20/40/80/160

  double txPowerDbm = 20.0;
  double simTime = 3.0;

  // traffic
  uint32_t maxPackets = 3;
  double interval = 0.3; // seconds
  uint32_t packetSize = 1024;
  uint32_t staIndex = 0; // which sta row to use as node1

  // output
  std::string outPrefix = "mc";

  CommandLine cmd(__FILE__);
  cmd.AddValue("verbose", "Enable logging", verbose);
  cmd.AddValue("tracing", "Enable pcap tracing", tracing);
  cmd.AddValue("caching", "Enable Sionna cache (rt/unet)", caching);
  cmd.AddValue("environment", "XML file of environment (relative to model_folder on server)", environment);
  cmd.AddValue("placementsCsv", "placements.csv path", placementsCsv);
  cmd.AddValue("propModel", "rt|unet|friis", propModel);
  cmd.AddValue("channel", "WiFi channel number (center index for width)", wifiChannelNum);
  cmd.AddValue("channelWidth", "Channel width MHz: 20/40/80/160", channelWidth);
  cmd.AddValue("txPowerDbm", "TX power dBm", txPowerDbm);
  cmd.AddValue("simTime", "Simulation time seconds", simTime);
  cmd.AddValue("maxPackets", "Packets each direction", maxPackets);
  cmd.AddValue("interval", "Inter-packet interval seconds", interval);
  cmd.AddValue("packetSize", "UDP packet size bytes", packetSize);
  cmd.AddValue("staIndex", "Which sta line to use (0-based)", staIndex);
  cmd.AddValue("outPrefix", "Prefix for pcap output", outPrefix);
  cmd.Parse(argc, argv);

  if (verbose)
  {
    LogComponentEnable("ExampleSionnaMonteCarlo", LOG_INFO);
    LogComponentEnable("SionnaPropagationDelayModel", LOG_INFO);
    LogComponentEnable("SionnaPropagationLossModel", LOG_INFO);
    LogComponentEnable("SionnaPropagationCache", LOG_INFO);
    LogComponentEnable("SionnaSpectrumPropagationLossModel", LOG_INFO);
  }

  std::cout << "ns3sionna montecarlo (2 nodes, bidirectional UDP)\n"
            << "propModel=" << propModel
            << " env=" << environment
            << " width=" << channelWidth
            << " chan=" << wifiChannelNum << "\n";

  // --- placements ---
  Vector txPos(1.0, 1.0, 1.0);
  Vector rxPos(2.0, 2.0, 1.0);
  if (!placementsCsv.empty())
  {
    auto rows = LoadPlacementsCsv(placementsCsv);
    txPos = PickTx(rows);
    auto stas = PickStas(rows);
    if (staIndex >= stas.size())
    {
      throw std::runtime_error("staIndex out of range for placementsCsv");
    }
    rxPos = stas[staIndex];
  }

  // --- nodes ---
  NodeContainer nodes;
  nodes.Create(2); // node 0, node 1

  // --- spectrum channel ---
  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();

  // If using Sionna (rt/unet), we need these:
  Ptr<SionnaPropagationCache> propagationCache;
  SionnaHelper* sionnaHelperPtr = nullptr;

  if (propModel == "rt" || propModel == "unet")
  {
    // server address fixed in your setup
    static SionnaHelper sionnaHelper(environment, "tcp://localhost:5555");
    sionnaHelperPtr = &sionnaHelper;

    propagationCache = CreateObject<SionnaPropagationCache>();
    propagationCache->SetSionnaHelper(sionnaHelper);
    propagationCache->SetCaching(caching);

    Ptr<SionnaPropagationLossModel> lossModel = CreateObject<SionnaPropagationLossModel>();
    lossModel->SetPropagationCache(propagationCache);
    spectrumChannel->AddPropagationLossModel(lossModel);

    Ptr<SionnaSpectrumPropagationLossModel> spectrumLossModel = CreateObject<SionnaSpectrumPropagationLossModel>();
    spectrumLossModel->SetPropagationCache(propagationCache);
    spectrumChannel->AddSpectrumPropagationLossModel(spectrumLossModel);

    Ptr<SionnaPropagationDelayModel> delayModel = CreateObject<SionnaPropagationDelayModel>();
    delayModel->SetPropagationCache(propagationCache);
    spectrumChannel->SetPropagationDelayModel(delayModel);
  }
  else if (propModel == "friis")
  {
    Ptr<FriisPropagationLossModel> pl = CreateObject<FriisPropagationLossModel>();
    spectrumChannel->AddPropagationLossModel(pl);

    Ptr<FriisSpectrumPropagationLossModel> spl = CreateObject<FriisSpectrumPropagationLossModel>();
    spectrumChannel->AddSpectrumPropagationLossModel(spl);

    Ptr<ConstantSpeedPropagationDelayModel> dm = CreateObject<ConstantSpeedPropagationDelayModel>();
    spectrumChannel->SetPropagationDelayModel(dm);
  }
  else
  {
    throw std::runtime_error("Unknown propModel: " + propModel);
  }

  // --- wifi PHY/MAC ---
  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211ax);

  SpectrumWifiPhyHelper spectrumPhy;
  spectrumPhy.SetChannel(spectrumChannel);
  spectrumPhy.SetErrorRateModel("ns3::NistErrorRateModel");
  spectrumPhy.Set("TxPowerStart", DoubleValue(txPowerDbm));
  spectrumPhy.Set("TxPowerEnd", DoubleValue(txPowerDbm));

  // Channel settings string: {channelNumber, channelWidthMHz, band, primary20Index}
  std::string channelStr = "{" + std::to_string(wifiChannelNum) + ", " + std::to_string(channelWidth) + ", BAND_5GHZ, 0}";
  spectrumPhy.Set("ChannelSettings", StringValue(channelStr));

  WifiMacHelper mac;
  Ssid ssid = Ssid("mc-ssid");

  // Node0 AP, Node1 STA (works fine for 2 nodes)
  NetDeviceContainer apDev, staDev;

  mac.SetType("ns3::ApWifiMac",
              "Ssid", SsidValue(ssid),
              "BeaconGeneration", BooleanValue(true),
              "BeaconInterval", TimeValue(MicroSeconds(1024 * 100)),
              "EnableBeaconJitter", BooleanValue(false));
  apDev = wifi.Install(spectrumPhy, mac, nodes.Get(0));

  mac.SetType("ns3::StaWifiMac",
              "Ssid", SsidValue(ssid),
              "ActiveProbing", BooleanValue(false));
  staDev = wifi.Install(spectrumPhy, mac, nodes.Get(1));

  // --- mobility ---
  MobilityHelper mobility;
  mobility.SetMobilityModel("ns3::SionnaMobilityModel"); // so ns3sionna can query positions consistently
  mobility.Install(nodes);

  nodes.Get(0)->GetObject<MobilityModel>()->SetPosition(txPos);
  nodes.Get(1)->GetObject<MobilityModel>()->SetPosition(rxPos);

  // --- internet ---
  InternetStackHelper stack;
  stack.Install(nodes);

  Ipv4AddressHelper address;
  address.SetBase("10.7.0.0", "255.255.255.0");
  Ipv4InterfaceContainer ifAp = address.Assign(apDev);
  Ipv4InterfaceContainer ifSta = address.Assign(staDev);

  Ipv4Address ip0 = ifAp.GetAddress(0);
  Ipv4Address ip1 = ifSta.GetAddress(0);

  // --- apps: bidirectional UDP using OnOff -> PacketSink ---
  uint16_t port01 = 9000; // 0 -> 1
  uint16_t port10 = 9001; // 1 -> 0

  PacketSinkHelper sink1("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port01));
  ApplicationContainer sinkApps1 = sink1.Install(nodes.Get(1));
  sinkApps1.Start(Seconds(0.1));
  sinkApps1.Stop(Seconds(simTime));

  PacketSinkHelper sink0("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port10));
  ApplicationContainer sinkApps0 = sink0.Install(nodes.Get(0));
  sinkApps0.Start(Seconds(0.1));
  sinkApps0.Stop(Seconds(simTime));

  OnOffHelper onoff01("ns3::UdpSocketFactory", InetSocketAddress(ip1, port01));
  onoff01.SetAttribute("PacketSize", UintegerValue(packetSize));
  onoff01.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
  onoff01.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
  onoff01.SetAttribute("DataRate", DataRateValue(DataRate("1Mbps"))); // rate doesn’t matter much; interval controls packet pacing
  ApplicationContainer app01 = onoff01.Install(nodes.Get(0));

  OnOffHelper onoff10("ns3::UdpSocketFactory", InetSocketAddress(ip0, port10));
  onoff10.SetAttribute("PacketSize", UintegerValue(packetSize));
  onoff10.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
  onoff10.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
  onoff10.SetAttribute("DataRate", DataRateValue(DataRate("1Mbps")));
  ApplicationContainer app10 = onoff10.Install(nodes.Get(1));

  // pace packets by stopping after a computed time window
  double sendDuration = maxPackets * interval + 0.05;
  app01.Start(Seconds(0.5));
  app01.Stop(Seconds(0.5 + sendDuration));

  // stagger reverse direction a bit so you can clearly separate logs if desired
  app10.Start(Seconds(0.5 + sendDuration + 0.2));
  app10.Stop(Seconds(0.5 + 2*sendDuration + 0.2));

  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  // --- configure Sionna helper if used ---
  if (propModel == "rt" || propModel == "unet")
  {
    // center frequency from device
    double fc = get_center_freq(apDev.Get(0));
    uint32_t fft = getFFTSize(WIFI_STANDARD_80211ax, channelWidth);
    double scs = getSubcarrierSpacing(WIFI_STANDARD_80211ax);

    std::cout << "Configure: fc=" << fc/1e6 << " MHz"
              << " channelWidth=" << channelWidth << " MHz"
              << " fft=" << fft
              << " scs=" << scs << " Hz"
              << " (fft*scs=" << (fft*scs/1e6) << " MHz)\n";

    sionnaHelperPtr->Configure(fc, channelWidth, fft, scs);
    sionnaHelperPtr->SetMode(SionnaHelper::MODE_P2P);
  }

  if (tracing)
  {
    spectrumPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    spectrumPhy.EnablePcap(outPrefix + "-ap", apDev.Get(0));
    spectrumPhy.EnablePcap(outPrefix + "-sta", staDev.Get(0));
  }

  Simulator::Stop(Seconds(simTime));

  if (propModel == "rt" || propModel == "unet")
  {
    sionnaHelperPtr->Start();
  }

  Simulator::Run();
  Simulator::Destroy();

  if (propModel == "rt" || propModel == "unet")
  {
    if (propagationCache) propagationCache->PrintStats();
    sionnaHelperPtr->Destroy();
  }

  // sinks summary
  auto s0 = DynamicCast<PacketSink>(sinkApps0.Get(0));
  auto s1 = DynamicCast<PacketSink>(sinkApps1.Get(0));
  std::cout << "[SINK] rxBytes(node0)=" << (s0 ? s0->GetTotalRx() : 0)
            << " rxBytes(node1)=" << (s1 ? s1->GetTotalRx() : 0) << "\n";

  return 0;
}
