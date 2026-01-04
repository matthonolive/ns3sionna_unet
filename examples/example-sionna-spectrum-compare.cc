/* example-sionna-spectrum-compare.cc */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/propagation-module.h"

// spectrum Wi-Fi
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/spectrum-module.h"
#include "ns3/multi-model-spectrum-channel.h"

// ns3sionna
#include "ns3/sionna-helper.h"
#include "ns3/sionna-propagation-cache.h"
#include "ns3/sionna-propagation-delay-model.h"
#include "ns3/sionna-propagation-loss-model.h"
#include "ns3/sionna-spectrum-propagation-loss-model.h"
#include "ns3/sionna-mobility-model.h"

// CFR tag
#include "ns3/cfr-tag.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ExampleSionnaSpectrumCompare");
static const std::string kZmqEndpoint = "tcp://localhost:5555";

// ---- small helpers ----
static double GetCenterFreqMhz(Ptr<NetDevice> dev) {
  auto wifiDev = dev->GetObject<WifiNetDevice>();
  NS_ABORT_MSG_IF(!wifiDev, "Not a WifiNetDevice");
  return (double)wifiDev->GetPhy()->GetFrequency(); // MHz
}
static double GetChannelWidthMhz(Ptr<NetDevice> dev) {
  auto wifiDev = dev->GetObject<WifiNetDevice>();
  NS_ABORT_MSG_IF(!wifiDev, "Not a WifiNetDevice");
  return (double)wifiDev->GetPhy()->GetChannelWidth(); // MHz
}
static uint16_t HeFftSizeFromChannelWidthMhz(uint16_t bwMhz) {
  // 20->256, 40->512, 80->1024, 160->2048
  return (uint16_t)std::lround((double)bwMhz * 12.8);
}
static double HeSubcarrierSpacingHz() { return 78.125e3; }

static bool LoadPlacementsCsv(const std::string& path, Vector& tx, std::vector<Vector>& stas) {
  std::ifstream f(path);
  if (!f.is_open()) return false;

  std::string line;
  bool haveTx = false;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    std::string tag, xs, ys, zs;
    if (!std::getline(ss, tag, ',')) continue;
    if (!std::getline(ss, xs, ',')) continue;
    if (!std::getline(ss, ys, ',')) continue;
    if (!std::getline(ss, zs, ',')) continue;
    double x = std::stod(xs), y = std::stod(ys), z = std::stod(zs);
    if (tag == "tx" || tag == "ap") { tx = Vector(x,y,z); haveTx = true; }
    if (tag == "sta" || tag == "rx") { stas.emplace_back(x,y,z); }
  }
  return haveTx && !stas.empty();
}
static void WritePlacementsCsv(const std::string& path, const Vector& tx, const std::vector<Vector>& stas) {
  std::ofstream f(path, std::ios::out | std::ios::trunc);
  f << "# tag,x,y,z\n";
  f << "tx," << tx.x << "," << tx.y << "," << tx.z << "\n";
  for (auto& s: stas) f << "sta," << s.x << "," << s.y << "," << s.z << "\n";
}
static void DumpComplexVecToCsv(const std::string& path, const std::vector<std::complex<double>>& v) {
  std::ofstream f(path, std::ios::out | std::ios::trunc);
  f << "k,real,imag,mag2\n";
  for (size_t k=0;k<v.size();++k) {
    double re=v[k].real(), im=v[k].imag();
    f << k << "," << std::setprecision(17) << re << "," << im << "," << (re*re+im*im) << "\n";
  }
}
static void AppendCfrSummary(const std::string& path, double t, uint32_t src, uint32_t dst,
                            uint32_t bytes, double pathlossDb, double meanMag2, uint32_t nSc) {
  const bool exists = std::ifstream(path).good();
  std::ofstream f(path, std::ios::out | std::ios::app);
  if (!exists) f << "time_s,src_node,dst_node,pkt_bytes,pathloss_db,mean_mag2,n_subcarriers\n";
  f << std::fixed << std::setprecision(6)
    << t << "," << src << "," << dst << "," << bytes << ","
    << std::setprecision(9) << pathlossDb << "," << meanMag2 << "," << nSc << "\n";
}

static void PrintResults(Ptr<FlowMonitor> mon, double simTimeS) {
  mon->CheckForLostPackets();
  auto stats = mon->GetFlowStats();
  uint64_t tx=0, rx=0, lost=0, rxBytes=0;
  for (auto& kv: stats) {
    tx += kv.second.txPackets;
    rx += kv.second.rxPackets;
    lost += kv.second.lostPackets;
    rxBytes += kv.second.rxBytes;
  }
  double thrMbps = (simTimeS>0) ? (double)rxBytes*8.0/simTimeS/1e6 : 0.0;
  std::cout << "\n=== RESULTS ===\n"
            << "tx/rx/lost = " << tx << "/" << rx << "/" << lost << "\n"
            << "throughput Mbps = " << thrMbps << "\n";
}

// ---- CFR dump plumbing ----
static std::map<Ipv4Address, uint32_t> g_ipToNode;
static std::map<std::pair<uint32_t,uint32_t>, uint32_t> g_linkCnt;
static bool g_dumpCfr=false, g_dumpFull=false;
static uint32_t g_dumpMaxPerLink=1;
static std::string g_outPrefix="run";

static void BuildIpToNodeIdMap() {
  g_ipToNode.clear();
  for (uint32_t i=0;i<NodeList::GetNNodes();++i) {
    auto n = NodeList::GetNode(i);
    auto ipv4 = n->GetObject<Ipv4>();
    if (!ipv4) continue;
    for (uint32_t j=0;j<ipv4->GetNInterfaces();++j) {
      for (uint32_t k=0;k<ipv4->GetNAddresses(j);++k) {
        auto addr = ipv4->GetAddress(j,k).GetLocal();
        if (!addr.IsLocalhost()) g_ipToNode[addr] = n->GetId();
      }
    }
  }
}
static uint32_t NodeIdFromContext(const std::string& ctx) {
  // "/NodeList/<id>/ApplicationList/..."
  auto p = ctx.find("/NodeList/");
  if (p==std::string::npos) return 0xFFFFFFFF;
  p += std::string("/NodeList/").size();
  auto q = ctx.find("/", p);
  if (q==std::string::npos) return 0xFFFFFFFF;
  return (uint32_t)std::stoul(ctx.substr(p, q-p));
}
static uint32_t NodeIdFromIpv4(Ipv4Address a) {
  auto it = g_ipToNode.find(a);
  return (it==g_ipToNode.end()) ? 0xFFFFFFFF : it->second;
}

static void SinkRxTrace(std::string ctx, Ptr<const Packet> pkt, const Address& from, const Address&) {
  if (!g_dumpCfr) return;
  uint32_t dst = NodeIdFromContext(ctx);
  uint32_t src = 0xFFFFFFFF;
  if (InetSocketAddress::IsMatchingType(from)) {
    src = NodeIdFromIpv4(InetSocketAddress::ConvertFrom(from).GetIpv4());
  }
  auto key = std::make_pair(src,dst);
  uint32_t& cnt = g_linkCnt[key];
  if (cnt >= g_dumpMaxPerLink) return;

  CFRTag tag;
  if (!pkt->PeekPacketTag(tag)) return;

  auto cfr = tag.GetComplexes();
  double meanMag2=0.0;
  for (auto& c: cfr) { meanMag2 += c.real()*c.real() + c.imag()*c.imag(); }
  if (!cfr.empty()) meanMag2 /= (double)cfr.size();

  AppendCfrSummary(g_outPrefix + "_cfr_summary.csv",
                   Simulator::Now().GetSeconds(),
                   src, dst, pkt->GetSize(),
                   tag.GetPathloss(), meanMag2, (uint32_t)cfr.size());

  if (g_dumpFull) {
    std::ostringstream fn;
    fn << g_outPrefix << "_cfr_src" << src << "_dst" << dst << "_n" << cnt << ".csv";
    DumpComplexVecToCsv(fn.str(), cfr);
  }
  cnt++;
}

int main(int argc, char** argv) {
  std::string propModel="sionna"; // sionna|friis
  std::string environment="2_rooms_with_door/2_rooms_with_door_open.xml";
  bool caching=true;

  uint32_t nSta=5;
  int wifiChannelNum=42;
  int channelWidth=80;
  double txPowerDbm=20.0;
  double offeredRateMbps=50.0;
  uint32_t packetSize=1200;
  double simTimeS=10.0;

  std::string placementsCsv="";
  std::string writePlacementsCsv="";
  bool dumpCfr=true, dumpFullCfr=false;
  uint32_t dumpMaxPerLink=1;
  std::string outPrefix="out";
  uint32_t seed=1;

  CommandLine cmd(__FILE__);
  cmd.AddValue("propModel", "sionna|friis", propModel);
  cmd.AddValue("environment", "env XML relative to server --model_folder", environment);
  cmd.AddValue("caching", "ns3sionna cache", caching);
  cmd.AddValue("nSta", "num STAs", nSta);
  cmd.AddValue("channel", "wifi channel num", wifiChannelNum);
  cmd.AddValue("channelWidth", "MHz", channelWidth);
  cmd.AddValue("txPowerDbm", "dBm", txPowerDbm);
  cmd.AddValue("offeredRateMbps", "AP total", offeredRateMbps);
  cmd.AddValue("packetSize", "bytes", packetSize);
  cmd.AddValue("simTime", "seconds", simTimeS);
  cmd.AddValue("seed", "seed", seed);
  cmd.AddValue("placementsCsv", "tx/sta positions", placementsCsv);
  cmd.AddValue("writePlacementsCsv", "dump positions", writePlacementsCsv);
  cmd.AddValue("dumpCfr", "extract CFRTag at sinks", dumpCfr);
  cmd.AddValue("dumpFullCfr", "write full CFR vectors", dumpFullCfr);
  cmd.AddValue("dumpMaxPerLink", "max dumps per (src,dst)", dumpMaxPerLink);
  cmd.AddValue("outPrefix", "output prefix", outPrefix);
  cmd.Parse(argc, argv);

  RngSeedManager::SetSeed(seed);
  RngSeedManager::SetRun(1);

  g_dumpCfr = dumpCfr;
  g_dumpFull = dumpFullCfr;
  g_dumpMaxPerLink = dumpMaxPerLink;
  g_outPrefix = outPrefix;

  NodeContainer staNodes; staNodes.Create(nSta);
  NodeContainer apNode; apNode.Create(1);

  Vector apPos(10,10,1.5);
  std::vector<Vector> staPos;
  staPos.reserve(nSta);

  if (!placementsCsv.empty()) {
    Vector tx; std::vector<Vector> stas;
    if (!LoadPlacementsCsv(placementsCsv, tx, stas) || stas.size() < nSta) return 1;
    apPos = tx;
    for (uint32_t i=0;i<nSta;++i) staPos.push_back(stas[i]);
  } else {
    Ptr<UniformRandomVariable> ux = CreateObject<UniformRandomVariable>();
    Ptr<UniformRandomVariable> uy = CreateObject<UniformRandomVariable>();
    ux->SetStream(seed+100); uy->SetStream(seed+200);
    for (uint32_t i=0;i<nSta;++i) staPos.emplace_back(ux->GetValue(0,80), uy->GetValue(0,40), 1.0);
  }

  MobilityHelper mobAp; mobAp.SetMobilityModel("ns3::SionnaMobilityModel"); mobAp.Install(apNode);
  apNode.Get(0)->GetObject<MobilityModel>()->SetPosition(apPos);

  MobilityHelper mobSta; mobSta.SetMobilityModel("ns3::SionnaMobilityModel"); mobSta.Install(staNodes);
  for (uint32_t i=0;i<nSta;++i) staNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(staPos[i]);

  if (!writePlacementsCsv.empty()) WritePlacementsCsv(writePlacementsCsv, apPos, staPos);

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211ax);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                               "DataMode", StringValue("HeMcs7"),
                               "ControlMode", StringValue("HeMcs0"));

  WifiMacHelper mac;
  Ssid ssid("sionna-spectrum-compare");

  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();

  Ptr<SionnaHelper> sionnaHelper;
  Ptr<SionnaPropagationCache> propCache;

  if (propModel == "sionna") {
    sionnaHelper = CreateObject<SionnaHelper>(environment, kZmqEndpoint);
    propCache = CreateObject<SionnaPropagationCache>();
    propCache->SetSionnaHelper(*sionnaHelper);
    propCache->SetCaching(caching);

    auto loss = CreateObject<SionnaPropagationLossModel>();
    loss->SetPropagationCache(propCache);
    spectrumChannel->AddPropagationLossModel(loss);

    auto specLoss = CreateObject<SionnaSpectrumPropagationLossModel>();
    specLoss->SetPropagationCache(propCache);
    spectrumChannel->AddSpectrumPropagationLossModel(specLoss);

    auto delay = CreateObject<SionnaPropagationDelayModel>();
    delay->SetPropagationCache(propCache);
    spectrumChannel->SetPropagationDelayModel(delay);
  } else {
    spectrumChannel->AddPropagationLossModel(CreateObject<FriisPropagationLossModel>());
    spectrumChannel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());
  }

  SpectrumWifiPhyHelper phy;
  phy.SetChannel(spectrumChannel);
  phy.SetErrorRateModel("ns3::NistErrorRateModel");
  phy.Set("TxPowerStart", DoubleValue(txPowerDbm));
  phy.Set("TxPowerEnd", DoubleValue(txPowerDbm));
  std::string channelStr = "{" + std::to_string(wifiChannelNum) + ", " + std::to_string(channelWidth) + ", BAND_5GHZ, 0}";
  phy.Set("ChannelSettings", StringValue(channelStr));

  mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
  NetDeviceContainer staDevs = wifi.Install(phy, mac, staNodes);

  mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid),
              "BeaconGeneration", BooleanValue(true),
              "BeaconInterval", TimeValue(Seconds(5.120)),
              "EnableBeaconJitter", BooleanValue(false));
  NetDeviceContainer apDevs = wifi.Install(phy, mac, apNode);

  InternetStackHelper stack; stack.Install(apNode); stack.Install(staNodes);
  Ipv4AddressHelper addr; addr.SetBase("10.1.1.0", "255.255.255.0");
  auto staIfs = addr.Assign(staDevs);
  addr.Assign(apDevs);

  BuildIpToNodeIdMap();

  // apps
  double perStaRateMbps = offeredRateMbps / std::max(1u,nSta);
  DataRate perStaRate(std::to_string(perStaRateMbps) + "Mbps");

  ApplicationContainer sinks, sources;
  for (uint32_t i=0;i<nSta;++i) {
    uint16_t port = 9000 + i;
    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    sinks.Add(sink.Install(staNodes.Get(i)));

    OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(staIfs.GetAddress(i), port));
    onoff.SetAttribute("DataRate", DataRateValue(perStaRate));
    onoff.SetAttribute("PacketSize", UintegerValue(packetSize));
    onoff.SetAttribute("StartTime", TimeValue(Seconds(1.0)));
    onoff.SetAttribute("StopTime", TimeValue(Seconds(simTimeS)));
    sources.Add(onoff.Install(apNode.Get(0)));
  }
  sinks.Start(Seconds(0.5));
  sinks.Stop(Seconds(simTimeS + 0.5));

  if (dumpCfr) {
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSink/RxWithAddresses",
                    MakeCallback(&SinkRxTrace));
  }

  if (propModel == "sionna") {
    double fcMhz = GetCenterFreqMhz(apDevs.Get(0));
    double bwMhz = GetChannelWidthMhz(apDevs.Get(0));
    uint16_t fft = HeFftSizeFromChannelWidthMhz((uint16_t)std::lround(bwMhz));
    double scs = HeSubcarrierSpacingHz();
    sionnaHelper->Configure(fcMhz, bwMhz, fft, scs);
    sionnaHelper->SetMode(SionnaHelper::MODE_P2P);
    sionnaHelper->Start();
  }

  FlowMonitorHelper fm;
  auto mon = fm.InstallAll();

  Simulator::Stop(Seconds(simTimeS + 0.5));
  Simulator::Run();

  PrintResults(mon, simTimeS);

  if (propModel == "sionna") {
    propCache->PrintStats();
    sionnaHelper->Destroy();
  }

  Simulator::Destroy();
  return 0;
}
