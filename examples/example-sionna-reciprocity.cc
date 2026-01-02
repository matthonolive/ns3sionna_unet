#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/udp-echo-helper.h"
#include "ns3/wifi-module.h"

// sionna
#include "ns3/sionna-module.h"

#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

using namespace ns3;

static uint32_t
NodeIdFromIp (Ipv4Address ip)
{
  // Assumes your addressing is 10.1.1.(nodeId+1) like in the sensing example
  const std::string s = ip.ToString ();
  // last octet
  const auto lastDot = s.find_last_of ('.');
  if (lastDot == std::string::npos) return 0;
  const int last = std::stoi (s.substr (lastDot + 1));
  return (last > 0) ? static_cast<uint32_t> (last - 1) : 0;
}

static void
DumpCfrCsv (Ptr<const Packet> packet, const Address &from, const Address &to, uint32_t rxNodeId)
{
  CFRTag tag;
  if (!packet->PeekPacketTag (tag))
    {
      return;
    }

  const auto fromInet = InetSocketAddress::ConvertFrom (from);
  const uint32_t txNodeId = NodeIdFromIp (fromInet.GetIpv4 ());

  std::ostringstream fname;
  fname << "csi_tx" << txNodeId << "_rx" << rxNodeId << ".csv";
  const std::string path = fname.str ();

  static std::map<std::string, std::unique_ptr<std::ofstream>> files;

  if (files.find (path) == files.end ())
    {
      files[path] = std::make_unique<std::ofstream> (path, std::ios::out);
      (*files[path]) << "t_s,f_hz,abs2,re,im\n";
      (*files[path]) << std::setprecision (12);
    }

  const auto freqs = tag.GetFrequencies ();
  const auto H = tag.GetComplexes ();

  const double t = Simulator::Now ().GetSeconds ();

  const uint32_t n = std::min (freqs.size (), H.size ());
  for (uint32_t i = 0; i < n; ++i)
    {
      const double f = freqs[i];
      const float re = H[i].real ();
      const float im = H[i].imag ();
      const double abs2 = static_cast<double> (re) * re + static_cast<double> (im) * im;

      (*files[path]) << t << "," << f << "," << abs2 << "," << re << "," << im << "\n";
    }

  files[path]->flush ();
}

int
main (int argc, char *argv[])
{
  std::string env = "2_rooms_with_door/2_rooms_with_door_open.xml";
  double simTime = 2.0;
  double txPowerDbm = 20.0;
  bool useSpectrum = true;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("env", "Scene XML under models/", env);
  cmd.AddValue ("simTime", "Simulation time [s]", simTime);
  cmd.AddValue ("txPowerDbm", "TX power [dBm]", txPowerDbm);
  cmd.AddValue ("useSpectrum", "Use spectrum propagation model", useSpectrum);
  cmd.Parse (argc, argv);

  // ---- nodes: 1 STA + 1 AP ----
  NodeContainer wifiStaNodes;
  wifiStaNodes.Create (1);
  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  // ---- mobility (fixed) ----
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiStaNodes);
  mobility.Install (wifiApNode);

  // Choose two positions you expect to be “interesting”
  wifiStaNodes.Get (0)->GetObject<MobilityModel> ()->SetPosition (Vector (2.0, 2.0, 1.0));
  wifiApNode.Get (0)->GetObject<MobilityModel> ()->SetPosition (Vector (10.0, 6.0, 1.0));

  // ---- wifi + spectrum channel ----
  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ax);

  SpectrumWifiPhyHelper phy;
  phy.Set ("TxPowerStart", DoubleValue (txPowerDbm));
  phy.Set ("TxPowerEnd", DoubleValue (txPowerDbm));

  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();

  // Sionna cache (talks to your python server)
  Ptr<SionnaPropagationCache> propagationCache = CreateObject<SionnaPropagationCache> ();
  propagationCache->SetAttribute ("ModelType", StringValue ("position"));
  propagationCache->SetAttribute ("Scene", StringValue (env));
  propagationCache->SetAttribute ("MaxCacheSize", UintegerValue (1000000));
  propagationCache->SetAttribute ("MinTc", TimeValue (MilliSeconds (100000))); // big => static
  spectrumChannel->AddPropagationLossModel (propagationCache);

  if (useSpectrum)
    {
      Ptr<SionnaSpectrumPropagationLossModel> sionnaSpectrumLoss = CreateObject<SionnaSpectrumPropagationLossModel> ();
      sionnaSpectrumLoss->SetPropagationCache (propagationCache);
      spectrumChannel->AddSpectrumPropagationLossModel (sionnaSpectrumLoss);
    }

  phy.SetChannel (spectrumChannel);

  WifiMacHelper mac;
  Ssid ssid = Ssid ("reciprocity-ssid");

  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));
  NetDeviceContainer staDevice = wifi.Install (phy, mac, wifiStaNodes);

  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevice = wifi.Install (phy, mac, wifiApNode);

  // ---- internet ----
  InternetStackHelper stack;
  stack.Install (wifiStaNodes);
  stack.Install (wifiApNode);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staIf = address.Assign (staDevice);
  Ipv4InterfaceContainer apIf = address.Assign (apDevice);

  // ---- apps: echo server on BOTH ends, and one client each direction ----
  const uint16_t portAp = 9;
  const uint16_t portSta = 10;

  // Server on AP (receives STA->AP)
  UdpEchoServerHelper apServer (portAp);
  ApplicationContainer apServerApp = apServer.Install (wifiApNode.Get (0));
  apServerApp.Start (Seconds (0.2));
  apServerApp.Stop (Seconds (simTime));

  // Server on STA (receives AP->STA)
  UdpEchoServerHelper staServer (portSta);
  ApplicationContainer staServerApp = staServer.Install (wifiStaNodes.Get (0));
  staServerApp.Start (Seconds (0.2));
  staServerApp.Stop (Seconds (simTime));

  // Client on STA sending to AP
  UdpEchoClientHelper staToAp (apIf.GetAddress (0), portAp);
  staToAp.SetAttribute ("MaxPackets", UintegerValue (1));
  staToAp.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
  staToAp.SetAttribute ("PacketSize", UintegerValue (1000));
  ApplicationContainer staClientApp = staToAp.Install (wifiStaNodes.Get (0));
  staClientApp.Start (Seconds (1.0));
  staClientApp.Stop (Seconds (simTime));

  // Client on AP sending to STA (offset slightly to avoid same-slot weirdness)
  UdpEchoClientHelper apToSta (staIf.GetAddress (0), portSta);
  apToSta.SetAttribute ("MaxPackets", UintegerValue (1));
  apToSta.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
  apToSta.SetAttribute ("PacketSize", UintegerValue (1000));
  ApplicationContainer apClientApp = apToSta.Install (wifiApNode.Get (0));
  apClientApp.Start (Seconds (1.000001));
  apClientApp.Stop (Seconds (simTime));

  // ---- CFRTag dumping on BOTH receivers ----
  const uint32_t staId = wifiStaNodes.Get (0)->GetId ();
  const uint32_t apId = wifiApNode.Get (0)->GetId ();

  // AP echo server RxWithAddresses => logs tx=STA, rx=AP
  {
    std::ostringstream p;
    p << "/NodeList/" << apId << "/ApplicationList/0/$ns3::UdpEchoServer/RxWithAddresses";
    Config::ConnectWithoutContext (p.str (), MakeBoundCallback (&DumpCfrCsv, apId));
  }

  // STA echo server RxWithAddresses => logs tx=AP, rx=STA
  {
    std::ostringstream p;
    p << "/NodeList/" << staId << "/ApplicationList/0/$ns3::UdpEchoServer/RxWithAddresses";
    Config::ConnectWithoutContext (p.str (), MakeBoundCallback (&DumpCfrCsv, staId));
  }

  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  Simulator::Destroy ();

  return 0;
}
