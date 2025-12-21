/*
 * example-sionna-unet-compare.cc
 *
 * Compare packet loss / throughput / delay with:
 *   - Sionna models (raytracer OR UNet server) via ns3sionna ZMQ server
 *   - FriisPropagationLossModel baseline (no server)
 *
 * Swap raytracer vs UNet by starting a different Python server on the same endpoint:
 *   python ns3sionna_server.py ...
 *   python ns3unet_server.py ...
 *
 * This file hardcodes the ZMQ endpoint like the upstream ns3sionna examples.
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/propagation-module.h"

// ns3sionna
#include "ns3/sionna-helper.h"
#include "ns3/sionna-propagation-cache.h"
#include "ns3/sionna-propagation-delay-model.h"
#include "ns3/sionna-propagation-loss-model.h"
#include "ns3/sionna-spectrum-propagation-loss-model.h"
#include "ns3/sionna-mobility-model.h"

// spectrum Wi-Fi
#include <ns3/wifi-spectrum-phy-interface.h>
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/spectrum-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ExampleSionnaUnetCompare");

// hardcoded like upstream examples
static const std::string kZmqEndpoint = "tcp://localhost:5555";

static double
GetCenterFreqMhz (Ptr<NetDevice> dev)
{
  Ptr<WifiNetDevice> wifiDev = dev->GetObject<WifiNetDevice> ();
  NS_ABORT_MSG_IF (!wifiDev, "Device is not a WifiNetDevice");
  return (double) wifiDev->GetPhy ()->GetFrequency (); // MHz
}

static double
GetChannelWidthMhz (Ptr<NetDevice> dev)
{
  Ptr<WifiNetDevice> wifiDev = dev->GetObject<WifiNetDevice> ();
  NS_ABORT_MSG_IF (!wifiDev, "Device is not a WifiNetDevice");
  return (double) wifiDev->GetPhy ()->GetChannelWidth (); // MHz
}

// For 802.11ax (HE): subcarrier spacing is 78.125 kHz, and FFT bins scale with bandwidth.
// 20MHz -> 256, 40 -> 512, 80 -> 1024, 160 -> 2048
static uint16_t
HeFftSizeFromChannelWidthMhz (uint16_t bwMhz)
{
  // bwMhz / 0.078125 = bwMhz * 12.8
  const double bins = (double) bwMhz * 12.8;
  return (uint16_t) std::lround (bins);
}

static double
HeSubcarrierSpacingHz ()
{
  return 78.125e3;
}

static void
PrintResults (Ptr<FlowMonitor> monitor, FlowMonitorHelper& flowmon, double simTimeS)
{
  monitor->CheckForLostPackets ();

  auto stats = monitor->GetFlowStats ();

  uint64_t txPkts = 0, rxPkts = 0, lostPkts = 0;
  uint64_t rxBytes = 0;
  double sumDelayS = 0.0, sumJitterS = 0.0;
  uint64_t rxPktForDelay = 0, rxPktForJitter = 0;

  for (const auto& kv : stats)
    {
      const FlowMonitor::FlowStats& st = kv.second;
      txPkts += st.txPackets;
      rxPkts += st.rxPackets;
      lostPkts += st.lostPackets;
      rxBytes += st.rxBytes;
      sumDelayS += st.delaySum.GetSeconds ();
      sumJitterS += st.jitterSum.GetSeconds ();
      rxPktForDelay += st.rxPackets;
      if (st.rxPackets > 1)
        {
          rxPktForJitter += (st.rxPackets - 1);
        }
    }

  const double lossRatio = (txPkts == 0) ? 0.0 : (double) lostPkts / (double) txPkts;
  const double throughputMbps = (simTimeS <= 0.0) ? 0.0 : (double) rxBytes * 8.0 / simTimeS / 1e6;
  const double meanDelayMs = (rxPktForDelay == 0) ? 0.0 : (sumDelayS / (double) rxPktForDelay) * 1e3;
  const double meanJitterMs = (rxPktForJitter == 0) ? 0.0 : (sumJitterS / (double) rxPktForJitter) * 1e3;

  std::cout << "\n=== RESULTS ===\n";
  std::cout << "txPkts/rxPkts   = " << txPkts << " / " << rxPkts << "\n";
  std::cout << "lostPkts        = " << lostPkts << " (loss ratio " << lossRatio << ")\n";
  std::cout << "throughput Mbps = " << throughputMbps << "\n";
  std::cout << "mean delay ms   = " << meanDelayMs << "\n";
  std::cout << "mean jitter ms  = " << meanJitterMs << "\n";
}

int
main (int argc, char* argv[])
{
  // --- Experiment controls ---
  std::string propModel = "sionna";     // sionna | friis
  bool useSpectrum = false;             // false: YansWifiChannel, true: SpectrumWifiPhy + MultiModelSpectrumChannel
  bool caching = true;                  // ns3sionna propagation cache
  std::string mobilityMode = "static";  // static | randomwalk2d

  // sionna config
  std::string environment = "exported_scene/scene.xml"; // relative to server --model_folder

  // topology
  uint32_t nSta = 10;

  // Wi-Fi params
  int wifiChannelNum = 36;
  int channelWidth = 20; // MHz
  double txPowerDbm = 20.0;
  WifiStandard wifiStandard = WIFI_STANDARD_80211ax;

  // traffic
  double simTimeS = 10.0;
  uint32_t packetSize = 1200;
  double offeredRateMbps = 50.0; // total across all STAs

  // placement
  double apX = 10.0, apY = 10.0, apZ = 1.5;
  double staZ = 1.5;
  double areaX = 20.0, areaY = 20.0;
  uint64_t seed = 1;

  // randomwalk params
  double rwSpeedMin = 0.5; // m/s
  double rwSpeedMax = 1.5; // m/s
  double rwPauseS = 0.2;

  CommandLine cmd(__FILE__);
  cmd.AddValue("propModel", "sionna|friis", propModel);
  cmd.AddValue("useSpectrum", "Use SpectrumWifiPhy + MultiModelSpectrumChannel (enables CFR path for sionna)", useSpectrum);
  cmd.AddValue("caching", "Enable ns3sionna propagation cache", caching);
  cmd.AddValue("mobility", "static|randomwalk2d", mobilityMode);

  cmd.AddValue("environment", "XML scene path relative to server --model_folder (sionna)", environment);
  cmd.AddValue("nSta", "Number of STAs", nSta);

  cmd.AddValue("channel", "WiFi channel number (e.g., 36)", wifiChannelNum);
  cmd.AddValue("channelWidth", "WiFi channel width MHz", channelWidth);
  cmd.AddValue("txPowerDbm", "TX power dBm", txPowerDbm);

  cmd.AddValue("simTime", "Simulation time seconds", simTimeS);
  cmd.AddValue("packetSize", "UDP payload bytes", packetSize);
  cmd.AddValue("offeredRateMbps", "Total offered rate across STAs (Mbps)", offeredRateMbps);

  cmd.AddValue("apX", "AP x position (m)", apX);
  cmd.AddValue("apY", "AP y position (m)", apY);
  cmd.AddValue("apZ", "AP z position (m)", apZ);
  cmd.AddValue("staZ", "STA z position (m)", staZ);
  cmd.AddValue("areaX", "STA area X size (m)", areaX);
  cmd.AddValue("areaY", "STA area Y size (m)", areaY);
  cmd.AddValue("seed", "RNG seed", seed);

  cmd.AddValue("rwSpeedMin", "RandomWalk2d min speed (m/s)", rwSpeedMin);
  cmd.AddValue("rwSpeedMax", "RandomWalk2d max speed (m/s)", rwSpeedMax);
  cmd.AddValue("rwPause", "RandomWalk2d pause (s)", rwPauseS);

  cmd.Parse(argc, argv);

  RngSeedManager::SetSeed ((uint32_t) seed);
  RngSeedManager::SetRun (1);

  if (mobilityMode != "static" && caching)
    {
      NS_LOG_WARN ("mobility != static while caching=true. Cache keys include positions in ns3sionna, "
                   "but for debugging it can be clearer to set --caching=false.");
    }

  // --- Nodes ---
  NodeContainer apNode;
  apNode.Create (1);
  NodeContainer staNodes;
  staNodes.Create (nSta);

  // --- Mobility ---
  // AP: fixed
  {
    MobilityHelper mobAp;
    mobAp.SetMobilityModel ("ns3::SionnaMobilityModel");
    mobAp.Install (apNode);
    apNode.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(apX, apY, apZ));
  }

  // STAs: static or random walk (2D)
  {
    MobilityHelper mobSta;

    if (mobilityMode == "static")
      {
        mobSta.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
                                   "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(areaX) + "]"),
                                   "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(areaY) + "]"));
        mobSta.SetMobilityModel ("ns3::SionnaMobilityModel");
        mobSta.Install (staNodes);

        for (uint32_t i = 0; i < nSta; ++i)
          {
            Vector p = staNodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
            staNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(p.x, p.y, staZ));
          }
      }
    else if (mobilityMode == "randomwalk2d")
      {
        mobSta.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
                                   "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(areaX) + "]"),
                                   "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(areaY) + "]"));

        std::ostringstream bounds;
        bounds << "ns3::Rectangle[MinX=0.0|MinY=0.0|MaxX=" << areaX << "|MaxY=" << areaY << "]";

        mobSta.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                "Bounds", StringValue(bounds.str()),
                                "Speed", StringValue("ns3::UniformRandomVariable[Min=" + std::to_string(rwSpeedMin) +
                                                     "|Max=" + std::to_string(rwSpeedMax) + "]"),
                                "Distance", DoubleValue(1.0),
                                "Mode", StringValue("Time"),
                                "Time", TimeValue(Seconds(0.2)),
                                "Direction", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=6.283185307]"));
        mobSta.Install (staNodes);

        for (uint32_t i = 0; i < nSta; ++i)
          {
            Vector p = staNodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
            staNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(p.x, p.y, staZ));
          }
      }
    else
      {
        NS_ABORT_MSG ("Unknown mobility mode: " << mobilityMode << " (use static|randomwalk2d)");
      }
  }

  // --- Wi-Fi common ---
  WifiHelper wifi;
  wifi.SetStandard (wifiStandard);

  // Use a fixed rate so comparisons aren't dominated by different rate adaptation histories
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("HeMcs7"),
                                "ControlMode", StringValue ("HeMcs0"));

  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns3sionna-unet-compare");

  NetDeviceContainer staDevs;
  NetDeviceContainer apDevs;

  // --- Channel / PHY selection ---
  Ptr<SionnaPropagationCache> propagationCache;
  Ptr<SionnaPropagationLossModel> sionnaLossModel;
  Ptr<SionnaPropagationDelayModel> sionnaDelayModel;
  Ptr<SionnaSpectrumPropagationLossModel> sionnaSpectrumLossModel;

  // Only used when propModel == sionna
  std::unique_ptr<SionnaHelper> sionnaHelper;

  if (propModel == "sionna")
    {
      sionnaHelper = std::make_unique<SionnaHelper> (environment, kZmqEndpoint);

      propagationCache = CreateObject<SionnaPropagationCache> ();
      propagationCache->SetSionnaHelper (*sionnaHelper);
      propagationCache->SetCaching (caching);
    }

  if (!useSpectrum)
    {
      // --- Yans ---
      YansWifiChannelHelper chan = YansWifiChannelHelper::Default ();
      Ptr<YansWifiChannel> yansChannel = chan.Create ();

      if (propModel == "sionna")
        {
          sionnaLossModel = CreateObject<SionnaPropagationLossModel> ();
          sionnaDelayModel = CreateObject<SionnaPropagationDelayModel> ();

          sionnaLossModel->SetPropagationCache (propagationCache);
          sionnaDelayModel->SetPropagationCache (propagationCache);

          yansChannel->SetPropagationLossModel (sionnaLossModel);
          yansChannel->SetPropagationDelayModel (sionnaDelayModel);
        }
      else if (propModel == "friis")
        {
          Ptr<FriisPropagationLossModel> friis = CreateObject<FriisPropagationLossModel> ();
          yansChannel->SetPropagationLossModel (friis);
          yansChannel->SetPropagationDelayModel (CreateObject<ConstantSpeedPropagationDelayModel> ());
        }
      else
        {
          NS_ABORT_MSG ("Unknown propModel: " << propModel << " (use sionna|friis)");
        }

      YansWifiPhyHelper phy = YansWifiPhyHelper::Default ();
      phy.SetChannel (yansChannel);
      phy.Set ("TxPowerStart", DoubleValue (txPowerDbm));
      phy.Set ("TxPowerEnd", DoubleValue (txPowerDbm));

      std::string channelStr = "{" + std::to_string(wifiChannelNum) + ", " + std::to_string(channelWidth) + ", BAND_5GHZ, 0}";
      phy.Set ("ChannelSettings", StringValue (channelStr));

      // Install devices
      mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid), "ActiveProbing", BooleanValue (false));
      staDevs = wifi.Install (phy, mac, staNodes);

      mac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid),
                   "BeaconGeneration", BooleanValue (true),
                   "BeaconInterval", TimeValue (Seconds (5.120)),
                   "EnableBeaconJitter", BooleanValue (false));
      apDevs = wifi.Install (phy, mac, apNode);
    }
  else
    {
      // --- Spectrum ---
      Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();

      if (propModel == "sionna")
        {
          // Average (wideband-like) loss model on the spectrum channel
          sionnaLossModel = CreateObject<SionnaPropagationLossModel> ();
          sionnaLossModel->SetPropagationCache (propagationCache);
          spectrumChannel->AddPropagationLossModel (sionnaLossModel);

          // Frequency-selective model (CFR/CSI)
          sionnaSpectrumLossModel = CreateObject<SionnaSpectrumPropagationLossModel> ();
          sionnaSpectrumLossModel->SetPropagationCache (propagationCache);
          spectrumChannel->AddSpectrumPropagationLossModel (sionnaSpectrumLossModel);

          // Delay
          sionnaDelayModel = CreateObject<SionnaPropagationDelayModel> ();
          sionnaDelayModel->SetPropagationCache (propagationCache);
          spectrumChannel->SetPropagationDelayModel (sionnaDelayModel);
        }
      else if (propModel == "friis")
        {
          Ptr<FriisPropagationLossModel> friis = CreateObject<FriisPropagationLossModel> ();
          spectrumChannel->AddPropagationLossModel (friis);
          spectrumChannel->SetPropagationDelayModel (CreateObject<ConstantSpeedPropagationDelayModel> ());
        }
      else
        {
          NS_ABORT_MSG ("Unknown propModel: " << propModel << " (use sionna|friis)");
        }

      SpectrumWifiPhyHelper phy;
      phy.SetChannel (spectrumChannel);
      phy.SetErrorRateModel ("ns3::NistErrorRateModel");
      phy.Set ("TxPowerStart", DoubleValue (txPowerDbm));
      phy.Set ("TxPowerEnd", DoubleValue (txPowerDbm));

      std::string channelStr = "{" + std::to_string(wifiChannelNum) + ", " + std::to_string(channelWidth) + ", BAND_5GHZ, 0}";
      phy.Set ("ChannelSettings", StringValue (channelStr));

      mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid), "ActiveProbing", BooleanValue (false));
      staDevs = wifi.Install (phy, mac, staNodes);

      mac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid),
                   "BeaconGeneration", BooleanValue (true),
                   "BeaconInterval", TimeValue (Seconds (5.120)),
                   "EnableBeaconJitter", BooleanValue (false));
      apDevs = wifi.Install (phy, mac, apNode);
    }

  // --- Internet ---
  InternetStackHelper stack;
  stack.Install (apNode);
  stack.Install (staNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staIfs = address.Assign (staDevs);
  Ipv4InterfaceContainer apIfs  = address.Assign (apDevs);

  // --- Apps: AP -> each STA, split offered rate across STAs ---
  const double perStaRateMbps = offeredRateMbps / std::max (1u, nSta);
  const DataRate perStaRate (std::to_string (perStaRateMbps) + "Mbps");

  ApplicationContainer sinks;
  ApplicationContainer sources;

  for (uint32_t i = 0; i < nSta; ++i)
    {
      const uint16_t port = 9000 + i;
      PacketSinkHelper sink ("ns3::UdpSocketFactory",
                             InetSocketAddress (Ipv4Address::GetAny (), port));
      sinks.Add (sink.Install (staNodes.Get (i)));

      OnOffHelper onoff ("ns3::UdpSocketFactory",
                         InetSocketAddress (staIfs.GetAddress (i), port));
      onoff.SetAttribute ("DataRate", DataRateValue (perStaRate));
      onoff.SetAttribute ("PacketSize", UintegerValue (packetSize));
      onoff.SetAttribute ("StartTime", TimeValue (Seconds (1.0)));
      onoff.SetAttribute ("StopTime", TimeValue (Seconds (simTimeS)));
      sources.Add (onoff.Install (apNode.Get (0)));
    }

  sinks.Start (Seconds (0.5));
  sinks.Stop (Seconds (simTimeS + 0.5));

  // --- Configure/start Sionna helper after devices exist (like upstream) ---
  if (propModel == "sionna")
    {
      const double fcMhz = GetCenterFreqMhz (apDevs.Get (0));
      const double bwMhz = GetChannelWidthMhz (apDevs.Get (0));

      if (!useSpectrum)
        {
          // wideband: center freq + bandwidth
          sionnaHelper->Configure (fcMhz, bwMhz);
        }
      else
        {
          // spectrum: also need FFT size and subcarrier spacing
          const uint16_t fftSize = HeFftSizeFromChannelWidthMhz ((uint16_t) bwMhz);
          const double scsHz = HeSubcarrierSpacingHz ();

          sionnaHelper->Configure (fcMhz, bwMhz, fftSize, scsHz);
          // Optional: only compute CSI on demand (P2P)
          sionnaHelper->SetMode (SionnaHelper::MODE_P2P);
        }

      sionnaHelper->Start ();

      NS_LOG_INFO ("Sionna helper started. endpoint=" << kZmqEndpoint
                   << " env=" << environment
                   << " fcMHz=" << fcMhz << " bwMHz=" << bwMhz
                   << " useSpectrum=" << useSpectrum);
    }

  // --- Flow monitor ---
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll ();

  Simulator::Stop (Seconds (simTimeS + 0.5));
  Simulator::Run ();

  PrintResults (monitor, flowmon, simTimeS);

  if (propModel == "sionna")
    {
      propagationCache->PrintStats ();
      sionnaHelper->Destroy ();
    }

  Simulator::Destroy ();
  return 0;
}
