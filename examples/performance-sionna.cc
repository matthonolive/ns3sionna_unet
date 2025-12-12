/*
 * Copyright (c) 2024 Yannik Pilz, Zubow
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Yannik Pilz <y.pilz@campus.tu-berlin.de>
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/sionna-helper.h"
#include "ns3/sionna-mobility-model.h"
#include "ns3/sionna-propagation-cache.h"
#include "ns3/sionna-propagation-delay-model.h"
#include "ns3/sionna-propagation-loss-model.h"
#include "ns3/sionna-spectrum-propagation-loss-model.h"
#include "ns3/spectrum-module.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/wifi-net-device.h"

/**
 * Example used for benchmarking ns3sionna with fixed WiFi AP and varying numbers of also fixed or
 * mobile stations. The traffic is Echo/UDP/IP broadcast initiated by the WiFi AP.
 * The WiFi setting is: 802.11ax, 20 MHz channel bandwidth.
 * Note: the execution time of the simulation depends heavily on the number of channel
 * recomputations which in turn depends on the speed of the mobile (coherence time) and the traffic
 * pattern.
 *
 * Limitations: only SISO so far
 *
 *  To run: ./benchmark_ns3sionna.sh
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("PerformanceSionna");

double
RunSimulation(const std::string environment,
              const uint32_t numStas,
              const bool mobile_scenario,
              const double mobile_speed,
              const int udp_pkt_interval,
              const bool caching,
              const int mode,
              const int sub_mode,
              const bool verbose)
{
    // Wifi config
    int wifi_channel_num = 40; // center at 5200
    int channel_width = 20;

    SionnaHelper sionnaHelper(environment, "tcp://localhost:5555");

    // variable number of STAs
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(numStas);

    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelWidth",
                UintegerValue(channel_width));

    Ptr<SionnaPropagationCache> propagationCache = CreateObject<SionnaPropagationCache>();
    propagationCache->SetSionnaHelper(sionnaHelper);
    propagationCache->SetCaching(caching);

    // new
    Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();

    Ptr<SionnaPropagationLossModel> lossModel = CreateObject<SionnaPropagationLossModel>();
    lossModel->SetPropagationCache(propagationCache);

    spectrumChannel->AddPropagationLossModel(lossModel);

    // SISO only
    Ptr<SionnaSpectrumPropagationLossModel> spectrumLossModel =
        CreateObject<SionnaSpectrumPropagationLossModel>();
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

    std::string channelStr = "{" + std::to_string(wifi_channel_num) + ", " +
                             std::to_string(channel_width) + ", BAND_5GHZ, 0}";

    std::string wifiManager("Ideal");
    uint32_t rtsThreshold = 999999; // disabled even for large A-MPDU
    wifi.SetRemoteStationManager("ns3::" + wifiManager + "WifiManager",
                                 "RtsCtsThreshold",
                                 UintegerValue(rtsThreshold));

    NetDeviceContainer staDevices;
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    spectrumPhy.Set("ChannelSettings", StringValue(channelStr));
    staDevices = wifi.Install(spectrumPhy, mac, wifiStaNodes);

    NetDeviceContainer apDevices;
    mac.SetType("ns3::ApWifiMac",
                "Ssid",
                SsidValue(ssid),
                "BeaconGeneration",
                BooleanValue(true),
                "BeaconInterval",
                TimeValue(Seconds(5.120)),
                "EnableBeaconJitter",
                BooleanValue(false));
    spectrumPhy.Set("ChannelSettings", StringValue(channelStr));
    apDevices = wifi.Install(spectrumPhy, mac, wifiApNode);

    MobilityHelper mobility;

    if (mobile_scenario)
    {
        // static AP
        mobility.SetMobilityModel("ns3::SionnaMobilityModel");
        mobility.Install(wifiApNode);
        // mobile STA
        mobility.SetMobilityModel("ns3::SionnaMobilityModel",
                                  "Model",
                                  EnumValue(SionnaMobilityModel::MODEL_RANDOM_WALK),
                                  "Speed",
                                  StringValue("ns3::ConstantRandomVariable[Constant=" +
                                              std::to_string(mobile_speed) + "]"),
                                  "Wall",
                                  BooleanValue(true));
        mobility.Install(wifiStaNodes);
    }
    else
    {
        // all nodes are static
        mobility.SetMobilityModel("ns3::SionnaMobilityModel");
        mobility.Install(wifiStaNodes);
        mobility.Install(wifiApNode);
    }

    // random placement of all nodes
    wifiApNode.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(1.0, 2.0, 1.0));

    Ptr<UniformRandomVariable> randX = CreateObject<UniformRandomVariable>();
    Ptr<UniformRandomVariable> randY = CreateObject<UniformRandomVariable>();
    randX->SetAttribute("Min", DoubleValue(0.1));
    randX->SetAttribute("Max", DoubleValue(5.9));
    randY->SetAttribute("Min", DoubleValue(0.1));
    randY->SetAttribute("Max", DoubleValue(3.9));
    for (uint32_t i = 0; i < numStas; i++)
    {
        wifiStaNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(randX->GetValue(), randY->GetValue(), 1.0));
    }

    // configure IP
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);

    Ipv4AddressHelper address;

    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer wifiStaInterfaces = address.Assign(staDevices);
    Ipv4InterfaceContainer wifiApInterfaces = address.Assign(apDevices);

    // configure application
    UdpEchoServerHelper echoServer(9);

    ApplicationContainer serverApps = echoServer.Install(wifiStaNodes);
    serverApps.Start(Seconds(0.9));
    serverApps.Stop(Seconds(10.0));

    UdpEchoClientHelper echoClient(Ipv4Address("255.255.255.255"), 9);
    echoClient.SetAttribute("MaxPackets", UintegerValue(1e9));
    echoClient.SetAttribute("Interval", TimeValue(MilliSeconds(udp_pkt_interval)));
    echoClient.SetAttribute("PacketSize", UintegerValue(100));

    ApplicationContainer clientApps = echoClient.Install(wifiApNode);
    clientApps.Start(Seconds(1.0));
    clientApps.Stop(Seconds(10.0));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // set center frequency for Sionna
    double fc = get_center_freq(apDevices.Get(0));

    // set center frequency & bandwidth for Sionna
    int min_coherence_time_ms = 1000;
    sionnaHelper.Configure(fc,
                           channel_width,
                           getFFTSize(wifi_standard, channel_width),
                           getSubcarrierSpacing(wifi_standard),
                           min_coherence_time_ms);

    sionnaHelper.SetMode(mode);
    sionnaHelper.SetSubMode(sub_mode);

    Simulator::Stop(Seconds(10.0));

    auto startTime = std::chrono::system_clock::now();

    sionnaHelper.Start();

    Simulator::Run();
    Simulator::Destroy();

    propagationCache->PrintStats();

    sionnaHelper.Destroy();

    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    double computationTime = elapsedTime.count();
    std::cout << "Finished simulation with " << numStas << " stations in " << computationTime
              << " sec";

    std::cout << std::endl << std::endl;
    return computationTime;
}

int
main(int argc, char* argv[])
{
    bool verbose = false;
    bool caching = true;
    std::string environment = "simple_room/simple_room.xml";
    bool mobile_scenario = false;
    double mobile_speed = 0.0;
    int udp_pkt_interval = 20;
    int sim_min_stas = 1;
    int sim_max_stas = 4;
    int mode = 1; // todo: 3;
    int sub_mode = 16;

    CommandLine cmd(__FILE__);
    cmd.AddValue("mobile_scenario", "Enable node movement", mobile_scenario);
    cmd.AddValue("mobile_speed", "STA speed when mobile_scenario is true", mobile_speed);
    cmd.AddValue("udp_pkt_interval", "UDP packet interval (in ms) used by STAs", udp_pkt_interval);
    cmd.AddValue("sim_min_stas", "Min number of STAs to be simulated", sim_min_stas);
    cmd.AddValue("sim_max_stas", "Max number of STAs to be simulated", sim_max_stas);
    cmd.AddValue("environment", "Xml file of Sionna environment", environment);
    cmd.AddValue("caching", "Enable caching of propagation delay and loss", caching);
    cmd.AddValue("mode", "The Sionna mode", mode);
    cmd.AddValue("sub_mode", "The Sionna submode", sub_mode);
    cmd.AddValue("verbose", "Enable logging", verbose);
    cmd.Parse(argc, argv);

    if (verbose)
    {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoClientApplication", LOG_PREFIX_TIME);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_PREFIX_TIME);
        LogComponentEnable("YansWifiChannel", LOG_DEBUG);
        LogComponentEnable("YansWifiChannel", LOG_PREFIX_TIME);
        LogComponentEnable("SionnaPropagationDelayModel", LOG_INFO);
        LogComponentEnable("SionnaPropagationDelayModel", LOG_PREFIX_TIME);
        LogComponentEnable("SionnaPropagationCache", LOG_INFO);
        LogComponentEnable("SionnaPropagationCache", LOG_PREFIX_TIME);
    }

    std::cout << "Performance test: 1 AP and N STAs with ns3sionna" << std::endl;
    std::cout << "Config: mob " << mobile_scenario;
    std::cout << " speed " << mobile_speed << " pktinterval " << udp_pkt_interval;
    std::cout << " caching " << caching << " env " << environment;
    std::cout << " mode " << mode << " submode " << sub_mode << std::endl;

    uint32_t numStas = sim_min_stas;
    double computationTime = 0.0;
    while (computationTime < 2 * 60 * 60 &&
           numStas <= (uint32_t)sim_max_stas) // as long as a single run is below 2h
    {
        computationTime = RunSimulation(environment,
                                        numStas,
                                        mobile_scenario,
                                        mobile_speed,
                                        udp_pkt_interval,
                                        caching,
                                        mode,
                                        sub_mode,
                                        verbose);
        numStas = numStas * 2;
    }

    return 0;
}
