/*
 * Copyright (c) 2024 Yannik Pilz
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Yannik Pilz <y.pilz@campus.tu-berlin.de>
 */

#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/ssid.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ExampleNs3");

int
main(int argc, char* argv[])
{
    bool verbose = true;
    double frequency = 2437e6;
    bool tracing = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("verbose", "Enable logging", verbose);
    cmd.AddValue("frequency", "Frequency to be used in the simulation", frequency);
    cmd.AddValue("tracing", "Enable pcap tracing", tracing);
    cmd.Parse(argc, argv);

    if (verbose)
    {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
        LogComponentEnable("YansWifiChannel", LOG_DEBUG);
        LogComponentEnable("YansWifiChannel", LOG_PREFIX_TIME);
        LogComponentEnable("HybridBuildingsPropagationLossModel", LOG_INFO);
        LogComponentEnable("BuildingsPropagationLossModel", LOG_INFO);
        LogComponentEnable("ItuR1238PropagationLossModel", LOG_INFO);
    }

    std::cout << "Example scenario with ns3" << std::endl << std::endl;

    // Create a building
    double x_min = 0.0;
    double x_max = 6.0;
    double y_min = 0.0;
    double y_max = 4.0;
    double z_min = 0.0;
    double z_max = 2.5;
    Ptr<Building> b = CreateObject<Building>();
    b->SetBoundaries(Box(x_min, x_max, y_min, y_max, z_min, z_max));
    b->SetBuildingType(Building::Office);
    b->SetExtWallsType(Building::StoneBlocks);
    b->SetNFloors(1);
    b->SetNRoomsX(1);
    b->SetNRoomsY(1);

    // Create nodes
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(2);

    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    // Create a channel
    Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel>();
    Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel>();
    Ptr<HybridBuildingsPropagationLossModel> lossModel = CreateObject<HybridBuildingsPropagationLossModel>();
    lossModel->SetFrequency(frequency);
    channel->SetPropagationLossModel(lossModel);
    channel->SetPropagationDelayModel(delayModel);

    // WiFi configuration
    YansWifiPhyHelper phy;
    phy.SetChannel(channel);

    WifiMacHelper mac;
    Ssid ssid = Ssid("ns-3-ssid");

    WifiHelper wifi;

    NetDeviceContainer staDevices;
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    staDevices = wifi.Install(phy, mac, wifiStaNodes);

    NetDeviceContainer apDevices;
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid), "BeaconGeneration", BooleanValue(true), "BeaconInterval", TimeValue(Seconds(5.120)), "EnableBeaconJitter", BooleanValue(false));
    apDevices = wifi.Install(phy, mac, wifiApNode);

    // Mobility configuration
    MobilityHelper mobility;

    mobility.Install(wifiStaNodes);
    mobility.Install(wifiApNode);

    BuildingsHelper::Install(wifiStaNodes);
    BuildingsHelper::Install(wifiApNode);

    wifiStaNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(5.0, 2.05, 1.0));
    wifiStaNodes.Get(1)->GetObject<MobilityModel>()->SetPosition(Vector(5.0, 1.95, 1.0));
    wifiApNode.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(1.0, 2.0, 1.0));

    // Set up Internet stack and assign IP addresses
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);

    Ipv4AddressHelper address;

    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer wifiStaInterfaces = address.Assign(staDevices);
    Ipv4InterfaceContainer wifiApInterfaces = address.Assign(apDevices);

    // Set up applications
    UdpEchoServerHelper echoServer(9);

    ApplicationContainer serverApps = echoServer.Install(wifiApNode);
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(10.0));

    UdpEchoClientHelper echoClient(wifiApInterfaces.GetAddress(0), 9);
    echoClient.SetAttribute("MaxPackets", UintegerValue(2));
    echoClient.SetAttribute("Interval", TimeValue(Seconds(1.0)));
    echoClient.SetAttribute("PacketSize", UintegerValue(1024));

    ApplicationContainer clientApps = echoClient.Install(wifiStaNodes);
    clientApps.Start(Seconds(2.0));
    clientApps.Stop(Seconds(10.0));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Tracing
    if (tracing)
    {
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        phy.EnablePcap("example-ns3", apDevices.Get(0));
        phy.EnablePcap("example-ns3", staDevices.Get(0));
        phy.EnablePcap("example-ns3", staDevices.Get(1));
    }

    if (verbose)
    {
        // Print node information
        std::cout << "----------Node Information----------" << std::endl;
        NodeContainer c = NodeContainer::GetGlobal();
        for (auto iter = c.Begin(); iter != c.End(); ++iter)
        {
            std::cout << "NodeID: " << (*iter)->GetId() << ", ";

            Ptr<MobilityModel> mobilityModel = (*iter)->GetObject<MobilityModel>();
            if (mobilityModel)
            {
                std::cout << mobilityModel->GetInstanceTypeId().GetName() << " (";
                Vector position = mobilityModel->GetPosition();
                Vector velocity = mobilityModel->GetVelocity();
                std::cout << "Pos: [" << position.x << ", " << position.y << ", " << position.z << "]" << ", ";
                std::cout << "Vel: [" << velocity.x << ", " << velocity.y << ", " << velocity.z << "])" << std::endl;
            }
            else
            {
                std::cout << "No MobilityModel" << std::endl;
            }
        }
    }

    // Simulation
    Simulator::Stop(Seconds(5.0));

    Simulator::Run();
    Simulator::Destroy();

    return 0;
}