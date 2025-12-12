/*
 * Copyright (c) 2024 Yannik Pilz
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Yannik Pilz <y.pilz@campus.tu-berlin.de>
 */

#include "sionna-propagation-loss-model.h"

#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SionnaPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED(SionnaPropagationLossModel);

TypeId
SionnaPropagationLossModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SionnaPropagationLossModel")
            .SetParent<PropagationLossModel>()
            .SetGroupName("Propagation")
            .AddConstructor<SionnaPropagationLossModel>();
    return tid;
}

SionnaPropagationLossModel::SionnaPropagationLossModel()
    : m_propagationCache(nullptr)
{
}

SionnaPropagationLossModel::~SionnaPropagationLossModel()
{
}

void
SionnaPropagationLossModel::SetPropagationCache(Ptr<SionnaPropagationCache> propagationCache)
{
    m_propagationCache = propagationCache;
}

double
SionnaPropagationLossModel::DoCalcRxPower(double txPowerDbm,
                                          Ptr<MobilityModel> a,
                                          Ptr<MobilityModel> b) const
{
    NS_ASSERT_MSG(m_propagationCache, "SionnaPropagationLossModel must have a SionnaPropagationCache.");
    return (txPowerDbm - m_propagationCache->GetPropagationLoss(a, b, txPowerDbm));
}

int64_t
SionnaPropagationLossModel::DoAssignStreams(int64_t stream)
{
    return 0;
}

} // namespace ns3
