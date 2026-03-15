./ns3 run "ns3sionna-example-sionna-spectrum-compare \
  --propModel=friis --nSta=5 --simTime=2 --seed=1 \
  --writePlacementsCsv=placements.csv \
  --outPrefix=gen --dumpCfr=0"

  python contrib/sionna/model/ns3sionna/ns3unet_spectrum.py \
  --model_folder $ENV_ROOT --single_run --est_csi

  ./ns3 run "ns3sionna-example-sionna-spectrum-compare \
  --propModel=sionna --environment=$ENV_XML_REL \
  --placementsCsv=placements.csv \
  --outPrefix=rt --dumpCfr=1 --dumpFullCfr=1 --dumpMaxPerLink=1 --txPowerDbm=10"


  python contrib/sionna/model/ns3sionna/ns3unet_spectrum.py \
  --model_folder $ENV_ROOT --single_run --est_csi \
  --use_unet --unet_run $UNET_RUN --unet_device cuda:0


  ./ns3 run "ns3sionna-example-sionna-spectrum-compare \
  --propModel=sionna --environment=$ENV_XML_REL \
  --placementsCsv=placements.csv \
  --outPrefix=unet --dumpCfr=1 --dumpFullCfr=1 --dumpMaxPerLink=1"

  ./ns3 run "ns3sionna-example-sionna-spectrum-compare \
  --propModel=friis \
  --placementsCsv=placements.csv \
  --outPrefix=friis --dumpCfr=0 --txPowerDbm=10"

  python plot_walls_and_nodes.py \
  --xml $ENV_ROOT/$ENV_XML_REL \
  --placementsCsv placements.csv \
  --out overlay.png

  ./ns3 run "ns3sionna-example-sionna-mobileflow \
  --plModel=sionna \
  --server=tcp://localhost:5555 \
  --environment=seed1697/scene.xml \
  --placements=/home/matth/sionna_dev2/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/worldbuilding/seed1697/placements.csv \
  --txPowerDbm=30 \
  --enableMobility=1 --mobMode=wall \
  --outNodeTsCsv=ts.csv"

  ./ns3 run "ns3sionna-example-sionna-traceflow \
  --plModel=sionna \
  --server=tcp://localhost:5555 \
  --environment=seed1697/scene.xml \
  --placements=/home/matth/sionna_dev2/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/worldbuilding/seed1697/placements.csv \
  --txPowerDbm=10 \
  --enableMobility=1 --mobMode=wall \
  --samplePeriod=0.05 \
  --pokeSionnaPositions=1 \
  --outNodeTsCsv=mobility_trace_sionna.csv \
  --outFlowCsv=sionna_flow.csv --outPairCsv=sionna_pair.csv"


./ns3 run "ns3sionna-example-sionna-traceflow \
  --plModel=sionna \
  --server=tcp://localhost:5555 \
  --environment=seed1697/scene.xml \
  --placements=/home/matth/sionna_dev2/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/worldbuilding/seed1697/placements.csv \
  --txPowerDbm=30 \
  --enableMobility=0 \
  --pokeSionnaPositions=0 \
  --samplePeriod=0.05 \
  --outNodeTsCsv=mobility_trace_sionna.csv \
  --outFlowCsv=sionna_flow.csv --outPairCsv=sionna_pair.csv"

./ns3 run "ns3sionna-example-sionna-traceflow \
  --plModel=sionna \
  --server=tcp://localhost:5555 \
  --environment=seed1697/scene.xml \
  --placements=/home/matth/sionna_dev2/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/worldbuilding/seed1697/placements.csv \
  --txPowerDbm=10 \
  --enableMobility=0 \
  --pokeSionnaPositions=0 \
  --samplePeriod=0.05 \
  --outNodeTsCsv=mobility_trace_unet.csv \
  --mobilityTraceIn=mobility_trace_sionna.csv \
  --outFlowCsv=unet_flow.csv --outPairCsv=unet_pair.csv"

  ./ns3 run "ns3sionna-example-sionna-traceflow \
  --plModel=sionna \
  --server=tcp://localhost:5555 \
  --environment=seed1697/scene.xml \
  --placements=/home/matth/sionna_dev2/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/worldbuilding/seed1697/placements.csv \
  --txPowerDbm=10 \
  --enableMobility=0 \
  --pokeSionnaPositions=0 \
  --samplePeriod=0.05 \
  --outNodeTsCsv=mobility_trace_cost231.csv \
  --mobilityTraceIn=mobility_trace_sionna.csv \
  --outFlowCsv=cost231_flow.csv --outPairCsv=cost231_pair.csv"

  ./ns3 run "ns3sionna-example-sionna-traceflow \
  --plModel=friis \
  --placements=/home/matth/sionna_dev2/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/worldbuilding/seed1697/placements.csv \
  --mobilityTraceIn=mobility_trace_sionna.csv \
  --enableMobility=0 \
  --txPowerDbm=10 \
  --outNodeTsCsv=mobility_trace_friis.csv \
  --samplePeriod=0.05 \
  --outFlowCsv=friis_flow.csv --outPairCsv=friis_pair.csv"