build_lib_example(
    NAME ns3sionna-example-ns3
    SOURCE_FILES example-ns3.cc
    LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

build_lib_example(
        NAME ns3sionna-example-sionna
        SOURCE_FILES example-sionna.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

build_lib_example(
        NAME ns3sionna-example-mobility-sionna
        SOURCE_FILES example-mobility-sionna.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

build_lib_example(
        NAME ns3sionna-example-munich-sionna
        SOURCE_FILES example-munich-sionna.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

build_lib_example(
        NAME ns3sionna-example-munich-mobility-sionna
        SOURCE_FILES example-munich-mobility-sionna.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

build_lib_example(
        NAME ns3sionna-example-sionna-spectrummodel
        SOURCE_FILES example-sionna-spectrummodel.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

build_lib_example(
        NAME ns3sionna-example-optimized-sionna
        SOURCE_FILES example-optimized-sionna.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

build_lib_example(
        # todo: not yet implemented
        NAME ns3sionna-example-sionna-snr
        SOURCE_FILES example-sionna-snr.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

# sensing examples

build_lib_example(
        NAME ns3sionna-example-sionna-sensing
        SOURCE_FILES example-sionna-sensing.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

build_lib_example(
        NAME ns3sionna-example-sionna-sensing-mobile
        SOURCE_FILES example-sionna-sensing-mobile.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)


# performance & benchmark scripts

build_lib_example(
        NAME ns3sionna-performance-ns3
        SOURCE_FILES performance-ns3.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

build_lib_example(
        NAME ns3sionna-performance-sionna
        SOURCE_FILES performance-sionna.cc
        LIBRARIES_TO_LINK ${libsionna} ${libwifi} ${libbuildings} ${libinternet} ${libapplications}
)

