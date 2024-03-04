
set(STRATUM_HEADERS
autolykos_v2.hpp
etchash.hpp
ethash.hpp
evrprogpow.hpp
job_info.hpp
firopow.hpp
kawpow.hpp
progpow.hpp
sha256.hpp
smart_mining.cpp
stratum.hpp
stratums.hpp
)

set(STRATUM_SOURCES
autolykos_v2.cpp
etchash.cpp
ethash.cpp
evrprogpow.cpp
job_info.cpp
firopow.cpp
kawpow.cpp
progpow.cpp
sha256.cpp
smart_mining.cpp
stratum.cpp
stratums.cpp
)

target_sources(${MINER_EXE} PUBLIC
${STRATUM_HEADERS}
${STRATUM_SOURCES}
)
