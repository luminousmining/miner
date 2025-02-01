#include <algo/algo_type.hpp>


std::string algo::toString(algo::ALGORITHM const algorithm)
{
    switch (algorithm)
    {
        case algo::ALGORITHM::SHA256:       { return "sha256";       }
        case algo::ALGORITHM::ETHASH:       { return "ethash";       }
        case algo::ALGORITHM::ETCHASH:      { return "etchash";      }
        case algo::ALGORITHM::PROGPOW:      { return "progpow";      }
        case algo::ALGORITHM::PROGPOWQUAI:  { return "progpow-quai"; }
        case algo::ALGORITHM::PROGPOWZ:     { return "progpowz";     }
        case algo::ALGORITHM::KAWPOW:       { return "kawpow";       }
        case algo::ALGORITHM::MEOWPOW:      { return "meowpow";      }
        case algo::ALGORITHM::FIROPOW:      { return "firopow";      }
        case algo::ALGORITHM::EVRPROGPOW:   { return "evrprogpow";   }
        case algo::ALGORITHM::AUTOLYKOS_V2: { return "autolykosv2";  }
        case algo::ALGORITHM::BLAKE3:       { return "blake3";       }
        case algo::ALGORITHM::UNKNOW:       { return "unknow";       }
    }

    return "";
}


algo::ALGORITHM algo::toEnum(
    std::string const& algo)
{
    if      (algo == "sha256")       { return algo::ALGORITHM::SHA256;       }
    else if (algo == "ethash")       { return algo::ALGORITHM::ETHASH;       }
    else if (algo == "etchash")      { return algo::ALGORITHM::ETCHASH;      }
    else if (algo == "progpow")      { return algo::ALGORITHM::PROGPOW;      }
    else if (algo == "progpow-quai") { return algo::ALGORITHM::PROGPOWQUAI;  }
    else if (algo == "progpowz")     { return algo::ALGORITHM::PROGPOWZ;     }
    else if (algo == "kawpow")       { return algo::ALGORITHM::KAWPOW;       }
    else if (algo == "meowpow")      { return algo::ALGORITHM::MEOWPOW;      }
    else if (algo == "firopow")      { return algo::ALGORITHM::FIROPOW;      }
    else if (algo == "evrprogpow")   { return algo::ALGORITHM::EVRPROGPOW;   }
    else if (algo == "autolykosv2")  { return algo::ALGORITHM::AUTOLYKOS_V2; }
    else if (algo == "blake3")       { return algo::ALGORITHM::BLAKE3;       }

    return algo::ALGORITHM::UNKNOW;
}
