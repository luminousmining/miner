#include <gtest/gtest.h>

#include <algo/algo_type.hpp>


struct AlgoTypeTest : public testing::Test
{
    AlgoTypeTest() = default;
    ~AlgoTypeTest() = default;
};


TEST_F(AlgoTypeTest, stringToEnumSuccess)
{
    EXPECT_EQ(algo::toEnum("sha256"),       algo::ALGORITHM::SHA256);
    EXPECT_EQ(algo::toEnum("ethash"),       algo::ALGORITHM::ETHASH);
    EXPECT_EQ(algo::toEnum("etchash"),      algo::ALGORITHM::ETCHASH);
    EXPECT_EQ(algo::toEnum("progpow"),      algo::ALGORITHM::PROGPOW);
    EXPECT_EQ(algo::toEnum("progpowz"),     algo::ALGORITHM::PROGPOW);
    EXPECT_EQ(algo::toEnum("kawpow"),       algo::ALGORITHM::KAWPOW);
    EXPECT_EQ(algo::toEnum("meowpow"),      algo::ALGORITHM::MEOWPOW);
    EXPECT_EQ(algo::toEnum("firopow"),      algo::ALGORITHM::FIROPOW);
    EXPECT_EQ(algo::toEnum("evrprogpow"),   algo::ALGORITHM::EVRPROGPOW);
    EXPECT_EQ(algo::toEnum("progpow-quai"), algo::ALGORITHM::PROGPOWQUAI);
    EXPECT_EQ(algo::toEnum("progpow-z"),    algo::ALGORITHM::PROGPOWZ);
    EXPECT_EQ(algo::toEnum("autolykosv2"),  algo::ALGORITHM::AUTOLYKOS_V2);
    EXPECT_EQ(algo::toEnum("blake3"),       algo::ALGORITHM::BLAKE3);

    EXPECT_EQ(algo::toEnum(""),             algo::ALGORITHM::UNKNOW);
    EXPECT_EQ(algo::toEnum("unknow"),       algo::ALGORITHM::UNKNOW);
    EXPECT_EQ(algo::toEnum("!![]{}@#"),     algo::ALGORITHM::UNKNOW);
}


TEST_F(AlgoTypeTest, stringToEnumFail)
{
    EXPECT_NE(algo::toEnum("ethash"),       algo::ALGORITHM::SHA256);
    EXPECT_NE(algo::toEnum("etchash"),      algo::ALGORITHM::ETHASH);
    EXPECT_NE(algo::toEnum("progpow"),      algo::ALGORITHM::ETCHASH);
    EXPECT_NE(algo::toEnum("progpowz"),     algo::ALGORITHM::ETCHASH);
    EXPECT_NE(algo::toEnum("kawpow"),       algo::ALGORITHM::PROGPOW);
    EXPECT_NE(algo::toEnum("firopow"),      algo::ALGORITHM::KAWPOW);
    EXPECT_NE(algo::toEnum("firopow"),      algo::ALGORITHM::MEOWPOW);
    EXPECT_NE(algo::toEnum("evrprogpow"),   algo::ALGORITHM::FIROPOW);
    EXPECT_NE(algo::toEnum("progpow-quai"), algo::ALGORITHM::EVRPROGPOW);
    EXPECT_NE(algo::toEnum("progpow-z"),    algo::ALGORITHM::PROGPOW);
    EXPECT_NE(algo::toEnum("autolykosv2"),  algo::ALGORITHM::EVRPROGPOW);
    EXPECT_NE(algo::toEnum("sha256"),       algo::ALGORITHM::AUTOLYKOS_V2);
    EXPECT_NE(algo::toEnum("autolykosv2"),  algo::ALGORITHM::BLAKE3);
}
