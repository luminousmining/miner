# Global variables
VERSION=0.1
PROJECT=luminousminer
EXE=miner

# Delete folder
rm -rf ${PROJECT}

# Create folder
mkdir ${PROJECT}

# Setup folder
cp hiveos/h-config.sh ${PROJECT}/
cp hiveos/h-manifest.sh ${PROJECT}/
cp hiveos/h-run.sh ${PROJECT}/
cp hiveos/h-stats.sh ${PROJECT}/
cp bin/miner ${PROJECT}/
cp -r bin/Release/kernel ${PROJECT}/

# Zip folder
tar czvf ${PROJECT}-${VERSION}_hiveos.tar.gz ${PROJECT}
